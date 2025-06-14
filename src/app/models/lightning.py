# MIT License
#
# Copyright (c) 2024, Yebouet Cédrick-Armel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from collections import Counter
from typing import Any

import lightning.pytorch as L
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.utilities import dim_zero_cat

from app.metrics import BYUFbeta
from app.processings import post_process_pipeline
from app.utils import get_optimizer, get_scheduler, initialize_weights

from .models import Net


class LNet(L.LightningModule):
    """Lightning wrapper for models.Net"""

    def __init__(self, cfg: "DictConfig") -> None:
        """Init method called once"""
        super().__init__()
        self.cfg = cfg
        self.model = Net(cfg)
        self.validation_step_outputs: "list[torch.Tensor]" = []

    def on_fit_start(self) -> "None":
        """Called at the very beginning of fit."""
        if torch.distributed.is_initialized():
            if not hasattr(self, "gloo_group"):
                self.gloo_groupg = torch.distributed.new_group(backend="gloo")
                self.score_metric = BYUFbeta(
                    self.cfg,
                    process_group=self.gloo_groupg,
                    compute_on_cpu=self.cfg.byu_metric.compute_on_cpu,
                    dist_sync_on_step=self.cfg.byu_metric.dist_sync_on_step,
                    sync_on_compute=self.cfg.byu_metric.sync_on_compute,
                )
        else:
            self.score_metric = BYUFbeta(
                self.cfg,
                compute_on_cpu=self.cfg.byu_metric.compute_on_cpu,
                dist_sync_on_step=self.cfg.byu_metric.dist_sync_on_step,
                sync_on_compute=self.cfg.byu_metric.sync_on_compute,
            )

    def setup(self, stage: "str") -> "None":
        """Called at the beginning of each stage in oder to build model dynamically."""
        if stage == "fit":
            self.model.backbone.decoder = self.model.backbone.decoder.apply(
                lambda m: initialize_weights(cfg=self.cfg, module=m)
            )
            if self.cfg.freeze_encoder:
                for param in self.model.backbone.encoder.parameters():
                    param.requires_grad = False

        self.cfg.lr *= self.trainer.world_size
        stepping_batches = self.trainer.estimated_stepping_batches
        self.training_steps = (
            stepping_batches * self.cfg.max_epochs * self.trainer.world_size
        )

    def forward(self, batch: "dict[str, Any]") -> "torch.Tensor":
        return self.model(batch)

    def configure_optimizers(self) -> "dict[str, Any] | Optimizer":
        """Return the optimizer and an optionnal lr_scheduler"""
        optimizer: "Optimizer" = get_optimizer(self.cfg, self.model)
        scheduler: "LRScheduler" = get_scheduler(
            cfg=self.cfg, optimizer=optimizer, training_steps=self.training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler, interval="step", frequency=1, name="lr"
            ),
        )

    def training_step(
        self, batch: "dict[str, Any]", batch_idx: "int"
    ) -> "torch.Tensor":
        output_dict = self(batch)
        loss: "torch.Tensor" = output_dict["loss"]
        log_dict: "dict[str, torch.Tensor]" = dict(
            train_loss=loss,
            train_bg_dice=output_dict["dice"][0],
            train_fg_dice=output_dict["dice"][1],
        )
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        params = [p for p in self.parameters() if p.grad is not None]
        if len(params) == 0:
            total_norm = torch.tensor(0.0)
        else:
            total_norm = torch.norm(
                torch.cat([p.detach().view(-1) for p in params]),
                p=self.cfg.grad_norm_type,
            )
        self.log(
            "weight_norm",
            total_norm,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )

    def validation_step(
        self, batch: "dict[str, Any]", batch_idx: "int"
    ) -> "torch.Tensor":
        """Operates on a single batch of data from the validation set"""
        zyx = batch["zyx"]
        output_dict = self(batch)
        loss: "torch.Tensor" = output_dict["loss"]
        preds: "torch.Tensor" = post_process_pipeline(self.cfg, output_dict)
        log_dict: "dict[str, torch.Tensor]" = dict(
            val_loss=loss,
            val_bg_dice=output_dict["dice"][0],
            val_fg_dice=output_dict["dice"][1],
        )
        self.validation_step_outputs.append(preds)
        self.score_metric.update(preds, zyx)
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        return preds

    def on_validation_epoch_end(self) -> "None":
        """Called after the epoch ends to agg preds and logging"""
        metrics = self.score_metric.compute()
        preds = dim_zero_cat(self.validation_step_outputs)

        preds = preds.cpu()

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        torch.save(
            preds,
            os.path.join(
                self.cfg.default_root_dir,
                f"val_epoch_{self.current_epoch}_end_step{self.global_step}_rank{self.global_rank}.pt",
            ),
        )
        self.score_metric.reset()
        self.validation_step_outputs.clear()

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: "float | None" = None,
        gradient_clip_algorithm: "Any | None" = None,
    ) -> "None":
        """Gradient clipping and tracking before/afater clipping"""
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            total_norm_before = torch.tensor(0.0)
        else:
            total_norm_before = torch.norm(
                torch.cat([g.detach().view(-1) for g in grads]),
                p=self.cfg.grad_norm_type,
            )

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            total_norm_after = torch.tensor(0.0)
        else:
            total_norm_after = torch.norm(
                torch.cat([g.detach().view(-1) for g in grads]),
                p=self.cfg.grad_norm_type,
            )
              
        log_dict: "dict[str, torch.Tensor]" = dict(
            grad_norm=total_norm_before, clip_grad_norm=total_norm_after
        )

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )
