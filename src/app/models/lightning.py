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
from typing import Any

import lightning as L
import torch

from omegaconf import DictConfig

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.utilities import dim_zero_cat
from app.utils import get_multistep_schedule_with_warmup, get_optimizer
from app.metrics import BYUFbeta
from app.processings import post_process_pipeline
from .models import Net


class LNet(L.LightningModule):
    """Lightning wrapper for models.Net"""

    def __init__(self, cfg: "DictConfig") -> None:
        """Init method called once"""
        super().__init__()
        self.cfg = cfg
        self.model = Net(cfg)
        self.score_metric = BYUFbeta(
            self.cfg,
            compute_on_cpu=True,
            dist_sync_on_step=False,
        )
        self.validation_step_outputs: "list[torch.Tensor]" = []

    def setup(self, stage: "str") -> "None":
        """Called at the beginning of each stage in oder to build model dynamically."""
        if stage == "fit" and self.cfg.pretrained:
            for param in self.model.backbone.encoder.parameters():
                param.requires_grad = False

    def forward(self, batch: "dict[str, Any]") -> "torch.Tensor":
        return self.model(batch)

    def configure_optimizers(self) -> "dict[str, Any]":
        """Return the optimizer and an optionnal lr_scheduler"""
        training_steps: "int" = self.trainer.num_training_batches * self.cfg.max_epochs
        optimizer: "Optimizer | None" = get_optimizer(self.cfg, self.model)
        scheduler: LRScheduler = get_multistep_schedule_with_warmup(
            optimizer,
            warmup_steps=self.cfg.warmup,
            m=self.cfg.milestones,
            training_steps=training_steps,
            end_lambda=self.cfg.end_lambda,
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
        loss = output_dict["loss"]
        if self.trainer.is_global_zero:
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=True
            )
        return loss

    def validation_step(
        self, batch: "dict[str, Any]", batch_idx: "int"
    ) -> "torch.Tensor":
        """Operates on a single batch of data from the validation set"""
        zyx = batch["zyx"]
        output_dict = self(batch)
        loss = output_dict["loss"]
        preds = post_process_pipeline(self.cfg, output_dict)
        self.validation_step_outputs.append(preds)
        self.score_metric.update(preds, zyx)
        self.log(
            "val_loss",
            loss,
            on_step=True,
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
        preds = dim_zero_cat(self.all_gather(preds))  # preds should have the same size across all device/processes
        preds = preds.cpu()
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=False,
        )
        
        self.score_metric.reset()
        self.validation_step_outputs.clear()

        if self.trainer.is_global_zero:
            torch.save(
                preds,
                os.path.join(
                    self.cfg.default_root_dir,
                    f"val_epoch_{self.current_epoch}_end_step{self.global_step}.pt",
                ),
            )


    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: "float | None" = None,
        gradient_clip_algorithm: "Any | None" = None,
    ) -> "None":
        """Gradient clipping and tracking before/afater clipping"""
        # TODO: if self.trainer.global_step % self.cfg.log_every_n_steps == 0:
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        total_norm_before = torch.norm(
            torch.stack(
                [torch.norm(g.detach(), self.cfg.grad_norm_type) for g in grads]
            ),
            2,
        )
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        total_norm_after = torch.norm(
            torch.stack(
                [torch.norm(g.detach(), self.cfg.grad_norm_type) for g in grads]
            ),
            2,
        )
        if self.trainer.is_global_zero:
            self.log(
                "grad_norm",
                total_norm_before,
                on_step=True,
                on_epoch=True,
                logger=True,
                rank_zero_only=True,
                prog_bar=False,
            )
            self.log(
                "clip_grad_norm",
                total_norm_after,
                on_step=True,
                on_epoch=True,
                logger=True,
                rank_zero_only=True,
                prog_bar=False,
            )


# TODO: compute metric on CPU: default false -> only for list states
# TODO: sync on compute True, default True
# TODO: dist_sync_on_step commencer avec False, puis True pour garder la meilleure option
# TODO: metric states behave as buffers
# TODO: dist_reduce_fx="cat"
# TODO: reduce_fx: Reduction function over step values for end of epoch. Uses torch.mean() by default and is not applied when a torchmetrics.Metric is logged.
# TODO: self.training_step_outputs.append(loss)
