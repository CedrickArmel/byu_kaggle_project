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

import lightning as L
from lightning.pytorch.callbacks import Callback
from omegaconf.OmegaConf import DictConfig


def lightning_trainer(cfg: "DictConfig", callbacks: "list[Callback]"):
    return L.Trainer(
        accelerator=cfg.accelerator,
        strategy=cfg.strategy,
        devices=cfg.devices,
        num_nodes=cfg.num_nodes,
        precision=cfg.precision,
        logger=cfg.logger,
        callbacks=cfg.callbacks,
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=cfg.max_epochs,
        min_epochs=cfg.min_epochs,
        max_steps=cfg.max_steps,
        min_steps=cfg.max_steps,
        max_time=cfg.max_time,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        limit_test_batches=cfg.limit_test_batches,
        limit_predict_batches=cfg.limit_predict_batches,
        overfit_batches=cfg.overfit_batches,
        val_check_interval=cfg.val_check_interval,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        log_every_n_steps=cfg.log_every_n_steps,
        enable_checkpointing=cfg.enable_checkpointing,
        enable_progress_bar=cfg.enable_progress_bar,
        enable_model_summary=cfg.enable_model_summary,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gradient_clip_val=cfg.gradient_clip_val,
        gradient_clip_algorithm=cfg.gradient_clip_algorithm,
        deterministic=cfg.deterministic,
        benchmark=cfg.benchmark,
        inference_mode=cfg.inference_mode,
        use_distributed_sampler=cfg.use_distributed_sampler,
        profiler=cfg.profiler,
        detect_anomaly=cfg.detect_anomaly,
        barebones=cfg.barebones,
        plugins=cfg.plugins,
        sync_batchnorm=cfg.sync_batchnorm,
        reload_dataloaders_every_n_epochs=cfg.reload_dataloaders_every_n_epochs,
        default_root_dir=cfg.default_root_dir,
        model_registry=cfg.model_registry,
    )
