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

from types import SimpleNamespace

import numpy as np

cfg = SimpleNamespace(**{})

## Callbacks
# Checkpointing
cfg.filename = "{epoch}-{step}-{val_loss:.2f}-{other_metric:.2f}"
cfg.monitor = "best_score"
cfg.save_last = True
cfg.save_top_k = 3
cfg.mode = "max"
cfg.auto_insert_metric_name = True
cfg.save_weights_only = False
cfg.every_n_train_steps = 24
cfg.train_time_interval = None
cfg.every_n_epochs = 1
cfg.save_on_train_epoch_end = False
cfg.enable_version_counter = True
# LR monitoring
cfg.logging_interval = "step"
cfg.log_momentum = True
cfg.log_weight_decay = True


## Dataset/DataLoader
cfg.batch_size = 4
cfg.batch_size_val = 2
cfg.drop_last = False
cfg.pin_memory = False
cfg.val_pin_memory = False
cfg.test_pin_memory = False
cfg.num_workers = 0
cfg.shuffle = False
cfg.train_sub_epochs = 2
cfg.val_sub_epochs = 1
cfg.prefetch_factor = None


## Metrics
cfg.dt_multiplier = 1
cfg.max_th = 0.5
cfg.motor_radius = 500
cfg.score_beta = 2


# model
cfg.backbone = "resnet10"
cfg.class_weights = np.array([256, 1])
cfg.in_channels = 1
cfg.mixup_p = 0.0
cfg.mixup_beta = 1.0
cfg.n_classes = 1
cfg.pretrained = True
cfg.spatial_dims = 3
cfg.virt_sub_batch_size = -1  # split N, C, D, H, W from the loader in sub_batchs
cfg.virt_eval_sub_batch_size = -1
cfg.lvl_weights = np.array([0, 0, 0, 1])


## Optimizer
cfg.lr = 1e-5
cfg.optimizer = "SGD"
cfg.sgd_momentum = 0.8
cfg.sgd_nesterov = True
cfg.weight_decay = 0.0


# Paths
cfg.data_folder = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
cfg.df_path = (
    "/kaggle/input/drx75c-byu-ds01-folded-tomograms/byu_folded_tomograms_seed57.csv"
)
cfg.submission_dir = "/kaggle/working"


## Scheduler
cfg.end_lambda = 1e-5
cfg.milestones = 10
cfg.schedule = "multistep"
cfg.warmup = 100  # Warmup steps, not epochs


## Tracking
cfg.track_grad_norm = True
cfg.track_weight_norm = True
cfg.write_xla_metrics = True
cfg.async_closure = False
cfg.log_grad_step = 24

## L.Trainer
# Optimisation
cfg.accumulate_grad_batches = 1
cfg.gradient_clip_algorithm = "norm"
cfg.gradient_clip_val = 2
cfg.precision = "16-mixed"  # TODO: "bf16-mixed" ?
cfg.sync_batchnorm = True
# cfg.downcast_bf16 = True
# Device/Strategy
cfg.accelerator = "auto"
cfg.devices = "auto"
cfg.num_nodes = 1
cfg.strategy = "auto"
cfg.use_distributed_sampler = True
cfg.reload_dataloaders_every_n_epochs = 0
cfg.benchmark = None
# Debugging
cfg.barebones = False
cfg.detect_anomaly = False
cfg.fast_dev_run = 10
cfg.num_sanity_val_steps = 2
cfg.overfit_batches = 0.0
cfg.profiler = None
# Logging/Chepointing
cfg.default_root_dir = "/kaggle/working/"
cfg.enable_checkpointing = True
cfg.enable_model_summary = False
cfg.enable_progress_bar = True
cfg.logger = True
cfg.log_every_n_steps = 24
cfg.model_registry = None
cfg.callbacks = None
# Training
cfg.check_val_every_n_epoch = 1
cfg.inference_mode = True
cfg.limit_predict_batches = 1.0
cfg.limit_test_batches = 1.0
cfg.limit_train_batches = 1.0
cfg.max_epochs = 1000
cfg.max_steps = -1
cfg.max_time = None
cfg.min_steps = None
cfg.val_check_interval = 1.0
# Determism
cfg.deterministic = "warn"


## Training
cfg.best_th = 0.5
cfg.epochs_step = None
cfg.fold = 0
cfg.last_score = 0.0
cfg.milestones = None
cfg.overfit_tomos = [
    "tomo_00e463",
    "tomo_226cd8",
    "tomo_08bf73",
    "tomo_1ab322",
    "tomo_ae347a",
    "tomo_e22370",
    "tomo_2daaee",
    "tomo_183270",
]  # 6, 10, 0, 1, 2, 3, 4, 0
cfg.seed = 57

## PP
cfg.nms_radius = 16

## TRANSFORMS
cfg.gamma = 2
cfg.new_size = (300, 500, 500)
cfg.roi_size = [96, 96, 96]
cfg.sub_batch_size = 2
