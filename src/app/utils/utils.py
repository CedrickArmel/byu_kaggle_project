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
import random
from collections import defaultdict
from glob import glob
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.profilers import Profiler, PyTorchProfiler, XLAProfiler
from omegaconf import DictConfig
from torch.nn.init import (
    calculate_gain,
    kaiming_normal_,
    kaiming_uniform_,
    xavier_normal_,
    xavier_uniform_,
)
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from app.data import BYUCustomDataset


def collate_fn(batch: "list[dict[str, Any]]") -> "dict[str, Any]":
    """Collate function to batch a list of dictionaries into a single dictionary."""
    batch_data = defaultdict(lambda: [])
    str_keys = []
    tensor_keys = []
    for b in batch:
        s = len(b["input"])
        for key, value in b.items():
            if not isinstance(value, torch.Tensor):
                batch_data[key].extend([value] * s)
                str_keys.append(key)
            else:
                batch_data[key].append(value)
                tensor_keys.append(key)
    batch_dict: "dict[str, Any]" = {key: batch_data[key] for key in str_keys}
    for key in tensor_keys:
        batch_dict[key] = (
            torch.cat(batch_data[key])
            if key in ["input", "target", "zyx"]
            else torch.stack(batch_data[key])
        )
    return batch_dict


def create_milestones(steps: "int", m: "int") -> "list[int]":
    """returns a list of milestones for the given number of steps and m."""
    g = int(steps // m)
    milestones = []
    for i in range(1, m + 1):
        milestones += [i] * g
    return milestones


def get_callbacks(cfg: "DictConfig") -> "tuple[Callback, ...]":
    chckpt_cb = ModelCheckpoint(**cfg.callbacks_args.checkpoint)
    lr_cb = LearningRateMonitor(**cfg.callbacks_args.lr_monitor)
    return chckpt_cb, lr_cb


def get_data(cfg: "DictConfig", mode: "str" = "fit") -> "tuple[pd.DataFrame,...]":
    """
    Loads and splits a dataset into training and validation DataFrames based on the provided configuration.
    Args:
        cfg (DictConfig): A configuration object containing the following attributes:
            - df_path (str): Path to the CSV file containing the dataset.
            - fold (int): Fold index for splitting the data. If -1, a default split is used.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training DataFrame and validation DataFrame.
    """
    if mode not in ["fit", "test"]:
        raise ValueError("mode argument must be one of train, validation or test!")
    if cfg.manual_overfit:
        overfit_samples = cfg.overfit_tomos[: cfg.batch_size]
        df = df = pd.read_csv(cfg.df_path)
        train_df = df[df.tomo_id.isin(overfit_samples)]
        val_df = df[df.fold == 0]
        data = (train_df, val_df)

    elif mode == "fit":
        df = pd.read_csv(cfg.df_path)
        if cfg.fold > -1:
            train_df = df[df.fold != cfg.fold]
            val_df = df[df.fold == cfg.fold]
        else:
            train_df = df[df.fold != 0]
            val_df = df[df.fold == 0]
        data = (train_df, val_df)

    elif mode == "test":
        test_tomo_id = sorted(
            [
                path.split("/")[-1]
                for path in glob(os.path.join(cfg.data_folder, "test", "**"))
            ]
        )
        num_ids = list(range(0, len(test_tomo_id)))
        data = pd.DataFrame(dict(tomo_id=test_tomo_id, id=num_ids))
    return data


def get_data_loader(
    cfg: "DictConfig", df: "pd.DataFrame", mode: "str"
) -> "DataLoader | tuple[DataLoader, ...] | None":
    """
    Creates and returns PyTorch DataLoader objects for training, validation, or testing
    based on the provided configuration.
    Args:
        cfg (DictConfig): A configuration object containing the following attributes:
            - train (bool): Whether to create DataLoaders for training and validation.
            - test (bool): Whether to create a DataLoader for testing.
            - batch_size (int): Batch size for training DataLoader.
            - batch_size_val (int): Batch size for validation and test DataLoaders.
            - num_workers (int): Number of worker threads for data loading.
            - pin_memory (bool): Whether to pin memory for training DataLoader.
            - val_pin_memory (bool): Whether to pin memory for validation DataLoader.
            - test_pin_memory (bool): Whether to pin memory for test DataLoader.
            - shuffle (bool): Whether to shuffle the training data.
            - drop_last (bool): Whether to drop the last incomplete batch in training.
            - prefetch_factor (int): Number of batches to prefetch.
    Returns:
        DataLoader
    """
    g = get_seeded_generator(cfg)
    dataset = BYUCustomDataset(cfg, df=df, mode=mode)
    args = cfg.dataloader_args.train if mode == "train" else cfg.dataloader_args.eval
    loader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
        **args,
    )
    return loader


def get_optimizer(cfg: "DictConfig", model: "torch.nn.Module") -> "optim.Optimizer":
    """
    Creates and returns an optimizer based on the configuration provided.
    Args:
        cfg (DictConfig): A configuration object containing the optimizer settings.
            Expected attributes:
                - optimizer (str): The type of optimizer to use. Supported values are
                  "Adam", "AdamW_plus", "AdamW", and "SGD".
                - lr (float): The learning rate for the optimizer.
                - weight_decay (float): The weight decay (L2 regularization) factor.
                - sgd_momentum (float, optional): Momentum factor for SGD (required if optimizer is "SGD").
                - sgd_nesterov (bool, optional): Whether to enable Nesterov momentum for SGD (required if optimizer is "SGD").
        model (torch.nn.Module): The model whose parameters will be optimized.
    Returns:
        optim.Optimizer | None: The configured optimizer instance, or None if the optimizer type is not supported.
    Notes:
        - For "AdamW_plus", the parameters are divided into two groups: those with weight decay and those without.
        - The "AdamW_plus" optimizer applies weight decay only to parameters not in the `no_decay` list, which includes
          "bias" and "LayerNorm.bias".
    """
    params_ = model.parameters()
    if cfg.optimizer == "Adam":
        optimizer: "optim.Optimizer" = optim.Adam(
            params_, lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    elif cfg.optimizer == "AdamW_plus":
        nparams_ = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        params = [
            {
                "params": [
                    param
                    for name, param in nparams_
                    if (not any(nd in name for nd in no_decay))
                ],
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in nparams_
                    if (any(nd in name for nd in no_decay))
                ],
                "lr": cfg.lr,
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(params, lr=cfg.lr)

    elif cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(params_, lr=cfg.lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            params_,
            lr=cfg.lr,
            momentum=cfg.sgd_momentum,
            nesterov=cfg.sgd_nesterov,
            weight_decay=cfg.weight_decay,
        )
    return optimizer


def get_profiler(cfg: "DictConfig") -> "Profiler | None":
    """Returns a suitable profiler for the used accelerator"""
    if cfg.profiler:
        if cfg.accelerator == "tpu":
            profiler = XLAProfiler(port=cfg.profiler_port)
        else:
            profiler = PyTorchProfiler(
                filename=cfg.prof_filename, emit_nvtx=cfg.emit_nvtx
            )
    else:
        profiler = None
    return profiler


def get_seeded_generator(cfg: "DictConfig") -> "torch.Generator":
    NP_MAX = np.iinfo(np.uint32).max
    MAX_SEED = NP_MAX + 1
    seed = cfg.seed % MAX_SEED
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_scheduler(
    cfg: "DictConfig", optimizer: "optim.Optimizer", training_steps: "int"
) -> "LRScheduler | None":
    """
    Creates and returns a learning rate scheduler based on the provided configuration.
    Returns:
        LRScheduler: The configured learning rate scheduler or None if no valid scheduler is specified.
    """
    if cfg.schedule not in ["multistep", "cosine", "cosine_wr", "constant", "none"]:
        raise ValueError(
            f"Invalid schedule type: {cfg.schedule}. Supported types are 'multistep', 'cosine', 'cosine_wr', 'constant', 'none'."
        )
    if cfg.schedule == "multistep":
        steps: "int" = training_steps - cfg.warmup
        milestones = range(1, steps, (steps // cfg.milestones))
        scheduler = MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=cfg.end_lambda
        )
    elif cfg.schedule == "cosine":
        scheduler = CosineAnnealingLR(optimizer=optimizer, **cfg.cosine_args)
    elif cfg.schedule == "cosine_wr":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer, **cfg.cosine_wr_args
        )
    elif cfg.schedule == "constant":
        scheduler = ConstantLR(optimizer=optimizer, **cfg.constant_args)
    elif cfg.schedule == "none":
        if cfg.warmup > 0:
            warn(
                "Warmup is set to a value greater than 0, but `none` is provided as schedule type."
                "Considering only the linear warmup phase. Set `warmup` to 0 to disable it."
            )
        scheduler = None

    else:
        if cfg.warmup > 0:
            warn(
                "Warmup is set to a value greater than 0, but an unsupported schedule"
                f"type: {cfg.schedule} is provided by user."
                "Considering only the linear warmup phase. Set `warmup` to 0 to disable it."
            )
        scheduler = None

    if cfg.warmup > 0:
        warmup_scheduler = LinearLR(optimizer=optimizer, **cfg.linear_args)
        if scheduler is None:
            sequential: "LinearLR" = warmup_scheduler
        else:
            sequential = SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[cfg.warmup],
            )
    else:
        sequential = scheduler
    return sequential


def initialize_weights(cfg: "DictConfig", module: "torch.nn.Module") -> "None":
    """Applies to a model to init its params"""
    if isinstance(module, torch.nn.Linear):
        gain = calculate_gain(nonlinearity=cfg.init_fn_args.nonlinearity)
        if cfg.ws_init_dist == "normal":
            xavier_normal_(tensor=module.weight, gain=gain)
        elif cfg.ws_init_dist == "uniform":
            xavier_uniform_(tensor=module.weight, gain=gain)
    elif isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
        if cfg.ws_init_dist == "normal":
            kaiming_normal_(tensor=module.weight, **cfg.init_fn_args)
        elif cfg.ws_init_dist == "uniform":
            kaiming_uniform_(tensor=module.weight, **cfg.init_fn_args)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(
    seed: "int | None" = 4294967295,
    cudnn_backend: "bool" = False,
    use_deterministic_algorithms: "bool" = False,
    warn_only: "bool" = True,
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    NP_MAX = np.iinfo(np.uint32).max
    MAX_SEED = NP_MAX + 1

    if seed is None:
        seed_ = torch.default_generator.seed() % MAX_SEED
        torch.manual_seed(seed_)
    else:
        seed = int(seed) % MAX_SEED
        torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    if seed is not None and cudnn_backend:
        torch.backends.cudnn.deterministic = True  # if True, causes cuDNN to only use deterministic convolution algorithms
        torch.backends.cudnn.benchmark = False  # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest

    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(
            mode=use_deterministic_algorithms, warn_only=warn_only
        )  # Sets whether PyTorch operations must use “deterministic” algorithms
