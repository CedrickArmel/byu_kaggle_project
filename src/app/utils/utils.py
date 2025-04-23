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
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch_xla as xla
import torch_xla.test.test_utils as tu
from data.datasets import BYUCustomDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def calc_grad_norm(
    parameters: "list[torch.Tensor] | torch.Tensor", norm_type: "float" = 2.0
) -> "torch.Tensor":
    """
    Calculate the gradient norm the parameters.
    This function computes the norm of the gradients of the given parameters
    using the specified norm type. If no gradients are available, it returns 0.
    Args:
        parameters (list[torch.Tensor] | torch.Tensor): A list of tensors or a single tensor
            whose gradients will be used to compute the norm. If a tensor does not have
            a gradient, it will be ignored.
        norm_type (float, optional): The type of norm to compute (e.g., 2.0 for L2 norm).
            Defaults to 2.0.
    Returns:
        torch.Tensor: The computed gradient norm as a tensor. If no gradients are available,
            returns a tensor with value 0.0.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    norm_type = float(norm_type)
    device: "torch.device" = parameters[0].grad.device  # type: ignore[union-attr]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]  # type: ignore[union-attr]
        ),
        norm_type,
    )
    return total_norm


def calc_weight_norm(
    parameters: "list[torch.Tensor] | torch.Tensor", norm_type: "float" = 2.0
) -> "torch.Tensor":
    """
    Calculate the average normalized weight of the parameters.
    Args:
        parameters (list[torch.Tensor] | torch.Tensor):Model parameters whose norms
            are to be calculated. Tensors without gradients are ignored.
        norm_type (float, optional): The type of norm to compute. Defaults to 2.0 (Euclidean norm).

    Returns:
        torch.Tensor: The mean of the computed norms of the tensors. If no tensors with gradients
        are provided, returns a tensor with value 0.0.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    norm_type = float(norm_type)
    device: "torch.device" = parameters[0].grad.device  # type: ignore[union-attr]
    total_norm = torch.stack(
        [torch.norm(p.detach(), norm_type).to(device) for p in parameters]
    ).mean()
    return total_norm


def create_checkpoint(
    cfg: "SimpleNamespace",
    model: "torch.nn.Module",
    optimizer: "optim.Optimizer",
    scheduler: "optim.lr_scheduler.LRScheduler | None" = None,
    scaler: "torch.amp.GradScaler | None" = None,
) -> "dict[str, Any]":
    """Create a checkpoint (state) for the model, optimizer, scaler, and scheduler."""
    state_dict = deepcopy(model.state_dict())
    if cfg.save_weights_only:
        checkpoint = {"model": state_dict}
        return checkpoint

    checkpoint = {
        "model": state_dict,
        "optimizer": deepcopy(optimizer.state_dict()),
        "epoch": cfg.curr_epoch,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = deepcopy(scheduler.state_dict())
    if scaler is not None:
        checkpoint["scaler"] = deepcopy(scaler.state_dict())
    return checkpoint


def load_checkpoint(
    cfg: "SimpleNamespace",
    model: "torch.nn.Module",
    optimizer: "optim.Optimizer",
    scheduler: "optim.lr_scheduler.LRScheduler | None" = None,
    scaler: "torch.amp.GradScaler | None" = None,
) -> "tuple[torch.nn.Module, optim.Optimizer, dict[str, Any], torch.amp.GradScaler | None, int]":
    """
    Loads a training checkpoint and restores the model, optimizer, scheduler, and scaler states.

    Args:
        cfg (SimpleNamespace): Configuration object containing the path to the checkpoint file
            (accessible via `cfg.resume_from`).
        model (torch.nn.Module): The model instance to load the state dictionary into.
        optimizer (optim.Optimizer): The optimizer instance to load the state dictionary into.
        scheduler (optim.lr_scheduler.LRScheduler | None, optional): The learning rate scheduler
            instance to restore state for. Defaults to None.
        scaler (torch.amp.GradScaler | None, optional): The gradient scaler instance for mixed
            precision training to restore state for. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The model with restored state.
            - optimizer (optim.Optimizer): The optimizer with restored state.
            - scheduler_dict (dict): The state dictionary of the scheduler.
            - scaler (torch.amp.GradScaler | None): The scaler with restored state, if provided.
            - epoch (int): The epoch number at which the checkpoint was saved.
    """
    checkpoint = torch.load(cfg.resume_from, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler_dict = checkpoint["scheduler"]
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = checkpoint["epoch"]
    return model, optimizer, scheduler_dict, scaler, epoch


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
            if key in ["input", "target"]
            else torch.stack(batch_data[key])
        )
    return batch_dict


def compute_grad_metrics(
    cfg: "SimpleNamespace", model: "torch.nn.Module"
) -> "torch.Tensor":
    """
    Computes gradient and weight norm metrics for a given model.
    Args:
        cfg (SimpleNamespace): Configuration object containing the following attributes:
            - track_grad_norm (bool): Whether to compute and track the gradient norm.
            - clip_grad (float): Value for gradient clipping. If greater than 0, gradients are clipped.
            - grad_norm_type (int): Type of norm to compute for gradients (e.g., 2 for L2 norm).
            - track_weight_norm (bool): Whether to compute and track the weight norm.
            - device (torch.device): Device to which the computed metrics tensor will be moved.
        model (torch.nn.Module): The model whose gradients and weights are analyzed.
    Returns:
        torch.Tensor: A tensor containing the following metrics:
            - grad_norm (float): The norm of the gradients before clipping, or NaN if not tracked.
            - grad_norm_after_clip (float): The norm of the gradients after clipping, or NaN if not tracked.
            - weight_norm (float): The norm of the model's weights, or NaN if not tracked.
    """
    grad_norm = calc_grad_norm(model.parameters()) if cfg.track_grad_norm else torch.nan  # type: ignore[arg-type]
    if cfg.clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
    grad_norm_after_clip = (
        calc_grad_norm(model.parameters(), cfg.grad_norm_type)  # type: ignore[arg-type]
        if (cfg.clip_grad > 0 and cfg.track_grad_norm)
        else torch.nan
    )
    weight_norm = (
        calc_weight_norm(model.parameters(), cfg.grad_norm_type)  # type: ignore[arg-type]
        if cfg.track_weight_norm
        else torch.nan
    )
    return torch.tensor([grad_norm, grad_norm_after_clip, weight_norm]).to(cfg.device)


def get_data_loaders(
    cfg: "SimpleNamespace",
) -> "DataLoader | tuple[DataLoader, ...] | None":
    """
    Creates and returns PyTorch DataLoader objects for training, validation, or testing
    based on the provided configuration.
    Args:
        cfg (SimpleNamespace): A configuration object containing the following attributes:
            - train (bool): Whether to create DataLoaders for training and validation.
            - test (bool): Whether to create a DataLoader for testing.
            - train_df (DataFrame): DataFrame containing training data.
            - val_df (DataFrame): DataFrame containing validation data.
            - train_transforms (callable): Transformations to apply to training data.
            - eval_transforms (callable): Transformations to apply to validation data.
            - batch_size (int): Batch size for training DataLoader.
            - batch_size_val (int): Batch size for validation and test DataLoaders.
            - num_workers (int): Number of worker threads for data loading.
            - pin_memory (bool): Whether to pin memory for training DataLoader.
            - val_pin_memory (bool): Whether to pin memory for validation DataLoader.
            - test_pin_memory (bool): Whether to pin memory for test DataLoader.
            - shuffle (bool): Whether to shuffle the training data.
            - drop_last (bool): Whether to drop the last incomplete batch in training.
            - prefetch_factor (int): Number of batches to prefetch.
            - tpu (bool): Whether training is being performed on a TPU.
            - world_size (int): Number of processes participating in distributed training.
            - rank (int): Rank of the current process in distributed training.
    Returns:
        DataLoader | tuple[DataLoader, ...] | None:
            - A tuple containing the training and validation DataLoaders if `cfg.train` is True.
            - A single DataLoader for testing if `cfg.test` is True.
            - None if neither `cfg.train` nor `cfg.test` is True.
    """

    if cfg.train:
        train_ds = BYUCustomDataset(
            cfg, mode="train", df=cfg.train_df, aug=cfg.train_transforms
        )

        train_sampler: "DistributedSampler | None" = (
            DistributedSampler(
                train_ds,
                num_replicas=cfg.world_size,
                rank=cfg.rank,
                shuffle=cfg.shuffle,
            )
            if cfg.tpu
            else None
        )
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=cfg.pin_memory,
            sampler=train_sampler,
            shuffle=False if train_sampler else cfg.shuffle,
            drop_last=cfg.drop_last,
            prefetch_factor=cfg.prefetch_factor,
        )

        val_ds = BYUCustomDataset(
            cfg, mode="validation", df=cfg.val_df, aug=cfg.eval_transforms
        )
        val_loader = DataLoader(
            dataset=val_ds,
            batch_size=cfg.batch_size_val,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=cfg.val_pin_memory,
            prefetch_factor=cfg.prefetch_factor,
        )
        return train_loader, val_loader

    if cfg.test:
        test_ds = BYUCustomDataset(cfg, mode="test")
        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=cfg.batch_size_val,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=cfg.test_pin_memory,
            prefetch_factor=cfg.prefetch_factor,
        )
        return test_loader
    return None


def get_data(cfg: "SimpleNamespace") -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Loads and splits a dataset into training and validation DataFrames based on the provided configuration.
    Args:
        cfg (SimpleNamespace): A configuration object containing the following attributes:
            - df_path (str): Path to the CSV file containing the dataset.
            - overfit (bool): Flag indicating whether to use a subset of the data for overfitting.
            - overfit_tomos (list): List of tomo IDs to use when overfitting is enabled.
            - fold (int): Fold index for splitting the data. If -1, a default split is used.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training DataFrame and validation DataFrame.
    """
    df = pd.read_csv(cfg.df_path)
    if cfg.overfit:
        train_df = df[df.tomo_id.isin(cfg.overfit_tomos)]
        val_df = df[df.fold == 0]
    else:
        if cfg.fold > -1:
            train_df = df[df.fold != cfg.fold]
            val_df = df[df.fold == cfg.fold]
        else:
            train_df = df[df.fold != 0]
            val_df = df[df.fold == 0]
    return train_df, val_df


def get_metrics_logger(
    cfg: "SimpleNamespace",
) -> "tuple[tu.SummaryWriter, tu.SummaryWriter]":
    """Creates and returns summary writers for training and validation metrics logging."""
    train_log_dir = os.path.join("logs", cfg.start_time, "train")
    val_log_dir = os.path.join("logs", cfg.start_time, "eval")
    return tu.get_summary_writer(train_log_dir), tu.get_summary_writer(val_log_dir)


def get_optimizer(
    cfg: "SimpleNamespace", model: "torch.nn.Module"
) -> "optim.Optimizer | None":
    """
    Creates and returns an optimizer based on the configuration provided.
    Args:
        cfg (SimpleNamespace): A configuration object containing the optimizer settings.
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
        return optim.Adam(params_, lr=cfg.lr, weight_decay=cfg.weight_decay)

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
        return optim.AdamW(params, lr=cfg.lr)

    elif cfg.optimizer == "AdamW":
        return optim.AdamW(params_, lr=cfg.lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "SGD":
        return optim.SGD(
            params_,
            lr=cfg.lr,
            momentum=cfg.sgd_momentum,
            nesterov=cfg.sgd_nesterov,
            weight_decay=cfg.weight_decay,
        )
    return None


def get_multistep_schedule_with_warmup(
    optimizer: "optim.Optimizer",
    warmup_steps: "int",
    training_steps: "int",
    end_lambda: "float" = 1e-3,
    m: "int" = 10,
    last_epoch: "int" = -1,
) -> "optim.lr_scheduler.LambdaLR":
    """
    Modified schedule:
      - Warmup: linearly increases from 0.001% of base LR to 100% of base LR over warmup_steps.
      - Then, in the non-warmup phase (total_steps - warmup_steps):
          * LR = base_lr * [(1- n/m) + end_lambda] where:
              * m: number of milestones
              * n: step number in [1, m]
    """

    def create_milestones(steps: "int", m: "int") -> "list[int]":
        """returns a list of milestones for the given number of steps and m."""
        g = steps // m
        milestones = []
        for i in range(1, m + 1):
            milestones += [i] * g
        return milestones

    def lr_lambda(current_step: int) -> "float":
        """returns the learning rate multiplier based on the current step."""
        if current_step <= warmup_steps:
            return 1e-5 + (1.0 - 1e-5) * float(current_step) / float(
                max(1, warmup_steps)
            )
        else:
            remaining_steps = training_steps - warmup_steps
            milestones = create_milestones(remaining_steps, m)
            delta = 1 - (milestones[current_step - warmup_steps] / m)
        return delta + end_lambda

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: "optim.Optimizer",
    num_warmup_steps: "int",
    num_training_steps: "int",
    end_lambda: "float" = 0.0,
    last_epoch: "int" = -1,
) -> "optim.lr_scheduler.LambdaLR":
    """
    from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        end_lambda (`float`):
            Mutiplicative factor at the end of training.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> "float":
        """returns the learning rate multiplier based on the current step."""
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            end_lambda,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(
    cfg: "SimpleNamespace", optimizer: "optim.Optimizer", total_steps: "int"
) -> "optim.lr_scheduler.LRScheduler | None":
    """
    Creates and returns a learning rate scheduler based on the provided configuration.
    Args:
        cfg (SimpleNamespace): Configuration object containing the scheduler type and its parameters.
            - cfg.schedule (str): The type of scheduler to use. Options are:
                - "steplr": StepLR scheduler.
                - "multistep": MultiStepLR scheduler with warmup.
                - "linear": Linear scheduler with warmup.
                - "CosineAnnealingLR": Cosine Annealing scheduler.
            - cfg.epochs_step (int): Number of epochs per step (used for "steplr").
            - cfg.batch_size (int): Batch size used in training.
            - cfg.world_size (int): Number of distributed training processes.
            - cfg.warmup (int): Number of warmup steps (used for "multistep" and "linear").
            - cfg.milestones (list[int]): Milestones for MultiStepLR (used for "multistep").
            - cfg.end_lambda (float): Final lambda value for MultiStepLR (used for "multistep").
            - cfg.epochs (int): Total number of epochs (used for "multistep" and "linear").
        optimizer (optim.Optimizer): The optimizer for which the scheduler will adjust the learning rate.
        total_steps (int): Total number of training steps.
    Returns:
        optim.lr_scheduler.LRScheduler | None: The configured learning rate scheduler, or None if no valid scheduler type is specified.
    """
    if cfg.schedule == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs_step
            * (total_steps // cfg.batch_size)
            // cfg.world_size,
            gamma=0.5,
        )
    elif cfg.schedule == "multistep":
        return get_multistep_schedule_with_warmup(
            optimizer,
            warmup_steps=cfg.warmup,
            m=cfg.milestones,
            training_steps=cfg.epochs
            * (total_steps // cfg.batch_size)
            // cfg.world_size,
            end_lambda=cfg.end_lambda,
        )

    elif cfg.schedule == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=cfg.epochs
            * (total_steps // cfg.batch_size)
            // cfg.world_size,
        )

    elif cfg.schedule == "CosineAnnealingLR":
        T_max = int(np.ceil(0.5 * total_steps))
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=1e-8
        )
    return None


def set_seed(seed: "int" = 57) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    xla.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
