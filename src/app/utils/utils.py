import os
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch_xla.test.test_utils as tu
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.datasets import BYUCustomDataset


def calc_grad_norm(parameters, norm_type=2.):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def calc_weight_norm(parameters,norm_type=2.):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters]).mean()
    return total_norm


def create_checkpoint(cfg, model, optimizer, scheduler=None, scaler=None):
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


def load_checkpoint(cfg, model, optimizer, scheduler=None, scaler=None):
    checkpoint = torch.load(cfg.resume_from, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler_dict = checkpoint['scheduler']
    if scaler is not None:    
        scaler.load_state_dict(checkpoint['scaler'])
        
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler_dict, scaler, epoch


def collate_fn(batch):
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
    batch_dict = {key:batch_data[key] for key in str_keys}
    for key in tensor_keys:
        batch_dict[key] = (torch.cat(batch_data[key]) if key in ["input", "target"]
                           else torch.stack(batch_data[key]))
    return batch_dict


def compute_grad_metrics(cfg, model):
    grad_norm = (calc_grad_norm(model.parameters())
                 if cfg.track_grad_norm else torch.nan)
    if cfg.clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
    grad_norm_after_clip = (calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                            if (cfg.clip_grad > 0 and cfg.track_grad_norm)
                            else torch.nan) 
    weight_norm = (calc_weight_norm(model.parameters(), cfg.grad_norm_type)
                   if cfg.track_weight_norm else torch.nan)
    return torch.tensor([grad_norm, grad_norm_after_clip, weight_norm]).to(cfg.device)


def get_data_loaders(cfg):
    if cfg.train:
        train_ds = BYUCustomDataset(cfg, mode="train", df=cfg.train_df, aug=cfg.train_transforms)
        
        train_sampler = (DistributedSampler(train_ds,
                                            num_replicas=cfg.world_size,
                                            rank=cfg.rank,
                                            shuffle=cfg.shuffle) if cfg.tpu else None)
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=cfg.pin_memory,
            sampler=train_sampler,
            shuffle=False if train_sampler else cfg.shuffle,
            drop_last=cfg.drop_last,
            prefetch_factor=cfg.prefetch_factor
        )
        
        val_ds = BYUCustomDataset(cfg, mode="validation", df=cfg.val_df, aug=cfg.eval_transforms)
        val_loader = DataLoader(
            dataset=val_ds,
            batch_size=cfg.batch_size_val,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=cfg.val_pin_memory,
            prefetch_factor=cfg.prefetch_factor
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
            prefetch_factor=cfg.prefetch_factor
        )
        return test_loader


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def extend_outputs(outputs):
    
    def recursive_defaultdict():
        return defaultdict(recursive_defaultdict)

    def recursive_extend(acc, item):
        for key, value in item.items():
            if isinstance(value, dict):
                recursive_extend(acc[key], value)
            elif isinstance(value, list):
                if not isinstance(acc[key], list):
                    acc[key] = []
                acc[key].extend(value)
            else:
                if not isinstance(acc[key], list):
                    acc[key] = []
                acc[key].append(value)
    extended_outputs = recursive_defaultdict()
    for item in outputs:
        recursive_extend(extended_outputs, item)
    return defaultdict_to_dict(extended_outputs)


def get_data(cfg):
    df = pd.read_csv(cfg.df_path)
    if cfg.fold > -1:
        train_df = df[df.fold != cfg.fold]
        val_df = df[df.fold == cfg.fold]
    else:
        train_df = df[df.fold != 0]
        val_df = df[df.fold == 0]
    return train_df, val_df


def get_metrics_logger(cfg):
    train_log_dir = os.path.join('logs', cfg.start_time, 'train')
    val_log_dir = os.path.join('logs', cfg.start_time, 'eval')
    return tu.get_summary_writer(train_log_dir), tu.get_summary_writer(val_log_dir)


def get_optimizer(cfg, model):
    params = model.parameters()
    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
   
    elif cfg.optimizer == "AdamW_plus":
        paras = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        params = [{"params": [param for name, param in paras if (not any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":cfg.weight_decay},
                  {"params": [param for name, param in paras if (any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":0.},
                 ]        
        optimizer = optim.AdamW(params, lr=cfg.lr)         

    elif cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.sgd_momentum,
            nesterov=cfg.sgd_nesterov,
            weight_decay=cfg.weight_decay,
        )
    return optimizer


def get_multistep_schedule_with_warmup(optimizer, warmup_steps, training_steps, end_lambda=1e-3, m=10, last_epoch=-1):
    """
    Modified schedule:
      - Warmup: linearly increases from 0.001% of base LR to 100% of base LR over warmup_steps.
      - Then, in the non-warmup phase (total_steps - warmup_steps):
          * LR = base_lr * [(1- n/m) + end_lambda] where:
              * m: number of milestones
              * n: step number in [1, m]
    """
    def create_milestones(steps, m):
        g =  steps // m
        milestones = []
        for i in range(1, m + 1):
            milestones += [i] * g
        return milestones
    
    def lr_lambda(current_step: int):
        if current_step <= warmup_steps:
            return 1e-5 + (1.0 - 1e-5) * float(current_step) / float(max(1, warmup_steps))
        else:
            remaining_steps = training_steps - warmup_steps
            milestones = create_milestones(remaining_steps, m)
            delta = 1 - (milestones[current_step - warmup_steps]/m)
        return delta +  end_lambda
        
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, end_lambda=0., last_epoch=-1):
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

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            end_lambda, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(cfg, optimizer, total_steps):
    if cfg.schedule == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs_step * (total_steps // cfg.batch_size) // cfg.world_size,
            gamma=0.5,
        )
    elif cfg.schedule == "multistep":
        scheduler = get_multistep_schedule_with_warmup(
            optimizer,
            warmup_steps=cfg.warmup,
            m=cfg.milestones,
            training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
            end_lambda=cfg.end_lambda)
        
    elif cfg.schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
        )
        
    elif cfg.schedule == "CosineAnnealingLR":
        T_max = int(np.ceil(0.5*total_steps))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=T_max,
                                                         eta_min=1e-8)  
    else:
        scheduler = None
    return scheduler


def set_seed(seed=57):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True