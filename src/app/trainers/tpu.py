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

# mypy: allow_untyped_calls,allow_untyped_defs
import datetime
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.distributed as D
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
import torch_xla.experimental.pjrt_backend
import torch_xla.runtime as xr
import torch_xla.test.test_utils as tu
from metrics import calc_metric
from processings.post_processing import post_process_pipeline
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from utils import (
    compute_grad_metrics,
    create_checkpoint,
    get_data,
    get_data_loaders,
    get_metrics_logger,
    get_optimizer,
    get_scheduler,
    set_seed,
)


def evaluate(
    cfg: SimpleNamespace, model: torch.nn.Module, val_loader: pl.MpDeviceLoader
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Evaluate the model on the validation set.
    Args:
        cfg (SimpleNamespace): Configuration object.
        model (torch.nn.Module): The model to evaluate.
        val_loader (pl.MpDeviceLoader): The validation data loader.
    """
    model.eval()
    torch.set_grad_enabled(False)
    losses = []
    coords = []
    disable_pbar = not xm.is_master_ordinal()
    pbar = tqdm(
        total=len(val_loader), desc="Validation", disable=disable_pbar, leave=False
    )
    with pbar:
        for i, batch in enumerate(val_loader):
            output_dict = model(batch)
            loss = output_dict["loss"]
            losses.append(loss.detach().item())
            coords.append(post_process_pipeline(cfg, output_dict))
            if (i + 1 % cfg.closure_steps == 0) or ((i + 1) % len(val_loader) == 0):
                pbar.update(min(cfg.closure_steps, len(val_loader) - i - 1))
    predictions = torch.cat(coords, dim=1)
    return torch.mean(torch.tensor(losses)), predictions


def train(
    cfg: "SimpleNamespace",
    model: "torch.nn.Module",
    batch: "dict[list[int], torch.Tensor]",
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.LRScheduler | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Train the model for one step.
    Args:
        cfg (SimpleNamespace): Configuration object.
        model (torch.nn.Module): The model to train.
        batch: The batch of data to train on.
        optimizer: The optimizer to use.
        scheduler: The learning rate scheduler to use.
    Returns:
        tuple: The loss and the gradient metrics.
    """
    optimizer.zero_grad()  # In case of gradient accumulation move this in the condidition scope
    model.train()
    torch.set_grad_enabled(True)
    with torch.autocast(device_type="xla"):
        output_dict = model(batch)
    loss = output_dict["loss"]
    grad_metrics = compute_grad_metrics(cfg, model)
    loss.backward()
    xm.optimizer_step(optimizer, barrier=True)
    if scheduler is not None:
        scheduler.step()
    return loss.detach().item(), grad_metrics


def trainer(cfg: SimpleNamespace, model: torch.nn.Module) -> None:
    """_Train the model on TPU.
    Args:
        cfg (SimpleNamespace): Configuration object.
        model (torch.nn.Module): The model to train.
    """
    # Initialization
    xm.master_print("Initialisation...")
    cfg.start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.session_dir = os.path.join(
        cfg.output_dir, f"seed_{cfg.seed}", f"fold{cfg.fold}", f"{cfg.start_time}"
    )
    if xm.is_master_ordinal():
        os.makedirs(cfg.session_dir, exist_ok=True)

    D.init_process_group(backend="xla", init_method="xla://")
    cfg.world_size = xr.world_size()
    cfg.rank = xr.global_ordinal()
    cfg.device = xm.xla_device()
    set_seed(cfg.seed)
    xla.manual_seed(cfg.seed, device=cfg.device)

    # Metrics logger
    xm.master_print("Metrics logger...")
    cfg.train_writer, cfg.val_writer = get_metrics_logger(cfg)

    # Load data
    xm.master_print("Load data...")
    cfg.train_df, cfg.val_df = get_data(cfg)
    train_loader, val_loader = get_data_loaders(cfg)  # type: ignore[misc]
    cfg.n_samples = len(train_loader.dataset)  # type: ignore[arg-type]

    cfg.lr = cfg.lr * cfg.world_size

    # Model and optimizer
    xm.master_print("Model and optimizer...")
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer, cfg.n_samples)  # type: ignore[arg-type]
    model = model.to(cfg.device)

    # Distributed Data Parallel setting
    xm.master_print("Distributed Data Parallel setting...")
    xm.broadcast_master_param(model)
    xm.master_print("Broadcasted model...")
    # model = DDP(model, find_unused_parameters=True, gradient_as_bucket_view=True)
    # xm.master_print("DDPed model...")
    train_device_loader = pl.MpDeviceLoader(train_loader, cfg.device)
    val_device_loader = pl.MpDeviceLoader(val_loader, cfg.device)
    xm.master_print("Set device load...")

    # Training loop vars initialization
    xm.master_print("Training loop vars initialization...")
    cfg.curr_epoch = 0
    cfg.curr_step = 0
    cfg.best_val_score = -1
    cfg.training_steps = len(train_loader) * cfg.epochs
    cfg.saved_best_model = False

    # Training loop
    xm.master_print("Training loop...")
    for epoch in range(cfg.epochs):
        cfg.curr_epoch = epoch + 1
        cfg.epoch_metrics = defaultdict(lambda: [])
        cfg.epoch_val_metrics = defaultdict(lambda: [])

        cfg.eff_bs = (
            cfg.sub_batch_size * cfg.batch_size * cfg.world_size
        )  # effective batch size accros all devices

        disable_pbar = not xm.is_master_ordinal()
        pbar = tqdm(total=len(train_device_loader), disable=disable_pbar)

        with pbar:
            for batch in train_device_loader:
                xm.master_print("starting batch training...")
                cfg.curr_step += 1

                # Train one step
                loss, train_metrics = train(cfg, model, batch, optimizer, scheduler)  # type: ignore[arg-type]

                # Gather metrics
                loss_value = xm.mesh_reduce("train_loss", loss, np.mean)
                train_metrics = xm.all_gather(train_metrics)

                # Reduce metrics
                lr = optimizer.param_groups[0]["lr"]  # type: ignore[union-attr]
                grad_norm, grad_norm_clip, weight_norm = torch.nanmean(
                    train_metrics, dim=0
                )
                train_metrics = {  # type: ignore[assignment]
                    "grad_norm": grad_norm,
                    "grad_norm_clip": grad_norm_clip,
                    "weight_norm": weight_norm,
                }
                train_metrics = {  # type: ignore[operator]
                    "loss": loss_value,
                    "batch_size": cfg.eff_bs,
                    "lr": lr,
                } | train_metrics
                train_metrics = {  # type: ignore[assignment]
                    k: v
                    for k, v in train_metrics.items()  # type: ignore[attr-defined]
                    if not torch.logical_or(v.isnan(), v.isinf())
                }

                # Add metrics to epoch metrics
                for key, value in train_metrics.items():  # type: ignore[attr-defined]
                    cfg.epoch_metrics[key].append(value)

                # Evaluation steps
                pred_df = None
                if (
                    cfg.curr_step % cfg.eval_steps == 0
                    or cfg.curr_step % cfg.training_steps == 0
                ):

                    # Evaluation pass
                    loss, predictions = evaluate(cfg, model, val_device_loader)

                    # Gather predictions: Must be a Tensor
                    predictions = xm.all_gather(predictions)
                    predictions = predictions.view(-1, 5)

                    # Reduce loss
                    loss_value = xm.mesh_reduce("val_loss", loss, np.mean)

                    # Predictions thresholding and metrics
                    pred_df = pd.DataFrame(
                        predictions.cpu().numpy(),
                        columns=["z", "y", "x", "conf", "ids"],
                    )
                    pred_df = pred_df.merge(
                        val_loader.dataset.tomo_mapping, on="ids", how="inner"  # type: ignore[union-attr]
                    )
                    val_score, best_th = calc_metric(
                        cfg, pred_df.copy(), cfg.val_df.copy()
                    )
                    val_metrics = {
                        "val_loss": loss_value,
                        "val_score": val_score,
                        "best_th": best_th,
                    }
                    val_metrics = {
                        k: v
                        for k, v in val_metrics.items()
                        if not torch.logical_or(v.isnan(), v.isinf())
                    }

                    # Add metrics to epoch metrics
                    for key, value in val_metrics.items():
                        cfg.epoch_val_metrics[key].append(value)

                    # Closure and logging
                    xm.add_step_closure(
                        tu.write_to_summary,
                        args=(cfg.val_writer, cfg.curr_step, val_metrics, False),
                        run_async=cfg.async_closure,
                    )

                    # Checkpointing best model
                    if val_score > cfg.best_val_score:
                        cfg.best_val_score = val_score
                        xm.wait_device_ops()
                        checkpoint = create_checkpoint(cfg, model, optimizer, scheduler)  # type: ignore[arg-type]
                        xm.save(
                            checkpoint,
                            os.path.join(cfg.session_dir, "best_model.pth"),
                            master_only=True,
                            global_master=True,
                        )
                        if xm.is_master_ordinal():
                            csv_path = os.path.join(
                                cfg.session_dir,
                                f"val_df_step_{cfg.curr_step}_proc_{cfg.rank}.csv",
                            )
                            pred_df.to_csv(csv_path, index=False)
                        xm.rendezvous("best_model_state")
                        cfg.saved_best_model = True

                # Steps checkpointing
                if (
                    cfg.save_checkpoint
                    and (cfg.curr_step % cfg.chkpt_step == 0)
                    and not cfg.saved_best_model
                ):
                    xm.wait_device_ops()
                    checkpoint = create_checkpoint(cfg, model, optimizer, scheduler)  # type: ignore[arg-type]
                    xm.save(
                        checkpoint,
                        os.path.join(cfg.session_dir, f"chkp_{cfg.curr_step}.pth"),
                        master_only=True,
                        global_master=True,
                    )
                    if pred_df is not None and xm.is_master_ordinal():
                        csv_path = os.path.join(
                            cfg.session_dir,
                            f"val_df_step_{cfg.curr_step}_proc_{cfg.rank}.csv",
                        )
                        pred_df.to_csv(csv_path, index=False)
                    xm.rendezvous("checkpoint_state")
                    cfg.saved_best_model = False

                last_losses_mean = (
                    torch.nanmean(
                        torch.tensor(cfg.epoch_metrics["loss"][-10:], device=cfg.device)
                    )
                    if len(cfg.epoch_metrics["loss"]) > 0
                    else 0
                )
                last_scores_mean = (
                    torch.nanmean(
                        torch.tensor(
                            cfg.epoch_val_metrics["val_score"][-10:], device=cfg.device
                        )
                    )
                    if len(cfg.epoch_val_metrics["val_score"]) > 0
                    else 0
                )
                last_val_losses_mean = (
                    torch.nanmean(
                        torch.tensor(
                            cfg.epoch_val_metrics["val_loss"][-10:], device=cfg.device
                        )
                    )
                    if len(cfg.epoch_val_metrics["val_loss"]) > 0
                    else 0
                )

                if (
                    cfg.curr_step % cfg.closure_steps == 0
                    or cfg.curr_step % cfg.training_steps == 0
                ):
                    # training closure and logging
                    xm.add_step_closure(
                        tu.write_to_summary,
                        args=(
                            cfg.train_writer,
                            cfg.curr_step,
                            train_metrics,
                            cfg.write_xla_metrics,
                        ),
                        run_async=cfg.async_closure,
                    )

                    # Update progress bar
                    xm.wait_device_ops()
                    pbar.set_description(
                        f"Epoch {cfg.curr_epoch}/{cfg.epochs} |"
                        f"Step {cfg.curr_step}/{cfg.training_steps} |"
                        f"LR: {lr:.2e} |"
                        f"Loss: {last_losses_mean:.2f} | "
                        f"Val Loss: {last_val_losses_mean:.2f} |"
                        f"Val Score: {last_scores_mean:.2f}"
                    )
                    pbar.update(
                        min(cfg.closure_steps, cfg.training_steps - cfg.curr_step)
                    )

        # End of epoch logging and checkpointing
        losses_mean = (
            torch.nanmean(torch.tensor(cfg.epoch_metrics["loss"], device=cfg.device))
            if len(cfg.epoch_metrics["loss"]) > 0
            else 0
        )
        scores_mean = (
            torch.nanmean(
                torch.tensor(cfg.epoch_val_metrics["val_score"], device=cfg.device)
            )
            if len(cfg.epoch_val_metrics["val_score"]) > 0
            else 0
        )
        val_losses_mean = (
            torch.nanmean(
                torch.tensor(cfg.epoch_val_metrics["val_loss"], device=cfg.device)
            )
            if len(cfg.epoch_val_metrics["val_loss"]) > 0
            else 0
        )
        xm.wait_device_ops()
        checkpoint = create_checkpoint(cfg, model, optimizer, scheduler)  # type: ignore[arg-type]
        xm.save(
            checkpoint,
            os.path.join(cfg.session_dir, "last_epoch_state.pth"),
            master_only=True,
            global_master=True,
        )
        pbar.set_description(
            f"epoch {cfg.curr_epoch}/{cfg.epochs} |"
            f"avg_loss: {losses_mean:.2f} |"
            f"avg_val_loss: {val_losses_mean:.2f} |"
            f"avg_val_score: {scores_mean:.2f}"
        )
        tu.close_summary_writer(cfg.train_writer)
        tu.close_summary_writer(cfg.val_writer)
        xm.rendezvous("last_epoch_state")
