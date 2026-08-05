"""Training loop for the displacement-increment PINN."""

from __future__ import annotations

from pathlib import Path
import time

import torch

from utils.callbacks import LossHistory, save_top_k_checkpoint
from utils.utils import get_lr


def _run_epoch(
    model: torch.nn.Module,
    model_loss: torch.nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    tbptt_length: int | None,
    gradient_clip: float | None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    metric_totals: dict[str, float] = {}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for loads, _ in loader:
            loads = loads.to(device)
            total_steps = loads.shape[-1]
            chunk_length = total_steps if tbptt_length is None else tbptt_length
            state = None
            loss_state = None
            if training:
                # Keep one set of network parameters throughout the complete
                # response history.  Gradients are accumulated over detached
                # TBPTT chunks, then one optimizer update is applied per batch.
                optimizer.zero_grad(set_to_none=True)
            for start in range(0, total_steps, chunk_length):
                stop = min(start + chunk_length, total_steps)
                load_chunk = loads[..., start:stop]
                prediction, state = model.forward_chunk(
                    load_chunk, state, compute_physics=True
                )
                loss, loss_state, metrics = model_loss(
                    load_chunk,
                    prediction,
                    previous_equilibrium_displacement=loss_state,
                    return_state=True,
                    return_metrics=True,
                )
                if training:
                    chunk_fraction = (stop - start) / total_steps
                    (loss * chunk_fraction).backward()
                weight = loads.shape[0] * (stop - start)
                total += float(loss.detach()) * weight
                for name, value in metrics.items():
                    metric_totals[name] = metric_totals.get(name, 0.0) + (
                        float(value) * weight
                    )
                count += weight
            if training:
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip
                    )
                optimizer.step()
    if count == 0:
        raise ValueError("The data loader did not produce any batches.")
    return total / count, {
        name: value / count for name, value in metric_totals.items()
    }


def fitOneEpoch_PINN_Increment_PhyLoss(
    model: torch.nn.Module,
    modelLoss: torch.nn.Module,
    lossHistory: LossHistory,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    genTrain,
    genVal,
    endEpoch: int,
    device: torch.device,
    checkpoint_dir: str | Path,
    checkpoint_data: dict,
    tbptt_length: int | None = None,
    gradient_clip: float | None = 1.0,
    save_period: int = 1,
) -> tuple[float, float]:
    train_start = time.perf_counter()
    train_loss, train_metrics = _run_epoch(
        model, modelLoss, genTrain, device, optimizer,
        tbptt_length, gradient_clip,
    )
    train_time_seconds = time.perf_counter() - train_start

    validation_start = time.perf_counter()
    val_loss, val_metrics = _run_epoch(
        model, modelLoss, genVal, device, None, tbptt_length, None,
    )
    validation_time_seconds = time.perf_counter() - validation_start
    epoch_compute_time_seconds = (
        train_time_seconds + validation_time_seconds
    )
    lossHistory.append_loss(
        epoch + 1,
        train_loss,
        val_loss,
        {
            "train_time_seconds": train_time_seconds,
            "validation_time_seconds": validation_time_seconds,
            "epoch_compute_time_seconds": epoch_compute_time_seconds,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
        },
    )
    print(
        f"Epoch {epoch + 1}/{endEpoch} - "
        f"loss: {train_loss:.6e} - val_loss: {val_loss:.6e} - "
        "full_dis_corr: "
        f"{train_metrics['full_displacement_correlation']:.4f}/"
        f"{val_metrics['full_displacement_correlation']:.4f} - "
        "increment/local_cumsum: "
        f"{train_metrics['increment_equilibrium_mse']:.3e}/"
        f"{train_metrics['weighted_local_cumsum_mse']:.3e} - "
        f"lr: {get_lr(optimizer):.3e} - "
        f"time: {train_time_seconds:.2f}/{validation_time_seconds:.2f} s"
    )
    if (epoch + 1) % save_period == 0 or epoch + 1 == endEpoch:
        save_top_k_checkpoint(
            checkpoint_dir,
            {
                **checkpoint_data,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_time_seconds": train_time_seconds,
                "validation_time_seconds": validation_time_seconds,
                "epoch_compute_time_seconds": epoch_compute_time_seconds,
                **{
                    f"train_{name}": value
                    for name, value in train_metrics.items()
                },
                **{
                    f"val_{name}": value
                    for name, value in val_metrics.items()
                },
            },
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            max_to_keep=10,
        )
    return train_loss, val_loss
