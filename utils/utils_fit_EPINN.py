"""One-epoch training/validation routine for E-PINN."""

from __future__ import annotations

from pathlib import Path

import torch

from utils.callbacks import LossHistory, save_top_k_checkpoint
from utils.utils import get_lr


def _run_epoch(
    model: torch.nn.Module,
    modelLoss: torch.nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    tbptt_length: int | None,
    gradient_clip: float | None,
) -> tuple[float, float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    gradient_norm_total = 0.0
    gradient_norm_count = 0
    metric_totals: dict[str, float] = {}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for loads, target in loader:
            loads = loads.to(device)
            reports_metrics = getattr(modelLoss, "reports_metrics", False)
            if reports_metrics:
                target_displacement = target["dis"].to(device)
            total_steps = loads.shape[-1]
            chunk_length = total_steps if tbptt_length is None else tbptt_length
            state = None
            for start in range(0, total_steps, chunk_length):
                stop = min(start + chunk_length, total_steps)
                load_chunk = loads[..., start:stop]
                if training:
                    optimizer.zero_grad(set_to_none=True)
                prediction, state = model.forward_chunk(load_chunk, state)
                if reports_metrics:
                    target_chunk = {
                        "dis": target_displacement[:, start:stop],
                    }
                    loss, metrics = modelLoss(
                        prediction,
                        target_chunk,
                        return_metrics=True,
                    )
                else:
                    loss = modelLoss(prediction)
                    metrics = {}
                if training:
                    loss.backward()
                    if gradient_clip is not None:
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip
                        )
                        gradient_norm_total += float(gradient_norm.detach())
                        gradient_norm_count += 1
                    optimizer.step()
                weight = loads.shape[0] * (stop - start)
                total += float(loss.detach()) * weight
                for name, value in metrics.items():
                    metric_totals[name] = metric_totals.get(name, 0.0) + (
                        float(value) * weight
                    )
                count += weight
    if count == 0:
        raise ValueError("The data loader did not produce any batches.")
    mean_gradient_norm = (
        gradient_norm_total / gradient_norm_count
        if gradient_norm_count > 0
        else 0.0
    )
    mean_metrics = {
        name: value / count for name, value in metric_totals.items()
    }
    return total / count, mean_gradient_norm, mean_metrics


def fitOneEpoch_EPINN_PhyLoss(
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
    train_loss, train_gradient_norm, train_metrics = _run_epoch(
        model, modelLoss, genTrain, device, optimizer, tbptt_length,
        gradient_clip,
    )
    val_loss, _, val_metrics = _run_epoch(
        model, modelLoss, genVal, device, None, tbptt_length, None,
    )
    metric_details = {
        **{f"train_{name}": value for name, value in train_metrics.items()},
        **{f"val_{name}": value for name, value in val_metrics.items()},
    }
    lossHistory.append_loss(
        epoch + 1,
        train_loss,
        val_loss,
        {
            "train_gradient_norm_before_clip": train_gradient_norm,
            **metric_details,
        },
    )
    metric_text = ""
    if train_metrics:
        metric_text = (
            " - full/trend: "
            f"{train_metrics['full_log_cosh']:.3e}/"
            f"{train_metrics['weighted_trend_log_cosh']:.3e}"
            " - true_rrmse: "
            f"{train_metrics['true_displacement_rrmse_percent']:.3f}%/"
            f"{val_metrics['true_displacement_rrmse_percent']:.3f}%"
            " - true_corr: "
            f"{train_metrics['true_displacement_correlation']:.4f}/"
            f"{val_metrics['true_displacement_correlation']:.4f}"
        )
    print(
        f"Epoch {epoch + 1}/{endEpoch} - "
        f"loss: {train_loss:.6e} - val_loss: {val_loss:.6e} - "
        f"grad_norm: {train_gradient_norm:.3e} - "
        f"lr: {get_lr(optimizer):.3e}{metric_text}"
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
                "train_gradient_norm_before_clip": train_gradient_norm,
                "gradient_clip": gradient_clip,
                **metric_details,
            },
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            max_to_keep=10,
        )
    return train_loss, val_loss
