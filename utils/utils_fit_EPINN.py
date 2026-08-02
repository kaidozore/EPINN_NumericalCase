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
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    gradient_norm_total = 0.0
    gradient_norm_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for loads, _ in loader:
            loads = loads.to(device)
            total_steps = loads.shape[-1]
            chunk_length = total_steps if tbptt_length is None else tbptt_length
            state = None
            for start in range(0, total_steps, chunk_length):
                stop = min(start + chunk_length, total_steps)
                load_chunk = loads[..., start:stop]
                if training:
                    optimizer.zero_grad(set_to_none=True)
                prediction, state = model.forward_chunk(load_chunk, state)
                loss = modelLoss(prediction)
                if training:
                    loss.backward()
                    if gradient_clip is not None:
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip
                        )
                        gradient_norm_total += float(gradient_norm.detach())
                        gradient_norm_count += 1
                    optimizer.step()
                weight = stop - start
                total += float(loss.detach()) * weight
                count += weight
    if count == 0:
        raise ValueError("The data loader did not produce any batches.")
    mean_gradient_norm = (
        gradient_norm_total / gradient_norm_count
        if gradient_norm_count > 0
        else 0.0
    )
    return total / count, mean_gradient_norm


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
    train_loss, train_gradient_norm = _run_epoch(
        model, modelLoss, genTrain, device, optimizer, tbptt_length,
        gradient_clip,
    )
    val_loss, _ = _run_epoch(
        model, modelLoss, genVal, device, None, tbptt_length, None,
    )
    lossHistory.append_loss(
        epoch + 1,
        train_loss,
        val_loss,
        {"train_gradient_norm_before_clip": train_gradient_norm},
    )
    print(
        f"Epoch {epoch + 1}/{endEpoch} - "
        f"loss: {train_loss:.6e} - val_loss: {val_loss:.6e} - "
        f"grad_norm: {train_gradient_norm:.3e} - "
        f"lr: {get_lr(optimizer):.3e}"
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
            },
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            max_to_keep=10,
        )
    return train_loss, val_loss
