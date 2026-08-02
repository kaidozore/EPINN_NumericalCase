"""One-epoch training/validation routine for the conventional PINN."""

from __future__ import annotations

from pathlib import Path

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
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for loads, _ in loader:
            loads = loads.to(device)
            total_steps = loads.shape[-1]
            chunk_length = total_steps if tbptt_length is None else tbptt_length
            state = None
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
                loss = model_loss(load_chunk, prediction)
                if training:
                    chunk_fraction = (stop - start) / total_steps
                    (loss * chunk_fraction).backward()
                weight = loads.shape[0] * (stop - start)
                total += float(loss.detach()) * weight
                count += weight
            if training:
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip
                    )
                optimizer.step()
    if count == 0:
        raise ValueError("The data loader did not produce any batches.")
    return total / count


def fitOneEpoch_PINN_DisIncrement_PhyLoss(
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
    train_loss = _run_epoch(
        model, modelLoss, genTrain, device, optimizer,
        tbptt_length, gradient_clip,
    )
    val_loss = _run_epoch(
        model, modelLoss, genVal, device, None, tbptt_length, None,
    )
    lossHistory.append_loss(epoch + 1, train_loss, val_loss)
    print(
        f"Epoch {epoch + 1}/{endEpoch} - "
        f"loss: {train_loss:.6e} - val_loss: {val_loss:.6e} - "
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
            },
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            max_to_keep=10,
        )
    return train_loss, val_loss
