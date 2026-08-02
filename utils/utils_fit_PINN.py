"""One-epoch training/validation routine for the conventional PINN."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from utils.callbacks import LossHistory, save_top_k_checkpoint
from utils.utils import get_lr, move_target


def _run_epoch(
    model: torch.nn.Module,
    modelLoss: torch.nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    tbptt_length: int | None,
    gradient_clip: float | None,
    compute_physics: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    data_keys = (
        "data", "increment", "displacement",
        "increment_mse_physical", "displacement_mse_physical",
    )
    totals = {key: 0.0 for key in (*data_keys, "physics")}
    data_count = 0
    physics_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for loads, target in loader:
            loads = loads.to(device)
            target = move_target(target, device)
            total_steps = loads.shape[-1]
            chunk_length = total_steps if tbptt_length is None else tbptt_length
            state = None
            for start in range(0, total_steps, chunk_length):
                stop = min(start + chunk_length, total_steps)
                load_chunk = loads[..., start:stop]
                target_chunk = {
                    key: (
                        value[:, start:stop]
                        if value.ndim >= 2 and value.shape[1] == total_steps
                        else value
                    )
                    for key, value in target.items()
                }
                if training:
                    optimizer.zero_grad(set_to_none=True)
                prediction, state = model.forward_chunk(
                    load_chunk, state, compute_physics=compute_physics
                )
                loss, parts = modelLoss(
                    load_chunk,
                    target_chunk,
                    prediction,
                    compute_physics=compute_physics,
                )
                if training:
                    loss.backward()
                    if gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip
                        )
                    optimizer.step()
                steps = stop - start
                labelled_count = int(target_chunk["labelled"].bool().sum())
                labelled_weight = labelled_count * steps
                all_sample_weight = loads.shape[0] * steps
                if labelled_weight > 0:
                    for key in data_keys:
                        totals[key] += float(parts[key]) * labelled_weight
                    data_count += labelled_weight
                totals["physics"] += float(parts["physics"]) * all_sample_weight
                physics_count += all_sample_weight
    if physics_count == 0 or data_count == 0:
        raise ValueError("The data loader did not produce any batches.")
    result = {
        key: totals[key] / data_count for key in data_keys
    }
    result["physics"] = totals["physics"] / physics_count
    result["total"] = (
        result["data"] + modelLoss.current_physics_weight * result["physics"]
    )
    result["increment_rmse_m"] = math.sqrt(result["increment_mse_physical"])
    result["displacement_rmse_m"] = math.sqrt(
        result["displacement_mse_physical"]
    )
    return result


def fitOneEpoch_PINN_DisIncrement_LabPhyLoss(
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
    # The equation is already nondimensionalized, so every training sample
    # uses the complete fixed physics loss from the first epoch.
    physics_fraction = modelLoss.physics_weight
    modelLoss.set_physics_fraction(1.0)
    train = _run_epoch(
        model,
        modelLoss,
        genTrain,
        device,
        optimizer,
        tbptt_length,
        gradient_clip,
        True,
    )
    modelLoss.set_physics_fraction(1.0)
    val = _run_epoch(
        model,
        modelLoss,
        genVal,
        device,
        None,
        tbptt_length,
        None,
        True,
    )
    lossHistory.append_loss(
        epoch + 1,
        train["data"],
        val["data"],
        {
            **{f"train_{key}": value for key, value in train.items()},
            **{f"val_{key}": value for key, value in val.items()},
            "physics_weight": physics_fraction,
        },
    )
    print(
        f"Epoch {epoch + 1}/{endEpoch} - "
        f"loss: {train['data']:.6e} - val_loss: {val['data']:.6e} - "
        f"increment: {train['increment']:.3e} - "
        f"displacement: {train['displacement']:.3e} - "
        f"physics: {train['physics']:.3e} - "
        f"val_physics: {val['physics']:.3e} - "
        f"val_u_RMSE: {val['displacement_rmse_m']:.3e} m - "
        f"physics_weight: {physics_fraction:.3e} - "
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
                "train_loss": train["data"],
                "val_loss": val["data"],
                "train_total_loss": train["total"],
                "val_physics_loss": val["physics"],
                "val_displacement_rmse_m": val["displacement_rmse_m"],
                "val_increment_rmse_m": val["increment_rmse_m"],
                "physics_weight": physics_fraction,
            },
            epoch=epoch + 1,
            train_loss=train["data"],
            val_loss=val["data"],
            max_to_keep=10,
        )
    return train["data"], val["data"]
