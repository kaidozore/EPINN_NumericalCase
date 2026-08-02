"""Prediction, metrics and MATLAB-output helpers."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch


def batched_predict(
    model: torch.nn.Module,
    load: np.ndarray,
    indices: Iterable[int],
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    indices = np.asarray(list(indices), dtype=np.int64)
    collected: Dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            load_tensor = torch.as_tensor(
                load[part].transpose(0, 2, 1)[:, None, :, :],
                dtype=torch.float64,
                device=device,
            )
            prediction = model(load_tensor)
            for key, value in prediction.items():
                collected.setdefault(key, []).append(
                    value.detach().cpu().numpy()
                )
    return {
        key: np.concatenate(values, axis=0)
        for key, values in collected.items()
    }


def acceleration_from_equilibrium(
    load: np.ndarray,
    displacement: np.ndarray,
    velocity: np.ndarray,
    internal_force: np.ndarray,
    mass: np.ndarray,
    damping: np.ndarray,
) -> np.ndarray:
    right_hand_side = (
        load
        - np.einsum("ij,btj->bti", damping, velocity)
        - internal_force
    )
    return np.einsum(
        "ij,btj->bti", np.linalg.inv(mass), right_hand_side
    )


def print_displacement_metrics(
    reference: np.ndarray, prediction: np.ndarray
) -> None:
    error = prediction - reference
    mae = np.mean(np.abs(error), axis=(0, 1))
    rmse = np.sqrt(np.mean(np.square(error), axis=(0, 1)))
    print("Displacement test metrics by DOF:")
    for index, (mae_value, rmse_value) in enumerate(
        zip(mae, rmse), start=1
    ):
        print(
            f"  DOF {index}: MAE={mae_value:.6e} m, "
            f"RMSE={rmse_value:.6e} m"
        )


def save_matlab_output(
    output_path,
    prediction: Dict[str, np.ndarray],
    time: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    try:
        from scipy.io import savemat
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required to write MATLAB .mat prediction files."
        ) from exc

    # Return to the MATLAB layout used by Newmark_beta.m: [DOF,time,sample].
    savemat(
        output_path,
        {
            "U": prediction["dis"].transpose(2, 1, 0),
            "V": prediction["vel"].transpose(2, 1, 0),
            "Acc": prediction["acc"].transpose(2, 1, 0),
            "Fint": prediction["force_internal"].transpose(2, 1, 0),
            "Fnonlinear": prediction["force_nonlinear"].transpose(2, 1, 0),
            "time": time.reshape(1, -1),
            "sampleIndexPython": sample_indices.reshape(1, -1),
            "sampleIndexMATLAB": (sample_indices + 1).reshape(1, -1),
        },
        do_compression=True,
    )
