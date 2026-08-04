"""Test increment or full E-PINN and export MATLAB-ready results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import re

import numpy as np
import torch

from config import CaseConfig
from nets.EPINN_Full_Net import EPINN_FullDis_PhyLSTM_NetBody
from nets.EPINN_Net import EPINN_PhyLSTM_NetBody
from utils.DataPreProcess import as_torch_case, build_data_split, load_case_data


_CHECKPOINT_PATTERN = re.compile(
    r"^ep(?P<epoch>\d+)-train(?P<train>[-+0-9.eE]+)-"
    r"val(?P<val>[-+0-9.eE]+)\.pth$"
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("increment", "full"), required=True
    )
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Specific loss_YYYY... directory; latest run is used by default.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Explicit checkpoint; otherwise the smallest retained val_loss.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--chunk-length", type=int, default=None,
        help="Prediction chunk length; checkpoint TBPTT length is the default.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _latest_run_directory(variant: str) -> Path:
    log_name = "EPINN_PhyLSTM" if variant == "increment" else "EPINN_Full_PhyLSTM"
    log_root = Path(__file__).resolve().parent / "logs" / log_name
    candidates = [path for path in log_root.glob("loss_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No training run was found under {log_root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _best_checkpoint(run_dir: Path) -> Path:
    ranked: list[tuple[float, float, int, Path]] = []
    for path in (run_dir / "checkpoints").glob("*.pth"):
        match = _CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue
        ranked.append(
            (
                float(match.group("val")),
                float(match.group("train")),
                int(match.group("epoch")),
                path,
            )
        )
    if not ranked:
        raise FileNotFoundError(
            f"No ranked checkpoint was found in {run_dir / 'checkpoints'}."
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        run_dir = (
            args.run_dir.expanduser().resolve()
            if args.run_dir is not None
            else checkpoint.parent.parent
        )
    else:
        run_dir = (
            args.run_dir.expanduser().resolve()
            if args.run_dir is not None
            else _latest_run_directory(args.variant).resolve()
        )
        checkpoint = _best_checkpoint(run_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "test_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, checkpoint, output_dir


def case_config_from_checkpoint(
    checkpoint: dict,
    data_root: Path,
) -> CaseConfig:
    stored = checkpoint["case_config"]
    accepted = {field.name for field in fields(CaseConfig)} - {"data_root"}
    values = {name: stored[name] for name in accepted if name in stored}
    return CaseConfig(data_root=data_root, **values)


def build_model(
    variant: str,
    checkpoint: dict,
    tensors: dict,
    config: CaseConfig,
    device: torch.device,
) -> torch.nn.Module:
    common = {
        "nLoad": int(checkpoint["n_load"]),
        "nLoadNL": int(checkpoint["n_dof"]),
        "influence_kernel": tensors["kernel"],
        "stiffness": tensors["stiffness"],
        "fiber": tensors["fiber"],
        "steel": tensors["steel"],
        "input_increment_scale": float(
            checkpoint.get(
                "input_increment_scale", config.displacement_increment_scale
            )
        ),
        "hidden_size": int(checkpoint["hidden_size"]),
        "fc_size": int(checkpoint["fc_size"]),
    }
    if variant == "increment":
        model = EPINN_PhyLSTM_NetBody(
            output_increment_scale=float(
                checkpoint.get(
                    "output_increment_scale",
                    config.displacement_increment_scale,
                )
            ),
            **common,
        )
    else:
        model = EPINN_FullDis_PhyLSTM_NetBody(
            input_displacement_scale=float(
                checkpoint.get(
                    "input_displacement_scale", config.displacement_scale
                )
            ),
            output_displacement_scale=float(
                checkpoint.get(
                    "output_displacement_scale", config.displacement_scale
                )
            ),
            **common,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.double().to(device).eval()


def chunked_batched_predict(
    model: torch.nn.Module,
    load: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    chunk_length: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if batch_size < 1 or chunk_length < 1:
        raise ValueError("batch_size and chunk_length must be positive.")
    wanted = ("dis_nl", "dis", "vel", "force_internal", "force_nonlinear")
    batches: dict[str, list[np.ndarray]] = {key: [] for key in wanted}
    with torch.no_grad():
        for batch_start in range(0, indices.size, batch_size):
            part = indices[batch_start : batch_start + batch_size]
            load_tensor = torch.as_tensor(
                load[part].transpose(0, 2, 1)[:, None],
                dtype=torch.float64,
                device=device,
            )
            state = None
            pieces: dict[str, list[torch.Tensor]] = {key: [] for key in wanted}
            total_steps = load_tensor.shape[-1]
            for start in range(0, total_steps, chunk_length):
                stop = min(start + chunk_length, total_steps)
                prediction, state = model.forward_chunk(
                    load_tensor[..., start:stop], state
                )
                for key in wanted:
                    pieces[key].append(prediction[key].detach().cpu())
            for key in wanted:
                batches[key].append(torch.cat(pieces[key], dim=1).numpy())
    return {key: np.concatenate(value, axis=0) for key, value in batches.items()}


def central_difference(
    displacement: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    if displacement.shape[1] < 4:
        raise ValueError("At least four time points are required.")
    dt = float(delta_t)
    velocity = np.zeros_like(displacement)
    acceleration = np.zeros_like(displacement)
    velocity[:, 1:-1] = (
        displacement[:, 2:] - displacement[:, :-2]
    ) / (2.0 * dt)
    velocity[:, -1:] = (
        3.0 * displacement[:, -1:]
        - 4.0 * displacement[:, -2:-1]
        + displacement[:, -3:-2]
    ) / (2.0 * dt)
    acceleration[:, 1:-1] = (
        displacement[:, 2:]
        - 2.0 * displacement[:, 1:-1]
        + displacement[:, :-2]
    ) / (dt * dt)
    acceleration[:, -1:] = (
        2.0 * displacement[:, -1:]
        - 5.0 * displacement[:, -2:-1]
        + 4.0 * displacement[:, -3:-2]
        - displacement[:, -4:-3]
    ) / (dt * dt)
    return velocity, acceleration


def displacement_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, np.ndarray | float]:
    error = prediction - reference
    mae = np.mean(np.abs(error), axis=(0, 1))
    rmse = np.sqrt(np.mean(error * error, axis=(0, 1)))
    reference_rms = np.sqrt(np.mean(reference * reference, axis=(0, 1)))
    rrmse = 100.0 * rmse / np.maximum(reference_rms, np.finfo(float).eps)
    correlation = np.empty(reference.shape[2], dtype=np.float64)
    for dof in range(reference.shape[2]):
        true_flat = reference[:, :, dof].reshape(-1)
        pred_flat = prediction[:, :, dof].reshape(-1)
        true_centered = true_flat - np.mean(true_flat)
        pred_centered = pred_flat - np.mean(pred_flat)
        denominator = np.linalg.norm(true_centered) * np.linalg.norm(pred_centered)
        correlation[dof] = (
            float(np.dot(true_centered, pred_centered) / denominator)
            if denominator > np.finfo(float).eps
            else 0.0
        )
    overall_rmse = float(np.sqrt(np.mean(error * error)))
    overall_true_rms = float(np.sqrt(np.mean(reference * reference)))
    return {
        "mae_m": mae,
        "rmse_m": rmse,
        "rrmse_percent": rrmse,
        "correlation": correlation,
        "overall_rmse_m": overall_rmse,
        "overall_rrmse_percent": (
            100.0
            * overall_rmse
            / max(overall_true_rms, np.finfo(float).eps)
        ),
    }


def print_metrics(
    network_metrics: dict,
    scl_metrics: dict,
) -> None:
    print("Test displacement metrics by DOF:")
    for dof in range(len(network_metrics["rmse_m"])):
        print(
            f"  DOF {dof + 1}: "
            f"network RMSE={network_metrics['rmse_m'][dof]:.6e} m, "
            f"RRMSE={network_metrics['rrmse_percent'][dof]:.3f}%, "
            f"corr={network_metrics['correlation'][dof]:.5f}; "
            f"SCL RMSE={scl_metrics['rmse_m'][dof]:.6e} m, "
            f"corr={scl_metrics['correlation'][dof]:.5f}"
        )
    print(
        "Overall network displacement: "
        f"RMSE={network_metrics['overall_rmse_m']:.6e} m, "
        f"RRMSE={network_metrics['overall_rrmse_percent']:.3f}%."
    )


def save_metrics_csv(
    path: Path,
    network_metrics: dict,
    scl_metrics: dict,
) -> None:
    fieldnames = [
        "dof",
        "network_mae_m", "network_rmse_m", "network_rrmse_percent",
        "network_correlation",
        "scl_mae_m", "scl_rmse_m", "scl_rrmse_percent",
        "scl_correlation",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for dof in range(len(network_metrics["rmse_m"])):
            writer.writerow(
                {
                    "dof": dof + 1,
                    "network_mae_m": network_metrics["mae_m"][dof],
                    "network_rmse_m": network_metrics["rmse_m"][dof],
                    "network_rrmse_percent": network_metrics[
                        "rrmse_percent"
                    ][dof],
                    "network_correlation": network_metrics["correlation"][dof],
                    "scl_mae_m": scl_metrics["mae_m"][dof],
                    "scl_rmse_m": scl_metrics["rmse_m"][dof],
                    "scl_rrmse_percent": scl_metrics["rrmse_percent"][dof],
                    "scl_correlation": scl_metrics["correlation"][dof],
                }
            )


def save_matlab_results(
    path: Path,
    variant: str,
    checkpoint_path: Path,
    checkpoint: dict,
    data,
    test_indices: np.ndarray,
    prediction: dict[str, np.ndarray],
    network_velocity: np.ndarray,
    network_acceleration: np.ndarray,
    scl_acceleration: np.ndarray,
    network_metrics: dict,
    scl_metrics: dict,
) -> None:
    try:
        from scipy.io import savemat
    except ImportError as exc:
        raise RuntimeError("scipy is required to save MATLAB results.") from exc

    matlab = lambda value: np.asarray(value).transpose(2, 1, 0)
    test_load = data.load[test_indices]
    savemat(
        path,
        {
            "method": f"EPINN_{variant}",
            "checkpointPath": str(checkpoint_path),
            "checkpointEpoch": int(checkpoint.get("epoch", -1)),
            "checkpointValLoss": float(checkpoint.get("val_loss", np.nan)),
            "time": data.time.reshape(1, -1),
            "deltaT": data.delta_t,
            "sampleIndexPython": test_indices.reshape(1, -1),
            "sampleIndexMATLAB": (test_indices + 1).reshape(1, -1),
            "Fwave": matlab(test_load),
            "U_pred": matlab(prediction["dis_nl"]),
            "V_pred": matlab(network_velocity),
            "Acc_pred": matlab(network_acceleration),
            "U_scl": matlab(prediction["dis"]),
            "V_scl": matlab(prediction["vel"]),
            "Acc_scl": matlab(scl_acceleration),
            "U_true": matlab(data.displacement[test_indices]),
            "V_true": matlab(data.velocity[test_indices]),
            "Acc_true": matlab(data.acceleration[test_indices]),
            "Fint_pred": matlab(prediction["force_internal"]),
            "Fnonlinear_pred": matlab(prediction["force_nonlinear"]),
            "Fint_true": matlab(data.internal_force[test_indices]),
            "Fnonlinear_true": matlab(data.nonlinear_force[test_indices]),
            "networkMAE_m": network_metrics["mae_m"].reshape(1, -1),
            "networkRMSE_m": network_metrics["rmse_m"].reshape(1, -1),
            "networkRRMSE_percent": network_metrics[
                "rrmse_percent"
            ].reshape(1, -1),
            "networkCorrelation": network_metrics["correlation"].reshape(
                1, -1
            ),
            "sclMAE_m": scl_metrics["mae_m"].reshape(1, -1),
            "sclRMSE_m": scl_metrics["rmse_m"].reshape(1, -1),
            "sclRRMSE_percent": scl_metrics["rrmse_percent"].reshape(1, -1),
            "sclCorrelation": scl_metrics["correlation"].reshape(1, -1),
        },
        do_compression=True,
    )


def save_preview(
    path: Path,
    time: np.ndarray,
    test_indices: np.ndarray,
    true_displacement: np.ndarray,
    network_displacement: np.ndarray,
    scl_displacement: np.ndarray,
) -> int | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib unavailable; preview plot was not saved.")
        return None
    top_dof = true_displacement.shape[2] - 1
    local_index = int(
        np.argmax(np.max(np.abs(true_displacement[:, :, top_dof]), axis=1))
    )
    figure, axes = plt.subplots(
        true_displacement.shape[2], 1, figsize=(10.0, 10.0), sharex=True
    )
    axes = np.atleast_1d(axes)
    for dof, axis in enumerate(axes):
        axis.plot(
            time, true_displacement[local_index, :, dof],
            color="black", linewidth=1.0, label="MATLAB",
        )
        axis.plot(
            time, network_displacement[local_index, :, dof],
            color="tab:blue", linewidth=0.9, label="Network",
        )
        axis.plot(
            time, scl_displacement[local_index, :, dof],
            color="tab:orange", linewidth=0.8, linestyle="--", label="SCL",
        )
        axis.set_ylabel(f"DOF {dof + 1}\n(m)")
        axis.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(
        "Test sample MATLAB index "
        f"{int(test_indices[local_index]) + 1}: displacement comparison"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return local_index


def main() -> None:
    args = parse_args()
    run_dir, checkpoint_path, output_dir = resolve_paths(args)
    device = torch.device(args.device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    expected_method = "EPINN" if args.variant == "increment" else "EPINN_FULL"
    if checkpoint.get("method") != expected_method:
        raise ValueError(
            f"Checkpoint method {checkpoint.get('method')!r} does not match "
            f"--variant {args.variant!r}."
        )
    config = case_config_from_checkpoint(
        checkpoint, args.data_root.expanduser().resolve()
    )
    data = load_case_data(config)
    split = build_data_split(config, data.load.shape[0])
    tensors = as_torch_case(data, device)
    model = build_model(args.variant, checkpoint, tensors, config, device)
    chunk_length = int(
        args.chunk_length
        if args.chunk_length is not None
        else checkpoint.get("tbptt_length", 500)
    )

    print(f"Variant: {args.variant}")
    print(f"Run directory: {run_dir}")
    print(f"Selected checkpoint: {checkpoint_path.name}")
    print(
        f"Checkpoint epoch/val_loss: {checkpoint.get('epoch')}/"
        f"{checkpoint.get('val_loss'):.6e}"
    )
    print(
        f"Testing {split.test.size} samples on {device} with "
        f"batch_size={args.batch_size}, chunk_length={chunk_length}."
    )
    prediction = chunked_batched_predict(
        model,
        data.load,
        split.test,
        args.batch_size,
        chunk_length,
        device,
    )
    network_velocity, network_acceleration = central_difference(
        prediction["dis_nl"], data.delta_t
    )
    _, scl_acceleration = central_difference(prediction["dis"], data.delta_t)
    reference = data.displacement[split.test]
    network_metrics = displacement_metrics(reference, prediction["dis_nl"])
    scl_metrics = displacement_metrics(reference, prediction["dis"])
    print_metrics(network_metrics, scl_metrics)

    metrics_path = output_dir / "test_metrics_by_dof.csv"
    mat_path = output_dir / "test_results.mat"
    summary_path = output_dir / "test_summary.json"
    preview_path = output_dir / "test_preview.png"
    save_metrics_csv(metrics_path, network_metrics, scl_metrics)
    save_matlab_results(
        mat_path,
        args.variant,
        checkpoint_path,
        checkpoint,
        data,
        split.test,
        prediction,
        network_velocity,
        network_acceleration,
        scl_acceleration,
        network_metrics,
        scl_metrics,
    )
    preview_local_index = save_preview(
        preview_path,
        data.time,
        split.test,
        reference,
        prediction["dis_nl"],
        prediction["dis"],
    )
    summary = {
        "variant": args.variant,
        "run_directory": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", np.nan)),
        "test_sample_count": int(split.test.size),
        "test_indices_python": split.test.tolist(),
        "test_indices_matlab": (split.test + 1).tolist(),
        "network_overall_rmse_m": network_metrics["overall_rmse_m"],
        "network_overall_rrmse_percent": network_metrics[
            "overall_rrmse_percent"
        ],
        "scl_overall_rmse_m": scl_metrics["overall_rmse_m"],
        "scl_overall_rrmse_percent": scl_metrics["overall_rrmse_percent"],
        "preview_sample_local_index": preview_local_index,
        "preview_sample_matlab_index": (
            None
            if preview_local_index is None
            else int(split.test[preview_local_index]) + 1
        ),
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Saved MATLAB results: {mat_path}")
    print(f"Saved DOF metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    if preview_path.exists():
        print(f"Saved preview: {preview_path}")


if __name__ == "__main__":
    main()
