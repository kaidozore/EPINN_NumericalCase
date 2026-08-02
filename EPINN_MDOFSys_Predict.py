"""Predict the 100 held-out samples with a trained E-PINN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import CaseConfig
from nets.EPINN_Net import EPINN_PhyLSTM_NetBody
from utils.DataPreProcess import (
    as_torch_case,
    build_data_split,
    load_case_data,
)
from utils.predict_utils import (
    acceleration_from_equilibrium,
    batched_predict,
    print_displacement_metrics,
    save_matlab_output,
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DataSet/EPINN/Outputs/MDOFSys_Wave_EPINN.mat"),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    stored_config = checkpoint["case_config"]
    config = CaseConfig(
        data_root=args.data_root,
        time_truncation=int(stored_config["time_truncation"]),
        sequence_length=stored_config.get("sequence_length"),
        displacement_increment_scale=float(
            stored_config["displacement_increment_scale"]
        ),
    )
    data = load_case_data(config)
    split = build_data_split(config, data.load.shape[0])
    tensors = as_torch_case(data, device)
    model = EPINN_PhyLSTM_NetBody(
        nLoad=checkpoint["n_load"],
        nLoadNL=checkpoint["n_dof"],
        influence_kernel=tensors["kernel"],
        stiffness=tensors["stiffness"],
        fiber=tensors["fiber"],
        steel=tensors["steel"],
        load_scale=torch.as_tensor(
            checkpoint["load_scale"], dtype=torch.float64
        ),
        increment_scale=config.displacement_increment_scale,
        hidden_size=checkpoint["hidden_size"],
        fc_size=checkpoint["fc_size"],
    ).double().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    prediction = batched_predict(
        model, data.load, split.test, args.batch_size, device
    )
    prediction["acc"] = acceleration_from_equilibrium(
        data.load[split.test],
        prediction["dis"],
        prediction["vel"],
        prediction["force_internal"],
        data.mass,
        data.damping,
    )
    print_displacement_metrics(
        data.displacement[split.test], prediction["dis"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_matlab_output(
        args.output, prediction, data.time, split.test
    )
    print(f"Saved E-PINN predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
