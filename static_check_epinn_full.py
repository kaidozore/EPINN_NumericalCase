"""Static checks for the independently added full-displacement E-PINN."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import CaseConfig
from nets.EPINN_Full_Loss import EPINN_MDOFSys_FullDis_PhyLoss
from nets.EPINN_Full_Net import EPINN_FullDis_PhyLSTM_NetBody
from utils.DataPreProcess import as_torch_case, load_case_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--check-steps", type=int, default=32)
    args = parser.parse_args()
    steps = max(4, int(args.check_steps))
    config = CaseConfig(data_root=args.data_root, time_truncation=steps)
    data = load_case_data(config)
    tensors = as_torch_case(data, torch.device("cpu"))
    model = EPINN_FullDis_PhyLSTM_NetBody(
        nLoad=data.load.shape[2],
        nLoadNL=data.load.shape[2],
        influence_kernel=tensors["kernel"],
        stiffness=tensors["stiffness"],
        fiber=tensors["fiber"],
        steel=tensors["steel"],
        input_increment_scale=config.displacement_increment_scale,
        input_displacement_scale=config.displacement_scale,
        output_displacement_scale=config.displacement_scale,
        hidden_size=8,
        fc_size=8,
    ).double()
    load = torch.as_tensor(
        data.load[:1, :steps].transpose(0, 2, 1)[:, None],
        dtype=torch.float64,
    )
    prediction = model(load)
    loss = EPINN_MDOFSys_FullDis_PhyLoss()(prediction)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not torch.isfinite(loss) or not all(
        value is not None and torch.isfinite(value).all()
        for value in gradients
    ):
        raise AssertionError("Full E-PINN forward/backward check failed.")
    print(
        "[PASS] Full E-PINN forward/SCL/loss/backward, "
        f"loss={float(loss.detach()):.6e}."
    )

    chunk_length = max(4, steps // 2)
    keys = ("elastic_dis", "dis_nl", "force_nonlinear", "dis", "vel")
    with torch.no_grad():
        whole = model(load)
        state = None
        pieces = {key: [] for key in keys}
        for start in range(0, steps, chunk_length):
            chunk, state = model.forward_chunk(
                load[..., start : start + chunk_length], state
            )
            for key in keys:
                pieces[key].append(chunk[key])
        errors = {
            key: float(
                torch.max(
                    torch.abs(torch.cat(pieces[key], dim=1) - whole[key])
                )
            )
            for key in keys
        }
    if max(errors.values()) > 1.0e-7:
        raise AssertionError(f"Full E-PINN TBPTT continuity failed: {errors}.")
    print(
        "[PASS] Full E-PINN TBPTT forward-state continuity, "
        f"max error={max(errors.values()):.3e}."
    )
    print("Full E-PINN static check completed; no optimizer step was executed.")


if __name__ == "__main__":
    main()
