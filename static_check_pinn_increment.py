"""Static checks for the independent displacement-increment PINN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import CaseConfig
from nets.EPINN_Net import EPINN_PhyLSTM_NetBody
from nets.PINN_Increment_Loss import PINN_MDOFSys_Increment_PhyLoss
from nets.PINN_Increment_Net import PINN_PhyLSTM3_Increment_NetBody
from nets.common import central_difference_kinematics
from utils.DataPreProcess import as_torch_case, load_case_data


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--check-steps", type=int, default=64)
    return parser.parse_args()


def finite_gradients(model: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(gradients) and all(
        torch.all(torch.isfinite(value)) for value in gradients
    )


def main() -> None:
    args = parse_args()
    config = CaseConfig(
        data_root=args.data_root, sequence_length=args.check_steps
    )
    data = load_case_data(config)
    tensors = as_torch_case(data, torch.device("cpu"))
    steps = data.load.shape[1]
    load = torch.as_tensor(
        data.load[:1].transpose(0, 2, 1)[:, None], dtype=torch.float64
    )
    common = dict(
        nLoad=data.load.shape[2],
        stiffness=tensors["stiffness"],
        influence_kernel=tensors["kernel"],
        fiber=tensors["fiber"],
        steel=tensors["steel"],
        input_increment_scale=config.displacement_increment_scale,
        hidden_size=8,
        fc_size=8,
    )

    model = PINN_PhyLSTM3_Increment_NetBody(
        nDOF=data.displacement.shape[2],
        delta_t=data.delta_t,
        output_increment_scale=config.displacement_increment_scale,
        **common,
    ).double()
    prediction = model(load)
    loss_module = PINN_MDOFSys_Increment_PhyLoss(
        tensors["mass"], tensors["damping"], tensors["stiffness"]
    ).double()
    loss = loss_module(load, prediction)

    load_sequence = load.squeeze(1).transpose(1, 2)
    force_without_elastic = (
        torch.einsum("ij,btj->bti", tensors["mass"], prediction["acc"])
        + torch.einsum("ij,btj->bti", tensors["damping"], prediction["vel"])
        + prediction["force_nonlinear"]
        - load_sequence
    )
    q = torch.einsum(
        "ij,btj->bti", torch.linalg.inv(tensors["stiffness"]),
        force_without_elastic,
    )
    delta_q = torch.diff(q, dim=1, prepend=torch.zeros_like(q[:, :1]))
    manual = torch.mean((prediction["dis_increment"] + delta_q).pow(2))
    if not torch.allclose(loss, manual, rtol=1.0e-12, atol=1.0e-14):
        raise AssertionError("Increment PINN loss does not match its equation.")
    loss.backward()
    if not torch.isfinite(loss) or not finite_gradients(model):
        raise AssertionError("Increment PINN forward/loss/backward failed.")
    print(
        f"[PASS] Increment PINN equation/forward/backward, "
        f"loss={loss.item():.6e}."
    )

    # The equilibrium-derived displacement must be differenced continuously
    # across TBPTT boundaries. With identical kinematics, chunked and whole
    # losses must therefore be exactly equivalent after length weighting.
    boundary = steps // 2
    loss_state = None
    chunk_total = loss.new_zeros(())
    for start, stop in ((0, boundary), (boundary, steps)):
        part = {
            key: value[:, start:stop]
            for key, value in prediction.items()
            if isinstance(value, torch.Tensor) and value.ndim == 3
        }
        chunk_loss, loss_state = loss_module(
            load[..., start:stop],
            part,
            previous_equilibrium_displacement=loss_state,
            return_state=True,
        )
        chunk_total = chunk_total + chunk_loss * ((stop - start) / steps)
    loss_continuity_error = torch.abs(chunk_total - loss).item()
    if loss_continuity_error > 1.0e-14:
        raise AssertionError("Increment loss is discontinuous across TBPTT.")
    print(
        "[PASS] Increment-loss TBPTT differencing continuity, "
        f"error={loss_continuity_error:.3e}."
    )

    # The ideal MATLAB displacement is evaluated with the same central
    # differences used during PINN training to establish a numerical floor.
    true_u = torch.as_tensor(data.displacement, dtype=torch.float64)
    true_du = torch.diff(
        true_u, dim=1, prepend=torch.zeros_like(true_u[:, :1])
    )
    true_v, true_a = central_difference_kinematics(true_u, data.delta_t)
    true_rnl = torch.as_tensor(
        data.internal_force
        - np.einsum("ij,btj->bti", data.stiffness, data.displacement),
        dtype=torch.float64,
    )
    true_load = torch.as_tensor(
        data.load.transpose(0, 2, 1)[:, None], dtype=torch.float64
    )
    true_loss = loss_module(
        true_load,
        {
            "dis_increment": true_du,
            "vel": true_v,
            "acc": true_a,
            "force_nonlinear": true_rnl,
        },
    )
    print(
        f"[PASS] MATLAB-response increment-loss reference="
        f"{true_loss.item():.6e} m^2."
    )

    epinn = EPINN_PhyLSTM_NetBody(
        nLoadNL=data.load.shape[2],
        output_increment_scale=config.displacement_increment_scale,
        **common,
    ).double()
    captured: list[torch.Tensor] = []
    hook = epinn.LSTM_Module.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    epinn_prediction = epinn(load)
    hook.remove()
    expected_input = (
        epinn_prediction["elastic_dis_increment"]
        / epinn.ElasticInput_Module.increment_scale
    )
    input_error = torch.max(torch.abs(captured[0] - expected_input)).item()
    if input_error > 1.0e-14:
        raise AssertionError("E-PINN LSTM input is not the elastic increment.")
    print(
        "[PASS] Increment E-PINN LSTM input is the fixed-scale elastic "
        f"displacement increment; max error={input_error:.3e}."
    )
    print(
        f"Static check completed for {steps} steps; no optimizer step was run."
    )


if __name__ == "__main__":
    main()
