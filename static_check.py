"""Static and short differentiability checks; this script never trains."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import CaseConfig
from nets.common import (
    FiberSteel02Module,
    SCL_Module,
    newmark_average_acceleration_kinematics,
)
from nets.EPINN_Loss import EPINN_MDOFSys_DisIncrement_PhyLoss
from nets.EPINN_Net import EPINN_PhyLSTM_NetBody
from nets.PINN_Loss import PINN_MDOFSys_DisIncrement_PhyLoss
from nets.PINN_Net import PINN_PhyLSTM3_DisIncrement_NetBody
from utils.DataPreProcess import (
    as_torch_case,
    build_data_split,
    load_case_data,
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--check-steps", type=int, default=32)
    parser.add_argument("--constitutive-steps", type=int, default=5000)
    return parser.parse_args()


def finite_gradients(model: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return bool(gradients) and all(
        value is not None and torch.isfinite(value).all() for value in gradients
    )


def main() -> None:
    args = parse_args()
    check_steps = max(3, int(args.check_steps))
    config = CaseConfig(
        data_root=args.data_root,
        time_truncation=check_steps,
    )
    data = load_case_data(config)
    split = build_data_split(config, data.load.shape[0])
    device = torch.device("cpu")
    tensors = as_torch_case(data, device)
    n_dof = data.load.shape[2]
    assert n_dof == 5
    assert data.mass.shape == data.damping.shape == data.stiffness.shape == (5, 5)
    assert data.influence_kernel.shape[1:] == (10, 10)
    print(
        "[PASS] MATLAB input: Fwave/U/V/Acc=[sample,time,5], "
        "M/C/K0=5x5, ETDM/A=[lag,10,10]."
    )

    # The PINN predicts displacement increments, while MATLAB used the
    # average-acceleration Newmark scheme.  Confirm that the same discrete
    # kinematics recover MATLAB velocity and acceleration to roundoff.
    true_displacement = torch.as_tensor(data.displacement[:1], dtype=torch.float64)
    true_increment = torch.diff(
        true_displacement,
        dim=1,
        prepend=torch.zeros_like(true_displacement[:, :1]),
    )
    recovered_velocity, recovered_acceleration = (
        newmark_average_acceleration_kinematics(true_increment, data.delta_t)
    )
    true_velocity = torch.as_tensor(data.velocity[:1], dtype=torch.float64)
    true_acceleration = torch.as_tensor(data.acceleration[:1], dtype=torch.float64)
    velocity_error = torch.sqrt(torch.mean((recovered_velocity - true_velocity) ** 2))
    acceleration_error = torch.sqrt(
        torch.mean((recovered_acceleration - true_acceleration) ** 2)
    )
    acceleration_rms = torch.sqrt(torch.mean(true_acceleration ** 2))
    relative_acceleration_error = float(acceleration_error / acceleration_rms)
    if relative_acceleration_error > 1.0e-7:
        raise AssertionError("Newmark velocity/acceleration recovery failed.")
    print(
        "[PASS] Newmark kinematics: velocity RMSE="
        f"{float(velocity_error):.3e}, acceleration relative RMSE="
        f"{relative_acceleration_error:.3e}."
    )

    # Verify the Python constitutive port against forces already computed by
    # MATLAB.  The sample with the largest displacement includes yielding.
    critical = int(np.argmax(np.max(np.abs(data.displacement), axis=(1, 2))))
    constitutive_steps = min(int(args.constitutive_steps), data.load.shape[1])
    constitutive = FiberSteel02Module(
        tensors["stiffness"], tensors["fiber"], tensors["steel"]
    ).double()
    with torch.no_grad():
        displacement = torch.as_tensor(
            data.displacement[critical : critical + 1, :constitutive_steps],
            dtype=torch.float64,
        )
        force, nonlinear_force, reduced_tangent = (
            constitutive._forward_state_commit(displacement)
        )
    force_reference = torch.as_tensor(
        data.internal_force[critical : critical + 1, :constitutive_steps],
        dtype=torch.float64,
    )
    force_error = torch.max(torch.abs(force - force_reference)).item()
    nonlinear_reference = force_reference - torch.einsum(
        "ij,btj->bti", tensors["stiffness"], displacement
    )
    nonlinear_error = torch.max(
        torch.abs(nonlinear_force - nonlinear_reference)
    ).item()
    if force_error > 1.0e-5 or nonlinear_error > 1.0e-5:
        raise AssertionError("Steel02 fiber-force reproduction failed.")
    print(
        f"[PASS] Steel02 fiber force: sample={critical + 1}, "
        f"steps={constitutive_steps}, max errors="
        f"{force_error:.3e}/{nonlinear_error:.3e} N."
    )

    # Check the local tangent used by the custom backward against the exact
    # trial response at one strongly excited time point.  Earlier committed
    # material states remain identical in every finite-difference evaluation.
    tangent_step = int(
        torch.argmax(torch.linalg.vector_norm(displacement[0], dim=1)).item()
    )
    tangent_prefix = displacement[:, : tangent_step + 1].clone()
    perturbation = 1.0e-7
    finite_difference_columns = []
    with torch.no_grad():
        for dof in range(n_dof):
            plus = tangent_prefix.clone()
            minus = tangent_prefix.clone()
            plus[0, -1, dof] += perturbation
            minus[0, -1, dof] -= perturbation
            force_plus = constitutive._forward_state_commit(plus)[0][0, -1]
            force_minus = constitutive._forward_state_commit(minus)[0][0, -1]
            finite_difference_columns.append(
                (force_plus - force_minus) / (2.0 * perturbation)
            )
    finite_difference_tangent = torch.stack(finite_difference_columns, dim=1)
    tangent_exact = reduced_tangent[0, tangent_step]
    tangent_relative_error = (
        torch.linalg.matrix_norm(finite_difference_tangent - tangent_exact)
        / torch.clamp(torch.linalg.matrix_norm(tangent_exact), min=1.0)
    ).item()
    if tangent_relative_error > 2.0e-5:
        raise AssertionError("Steel02 local tangent check failed.")
    print(
        f"[PASS] Steel02 reduced tangent: step={tangent_step + 1}, "
        f"relative finite-difference error={tangent_relative_error:.3e}."
    )

    # Check Aij orientation, nonlinear-force sign and initial-state correction.
    scl = SCL_Module(tensors["kernel"]).double()
    scl_input = torch.as_tensor(
        np.concatenate(
            [
                data.load[:, :check_steps],
                data.nonlinear_force[:, :check_steps],
            ],
            axis=2,
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        state = scl(scl_input)
    state_reference = torch.as_tensor(
        np.concatenate(
            [
                data.etdm_displacement[:, :check_steps],
                data.etdm_velocity[:, :check_steps],
            ],
            axis=2,
        ),
        dtype=torch.float64,
    )
    scl_error = torch.max(torch.abs(state - state_reference)).item()
    if scl_error > 1.0e-10:
        raise AssertionError("ETDM/SCL reconstruction failed.")
    print(f"[PASS] ETDM/SCL reconstruction max error={scl_error:.3e}.")

    load = torch.as_tensor(
        data.load[:1, :check_steps].transpose(0, 2, 1)[:, None],
        dtype=torch.float64,
    )
    common = dict(
        nLoad=n_dof,
        stiffness=tensors["stiffness"],
        influence_kernel=tensors["kernel"],
        fiber=tensors["fiber"],
        steel=tensors["steel"],
        input_increment_scale=config.displacement_increment_scale,
        hidden_size=8,
        fc_size=8,
    )
    pinn = PINN_PhyLSTM3_DisIncrement_NetBody(
        nDOF=n_dof,
        delta_t=data.delta_t,
        input_displacement_scale=config.displacement_scale,
        output_displacement_scale=config.displacement_scale,
        **common,
    ).double()
    pinn_prediction = pinn(load)
    pinn_loss = PINN_MDOFSys_DisIncrement_PhyLoss(
        tensors["mass"], tensors["damping"], tensors["stiffness"]
    )(load, pinn_prediction)
    pinn_loss.backward()
    if not torch.isfinite(pinn_loss) or not finite_gradients(pinn):
        raise AssertionError("PINN forward/backward check failed.")
    print(f"[PASS] PINN forward/loss/backward, loss={pinn_loss.item():.6e}.")

    epinn = EPINN_PhyLSTM_NetBody(
        nLoadNL=n_dof,
        output_increment_scale=config.displacement_increment_scale,
        **common,
    ).double()
    epinn_prediction = epinn(load)
    epinn_loss_module = EPINN_MDOFSys_DisIncrement_PhyLoss(
        increment_scale=config.displacement_increment_scale,
        displacement_scale=config.displacement_scale,
        increment_loss_weight=0.1,
    ).double()
    epinn_target = {
        "dis": torch.as_tensor(
            data.displacement[:1, :check_steps], dtype=torch.float64
        )
    }
    epinn_loss, epinn_metrics = epinn_loss_module(
        epinn_prediction, epinn_target, return_metrics=True
    )
    epinn_loss.backward()
    if not torch.isfinite(epinn_loss) or not finite_gradients(epinn):
        raise AssertionError("E-PINN forward/backward check failed.")
    if not all(
        torch.isfinite(value) for value in epinn_metrics.values()
    ):
        raise AssertionError("E-PINN monitoring metrics are invalid.")
    print(
        f"[PASS] E-PINN forward/SCL/loss/backward, "
        f"loss={epinn_loss.item():.6e}, "
        "true displacement correlation="
        f"{epinn_metrics['true_displacement_correlation'].item():.6f}."
    )

    # Chunking must preserve forward histories before optimizer updates.
    chunk_length = max(3, check_steps // 2)
    with torch.no_grad():
        for name, model, keys in (
            (
                "PINN", pinn,
                ("elastic_dis_increment", "dis", "force_nonlinear"),
            ),
            (
                "E-PINN", epinn,
                ("elastic_dis_increment", "dis_nl", "force_nonlinear", "dis"),
            ),
        ):
            whole = model(load)
            chunk_state = None
            pieces = {key: [] for key in keys}
            for start in range(0, check_steps, chunk_length):
                chunk, chunk_state = model.forward_chunk(
                    load[..., start : start + chunk_length], chunk_state
                )
                for key in keys:
                    pieces[key].append(chunk[key])
            key_errors = {
                key: torch.max(
                    torch.abs(torch.cat(pieces[key], dim=1) - whole[key])
                ).item()
                for key in keys
            }
            chunk_error = max(key_errors.values())
            if chunk_error > 1.0e-7:
                raise AssertionError(
                    f"{name} chunk-state continuity failed: {key_errors}."
                )
            print(
                f"[PASS] {name} TBPTT forward-state continuity, "
                f"max error={chunk_error:.3e}."
            )
    print("Static check completed; no optimizer step or training was executed.")


if __name__ == "__main__":
    main()
