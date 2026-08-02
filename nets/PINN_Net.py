"""Conventional PINN baseline for the 5-DOF wave-load system."""

from __future__ import annotations

import torch
import torch.nn as nn

from nets.common import (
    FiberSteel02Module,
    LSTM_FC_Module,
    increment_from_bounded_level,
    newmark_average_acceleration_kinematics,
)


class PINN_PhyLSTM3_DisIncrement_NetBody(nn.Module):
    """Predict all global displacement increments without an SCL module."""

    def __init__(
        self,
        nLoad: int,
        nDOF: int,
        delta_t: float,
        stiffness: torch.Tensor,
        fiber: dict[str, torch.Tensor],
        steel: dict[str, float],
        load_scale: torch.Tensor,
        increment_scale: float = 1.0e-4,
        displacement_scale: float | torch.Tensor | None = None,
        hidden_size: int = 240,
        fc_size: int = 240,
    ) -> None:
        super().__init__()
        self.nLoad = nLoad
        self.nDOF = nDOF
        self.delta_t = float(delta_t)
        level_scale_tensor = torch.as_tensor(
            increment_scale if displacement_scale is None else displacement_scale,
            dtype=load_scale.dtype,
            device=load_scale.device,
        ).reshape(-1)
        if level_scale_tensor.numel() == 1:
            level_scale_tensor = level_scale_tensor.expand(nDOF).clone()
        if level_scale_tensor.numel() != nDOF:
            raise ValueError("displacement_scale must be scalar or have nDOF entries.")
        self.register_buffer(
            "level_scale_vector",
            level_scale_tensor.reshape(1, 1, nDOF),
        )
        self.register_buffer("load_scale", load_scale.reshape(1, 1, nLoad))
        self.LSTM_Module = LSTM_FC_Module(
            nLoad, nDOF, hidden_size, fc_size
        )
        # Increments are now differences of a bounded level, so the default
        # Linear initialization is safe and lets gradients reach all LSTM/FC
        # layers from the first update.  Zero-initializing FC2 here collapses
        # the long-history network to the exact zero-response baseline.
        self.Constitutive_Module = FiberSteel02Module(
            stiffness, fiber, steel
        )

    def forward(self, load: torch.Tensor) -> dict[str, torch.Tensor]:
        load_sequence = load.squeeze(1).transpose(1, 2)
        network_input = load_sequence / self.load_scale
        level = self.LSTM_Module(network_input) * self.level_scale_vector
        increment = increment_from_bounded_level(
            level
        )
        displacement = torch.cumsum(increment, dim=1)
        velocity, acceleration = newmark_average_acceleration_kinematics(
            increment, self.delta_t
        )
        force_internal, force_nonlinear = self.Constitutive_Module(
            displacement
        )
        return {
            "dis_increment": increment,
            "dis": displacement,
            "vel": velocity,
            "acc": acceleration,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
        }

    def forward_chunk(self, load, state=None, compute_physics: bool = True):
        """Evaluate one consecutive TBPTT chunk and return detached history."""
        load_sequence = load.squeeze(1).transpose(1, 2)
        network_input = load_sequence / self.load_scale
        lstm_state = None if state is None else state["lstm"]
        level, lstm_state = self.LSTM_Module(
            network_input, lstm_state, True
        )
        level = level * self.level_scale_vector
        if state is None:
            previous_level = None
            displacement0 = torch.zeros_like(level[:, :1])
            velocity0 = torch.zeros_like(level[:, :1])
            acceleration0 = torch.zeros_like(level[:, :1])
            material_state = None
        else:
            previous_level = state["network_level"]
            displacement0 = state["displacement"]
            velocity0 = state["velocity"]
            acceleration0 = state["acceleration"]
            material_state = state.get("material")
        increment = increment_from_bounded_level(level, previous_level)
        displacement = displacement0 + torch.cumsum(increment, dim=1)
        velocity, acceleration = newmark_average_acceleration_kinematics(
            increment, self.delta_t, velocity0, acceleration0
        )
        if compute_physics:
            force_internal, force_nonlinear, material_state = (
                self.Constitutive_Module.forward_chunk(
                    displacement, material_state
                )
            )
        else:
            force_internal = None
            force_nonlinear = None
        next_state = {
            "lstm": tuple(
                (hidden.detach(), cell.detach())
                for hidden, cell in lstm_state
            ),
            "displacement": displacement[:, -1:].detach(),
            "network_level": level[:, -1:].detach(),
            "velocity": velocity[:, -1:].detach(),
            "acceleration": acceleration[:, -1:].detach(),
            "material": (
                None
                if material_state is None
                else {
                    key: value.detach()
                    for key, value in material_state.items()
                }
            ),
        }
        return {
            "dis_increment": increment,
            "dis": displacement,
            "vel": velocity,
            "acc": acceleration,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
        }, next_state
