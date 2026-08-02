"""Conventional PINN baseline for the 5-DOF wave-load system."""

from __future__ import annotations

import torch
import torch.nn as nn

from nets.common import (
    ElasticIncrementInput,
    FiberSteel02Module,
    LSTM_FC_Module,
    increment_from_bounded_level,
    newmark_average_acceleration_kinematics,
)


class PINN_PhyLSTM3_DisIncrement_NetBody(nn.Module):
    """Predict global increments from fixed-SCL elastic increments."""

    def __init__(
        self,
        nLoad: int,
        nDOF: int,
        delta_t: float,
        stiffness: torch.Tensor,
        influence_kernel: torch.Tensor,
        fiber: dict[str, torch.Tensor],
        steel: dict[str, float],
        input_increment_scale: float | torch.Tensor = 1.0e-1,
        displacement_scale: float | torch.Tensor | None = None,
        hidden_size: int = 240,
        fc_size: int = 240,
    ) -> None:
        super().__init__()
        self.nLoad = nLoad
        self.nDOF = nDOF
        self.delta_t = float(delta_t)
        level_scale_tensor = torch.as_tensor(
            input_increment_scale if displacement_scale is None else displacement_scale,
            dtype=stiffness.dtype,
            device=stiffness.device,
        ).reshape(-1)
        if level_scale_tensor.numel() == 1:
            level_scale_tensor = level_scale_tensor.expand(nDOF).clone()
        if level_scale_tensor.numel() != nDOF:
            raise ValueError("displacement_scale must be scalar or have nDOF entries.")
        self.register_buffer(
            "level_scale_vector",
            level_scale_tensor.reshape(1, 1, nDOF),
        )
        self.ElasticInput_Module = ElasticIncrementInput(
            influence_kernel, nDOF, input_increment_scale
        )
        self.LSTM_Module = LSTM_FC_Module(
            nLoad, nDOF, hidden_size, fc_size
        )
        # Start close to the elastic skip connection while retaining nonzero
        # gradients through all LSTM/FC layers from the first update.
        nn.init.xavier_uniform_(self.LSTM_Module.FC2.weight, gain=1.0e-2)
        nn.init.zeros_(self.LSTM_Module.FC2.bias)
        self.Constitutive_Module = FiberSteel02Module(
            stiffness, fiber, steel
        )

    def forward(self, load: torch.Tensor) -> dict[str, torch.Tensor]:
        load_sequence = load.squeeze(1).transpose(1, 2)
        network_input, elastic_increment, elastic_displacement = (
            self.ElasticInput_Module(load_sequence)
        )
        correction_level = (
            self.LSTM_Module(network_input) * self.level_scale_vector
        )
        correction_increment = increment_from_bounded_level(correction_level)
        increment = elastic_increment + correction_increment
        displacement = torch.cumsum(increment, dim=1)
        velocity, acceleration = newmark_average_acceleration_kinematics(
            increment, self.delta_t
        )
        force_internal, force_nonlinear = self.Constitutive_Module(
            displacement
        )
        return {
            "dis_increment": increment,
            "correction_dis_increment": correction_increment,
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
            "dis": displacement,
            "vel": velocity,
            "acc": acceleration,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
        }

    def forward_chunk(self, load, state=None, compute_physics: bool = True):
        """Evaluate one consecutive TBPTT chunk and return detached history."""
        load_sequence = load.squeeze(1).transpose(1, 2)
        elastic_state = None if state is None else state["elastic"]
        (
            network_input,
            elastic_increment,
            elastic_displacement,
            elastic_state,
        ) = self.ElasticInput_Module.forward_chunk(
            load_sequence, elastic_state
        )
        lstm_state = None if state is None else state["lstm"]
        correction_level, lstm_state = self.LSTM_Module(
            network_input, lstm_state, True
        )
        correction_level = correction_level * self.level_scale_vector
        if state is None:
            previous_level = None
            displacement0 = torch.zeros_like(correction_level[:, :1])
            velocity0 = torch.zeros_like(correction_level[:, :1])
            acceleration0 = torch.zeros_like(correction_level[:, :1])
            material_state = None
        else:
            previous_level = state["network_level"]
            displacement0 = state["displacement"]
            velocity0 = state["velocity"]
            acceleration0 = state["acceleration"]
            material_state = state.get("material")
        correction_increment = increment_from_bounded_level(
            correction_level, previous_level
        )
        increment = elastic_increment + correction_increment
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
            "network_level": correction_level[:, -1:].detach(),
            "elastic": elastic_state,
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
            "correction_dis_increment": correction_increment,
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
            "dis": displacement,
            "vel": velocity,
            "acc": acceleration,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
        }, next_state
