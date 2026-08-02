"""Conventional PINN baseline for the 5-DOF wave-load system."""

from __future__ import annotations

import torch
import torch.nn as nn

from nets.common import (
    ElasticIncrementInput,
    FiberSteel02Module,
    LSTM_FC_Module,
    force_initial_zero,
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
        output_increment_scale: float | torch.Tensor | None = None,
        hidden_size: int = 240,
        fc_size: int = 240,
    ) -> None:
        super().__init__()
        self.nLoad = nLoad
        self.nDOF = nDOF
        self.delta_t = float(delta_t)
        output_scale_tensor = torch.as_tensor(
            input_increment_scale
            if output_increment_scale is None
            else output_increment_scale,
            dtype=stiffness.dtype,
            device=stiffness.device,
        ).reshape(-1)
        if output_scale_tensor.numel() == 1:
            output_scale_tensor = output_scale_tensor.expand(nDOF).clone()
        if output_scale_tensor.numel() != nDOF:
            raise ValueError(
                "output_increment_scale must be scalar or have nDOF entries."
            )
        self.register_buffer(
            "output_increment_scale",
            output_scale_tensor.reshape(1, 1, nDOF),
        )
        self.ElasticInput_Module = ElasticIncrementInput(
            influence_kernel, nDOF, input_increment_scale
        )
        self.LSTM_Module = LSTM_FC_Module(
            nLoad, nDOF, hidden_size, fc_size
        )
        self.Constitutive_Module = FiberSteel02Module(
            stiffness, fiber, steel
        )

    def forward(self, load: torch.Tensor) -> dict[str, torch.Tensor]:
        load_sequence = load.squeeze(1).transpose(1, 2)
        network_input, elastic_increment, elastic_displacement = (
            self.ElasticInput_Module(load_sequence)
        )
        increment = force_initial_zero(
            self.LSTM_Module(network_input) * self.output_increment_scale
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
        increment, lstm_state = self.LSTM_Module(
            network_input, lstm_state, True
        )
        increment = increment * self.output_increment_scale
        if state is None:
            increment = force_initial_zero(increment)
            displacement0 = torch.zeros_like(increment[:, :1])
            velocity0 = torch.zeros_like(increment[:, :1])
            acceleration0 = torch.zeros_like(increment[:, :1])
            material_state = None
        else:
            displacement0 = state["displacement"]
            velocity0 = state["velocity"]
            acceleration0 = state["acceleration"]
            material_state = state.get("material")
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
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
            "dis": displacement,
            "vel": velocity,
            "acc": acceleration,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
        }, next_state
