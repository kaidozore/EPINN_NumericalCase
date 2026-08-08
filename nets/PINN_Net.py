"""Conventional PINN baseline for the 5-DOF wave-load system."""

from __future__ import annotations

import torch
import torch.nn as nn

from nets.common import (
    ElasticIncrementInput,
    FiberSteel02Module,
    LSTM_FC_Module,
    central_difference_kinematics,
    force_initial_zero,
)


class PINN_PhyLSTM3_DisIncrement_NetBody(nn.Module):
    """Predict the nonlinear total displacement from the wave-load history."""

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
        input_force_scale: float | torch.Tensor = 1.0e4,
        output_displacement_scale: float | torch.Tensor = 5.0e-1,
        hidden_size: int = 240,
        fc_size: int = 240,
    ) -> None:
        super().__init__()
        self.nLoad = nLoad
        self.nDOF = nDOF
        self.delta_t = float(delta_t)
        def fixed_scale(value, size, name):
            scale = torch.as_tensor(
                value, dtype=stiffness.dtype, device=stiffness.device
            ).reshape(-1)
            if scale.numel() == 1:
                scale = scale.expand(size).clone()
            if scale.numel() != size or torch.any(scale <= 0.0):
                raise ValueError(f"{name} must contain {size} positive scales.")
            return scale.reshape(1, 1, size)

        self.register_buffer(
            "input_force_scale",
            fixed_scale(input_force_scale, nLoad, "input_force_scale"),
        )
        self.register_buffer(
            "output_displacement_scale",
            fixed_scale(
                output_displacement_scale, nDOF, "output_displacement_scale"
            ),
        )
        self.ElasticInput_Module = ElasticIncrementInput(
            influence_kernel, nDOF, input_increment_scale
        )
        self.LSTM_Module = LSTM_FC_Module(
            nLoad, nDOF, hidden_size, fc_size
        )
        # A default linear head produces high-frequency random displacement;
        # numerical differentiation then amplifies it into enormous initial
        # acceleration.  Start near zero response without blocking gradients.
        nn.init.xavier_uniform_(self.LSTM_Module.FC2.weight, gain=1.0e-2)
        nn.init.zeros_(self.LSTM_Module.FC2.bias)
        self.Constitutive_Module = FiberSteel02Module(
            stiffness, fiber, steel
        )

    def forward(self, load: torch.Tensor) -> dict[str, torch.Tensor]:
        load_sequence = load.squeeze(1).transpose(1, 2)
        _, elastic_increment, elastic_displacement = (
            self.ElasticInput_Module(load_sequence)
        )
        network_input = load_sequence / self.input_force_scale
        displacement = force_initial_zero(
            self.LSTM_Module(network_input) * self.output_displacement_scale
        )
        increment = torch.diff(
            displacement,
            dim=1,
            prepend=torch.zeros_like(displacement[:, :1]),
        )
        velocity, acceleration = central_difference_kinematics(
            displacement, self.delta_t
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
            _,
            elastic_increment,
            elastic_displacement,
            elastic_state,
        ) = self.ElasticInput_Module.forward_chunk(
            load_sequence, elastic_state
        )
        network_input = load_sequence / self.input_force_scale
        lstm_state = None if state is None else state["lstm"]
        displacement, lstm_state = self.LSTM_Module(
            network_input, lstm_state, True
        )
        displacement = displacement * self.output_displacement_scale
        if state is None:
            displacement = force_initial_zero(displacement)
            previous_displacement = torch.zeros_like(displacement[:, :1])
            material_state = None
        else:
            previous_displacement = state["displacement"]
            material_state = state.get("material")
        increment = torch.cat(
            (
                displacement[:, :1] - previous_displacement,
                displacement[:, 1:] - displacement[:, :-1],
            ),
            dim=1,
        )
        velocity, acceleration = central_difference_kinematics(
            displacement,
            self.delta_t,
            None if state is None else previous_displacement,
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
