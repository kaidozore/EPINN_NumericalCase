"""Increment-output PINN for the five-DOF wave-load system."""

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


class PINN_PhyLSTM3_Increment_NetBody(nn.Module):
    """Map elastic increments directly to elastoplastic increments."""

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
        output_increment_scale: float | torch.Tensor = 1.0e-1,
        hidden_size: int = 240,
        fc_size: int = 240,
    ) -> None:
        super().__init__()
        self.nLoad = int(nLoad)
        self.nDOF = int(nDOF)
        self.delta_t = float(delta_t)

        scale = torch.as_tensor(
            output_increment_scale,
            dtype=stiffness.dtype,
            device=stiffness.device,
        ).reshape(-1)
        if scale.numel() == 1:
            scale = scale.expand(nDOF).clone()
        if scale.numel() != nDOF or torch.any(scale <= 0.0):
            raise ValueError(
                "output_increment_scale must contain positive DOF scales."
            )
        self.register_buffer(
            "output_increment_scale", scale.reshape(1, 1, nDOF)
        )
        self.ElasticInput_Module = ElasticIncrementInput(
            influence_kernel, nDOF, input_increment_scale
        )
        self.LSTM_Module = LSTM_FC_Module(
            nLoad, nDOF, hidden_size, fc_size
        )
        # Keep the initial increment history close to zero while preserving
        # gradients. This is the same stable output-head initialization used
        # by the total-displacement PINN.
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
        displacement_increment = force_initial_zero(
            self.LSTM_Module(network_input) * self.output_increment_scale
        )
        displacement = torch.cumsum(displacement_increment, dim=1)
        velocity, acceleration = central_difference_kinematics(
            displacement, self.delta_t
        )
        force_internal, force_nonlinear = self.Constitutive_Module(
            displacement
        )
        return {
            "dis_increment": displacement_increment,
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
            "dis": displacement,
            "vel": velocity,
            "acc": acceleration,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
        }

    def forward_chunk(self, load, state=None, compute_physics: bool = True):
        """Evaluate a consecutive TBPTT chunk with continuous histories."""

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
        displacement_increment, lstm_state = self.LSTM_Module(
            network_input, lstm_state, True
        )
        displacement_increment = (
            displacement_increment * self.output_increment_scale
        )
        if state is None:
            displacement_increment = force_initial_zero(
                displacement_increment
            )
            previous_displacement = torch.zeros_like(
                displacement_increment[:, :1]
            )
            material_state = None
        else:
            previous_displacement = state["displacement"]
            material_state = state.get("material")

        displacement = previous_displacement + torch.cumsum(
            displacement_increment, dim=1
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
            "dis_increment": displacement_increment,
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
            "dis": displacement,
            "vel": velocity,
            "acc": acceleration,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
        }, next_state
