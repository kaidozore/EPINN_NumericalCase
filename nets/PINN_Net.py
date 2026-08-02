"""Conventional PINN baseline for the 5-DOF wave-load system."""

from __future__ import annotations

import torch
import torch.nn as nn

from nets.common import (
    FiberSteel02Module,
    LSTM_FC_Module,
    central_difference,
    force_initial_zero,
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
        hidden_size: int = 240,
        fc_size: int = 240,
    ) -> None:
        super().__init__()
        self.nLoad = nLoad
        self.nDOF = nDOF
        self.delta_t = float(delta_t)
        increment_scale_tensor = torch.as_tensor(
            increment_scale, dtype=load_scale.dtype, device=load_scale.device
        ).reshape(-1)
        if increment_scale_tensor.numel() == 1:
            increment_scale_tensor = increment_scale_tensor.expand(nDOF).clone()
        if increment_scale_tensor.numel() != nDOF:
            raise ValueError("increment_scale must be scalar or have nDOF entries.")
        self.register_buffer(
            "increment_scale_vector",
            increment_scale_tensor.reshape(1, 1, nDOF),
        )
        self.register_buffer("load_scale", load_scale.reshape(1, 1, nLoad))
        self.LSTM_Module = LSTM_FC_Module(
            nLoad, nDOF, hidden_size, fc_size
        )
        # A generic random output produces a small biased increment which is
        # amplified by a 5000-step cumulative sum.  Start from the exact zero
        # response; FC2 begins learning on the first optimizer update.
        nn.init.zeros_(self.LSTM_Module.FC2.weight)
        nn.init.zeros_(self.LSTM_Module.FC2.bias)
        self.Constitutive_Module = FiberSteel02Module(
            stiffness, fiber, steel
        )

    def forward(self, load: torch.Tensor) -> dict[str, torch.Tensor]:
        load_sequence = load.squeeze(1).transpose(1, 2)
        network_input = load_sequence / self.load_scale
        increment = force_initial_zero(
            self.LSTM_Module(network_input) * self.increment_scale_vector
        )
        displacement = torch.cumsum(increment, dim=1)
        velocity = central_difference(displacement, self.delta_t)
        acceleration = central_difference(velocity, self.delta_t)
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
        increment, lstm_state = self.LSTM_Module(
            network_input, lstm_state, True
        )
        increment = increment * self.increment_scale_vector
        if state is None:
            increment = force_initial_zero(increment)
            displacement0 = torch.zeros_like(increment[:, :1])
            material_state = None
        else:
            displacement0 = state["displacement"]
            material_state = state.get("material")
        displacement = displacement0 + torch.cumsum(increment, dim=1)
        velocity = central_difference(displacement, self.delta_t)
        if compute_physics:
            acceleration = central_difference(velocity, self.delta_t)
            force_internal, force_nonlinear, material_state = (
                self.Constitutive_Module.forward_chunk(
                    displacement, material_state
                )
            )
        else:
            acceleration = None
            force_internal = None
            force_nonlinear = None
        next_state = {
            "lstm": tuple(
                (hidden.detach(), cell.detach())
                for hidden, cell in lstm_state
            ),
            "displacement": displacement[:, -1:].detach(),
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
