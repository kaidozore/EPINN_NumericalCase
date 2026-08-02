"""E-PINN network: nonlinear displacement increments plus fixed SCL."""

from __future__ import annotations

import torch
import torch.nn as nn

from nets.common import (
    FiberSteel02Module,
    LSTM_FC_Module,
    SCL_Module,
    increment_from_bounded_level,
)


class EPINN_PhyLSTM_NetBody(nn.Module):
    """Predict five increments and recover the state through Steel02 and SCL."""

    def __init__(
        self,
        nLoad: int,
        nLoadNL: int,
        influence_kernel: torch.Tensor,
        stiffness: torch.Tensor,
        fiber: dict[str, torch.Tensor],
        steel: dict[str, float],
        load_scale: torch.Tensor,
        increment_scale: float = 1.0e-4,
        displacement_scale: float | None = None,
        hidden_size: int = 120,
        fc_size: int = 120,
    ) -> None:
        super().__init__()
        if nLoadNL != nLoad:
            raise ValueError("All five reduced DOFs carry nonlinear fiber force.")
        self.nLoad = nLoad
        self.nLoadNL = nLoadNL
        self.level_scale = float(
            increment_scale if displacement_scale is None else displacement_scale
        )
        self.register_buffer("load_scale", load_scale.reshape(1, 1, nLoad))
        self.LSTM_Module = LSTM_FC_Module(
            nLoad, nLoadNL, hidden_size, fc_size
        )
        nn.init.zeros_(self.LSTM_Module.FC2.weight)
        nn.init.zeros_(self.LSTM_Module.FC2.bias)
        self.Constitutive_Module = FiberSteel02Module(
            stiffness, fiber, steel
        )
        self.SCL_Module = SCL_Module(influence_kernel)

    def forward(self, load: torch.Tensor) -> dict[str, torch.Tensor]:
        # Reference-code layout: [batch, 1, nLoad, timeLength].
        load_sequence = load.squeeze(1).transpose(1, 2)
        network_input = load_sequence / self.load_scale
        level_nl = self.LSTM_Module(network_input) * self.level_scale
        increment_nl = increment_from_bounded_level(level_nl)
        displacement_nl = torch.cumsum(increment_nl, dim=1)
        force_internal, force_nonlinear = self.Constitutive_Module(
            displacement_nl
        )

        # MATLAB ETDM uses LF=[I,-I] and g=fint-K0*u.  Therefore g is
        # appended without another sign change.
        load_total = torch.cat([load_sequence, force_nonlinear], dim=2)
        state = self.SCL_Module(load_total)
        displacement = state[:, :, : self.nLoad]
        velocity = state[:, :, self.nLoad :]
        displacement_nl_scl = displacement
        increment_nl_scl = torch.cat(
            [
                torch.zeros_like(displacement_nl_scl[:, :1, :]),
                displacement_nl_scl[:, 1:, :]
                - displacement_nl_scl[:, :-1, :],
            ],
            dim=1,
        )
        return {
            "dis_increment_nl": increment_nl,
            "dis_nl": displacement_nl,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
            "state": state,
            "dis": displacement,
            "vel": velocity,
            "dis_increment_scl": increment_nl_scl,
        }

    def forward_chunk(self, load, state=None):
        """Evaluate one consecutive TBPTT chunk with all physical histories."""
        load_sequence = load.squeeze(1).transpose(1, 2)
        network_input = load_sequence / self.load_scale
        lstm_state = None if state is None else state["lstm"]
        level_nl, lstm_state = self.LSTM_Module(
            network_input, lstm_state, True
        )
        level_nl = level_nl * self.level_scale
        if state is None:
            previous_level = None
            displacement0 = torch.zeros_like(level_nl[:, :1])
            material_state = None
            scl_history = None
            previous_scl_displacement = None
        else:
            previous_level = state["network_level_nl"]
            displacement0 = state["displacement_nl"]
            material_state = state["material"]
            scl_history = state["scl_history"]
            previous_scl_displacement = state["scl_displacement"]
        increment_nl = increment_from_bounded_level(
            level_nl, previous_level
        )
        displacement_nl = displacement0 + torch.cumsum(increment_nl, dim=1)
        force_internal, force_nonlinear, material_state = (
            self.Constitutive_Module.forward_chunk(
                displacement_nl, material_state
            )
        )
        load_total = torch.cat([load_sequence, force_nonlinear], dim=2)
        scl_input = (
            load_total
            if scl_history is None
            else torch.cat([scl_history, load_total], dim=1)
        )
        state_total = self.SCL_Module(scl_input)
        structural_state = state_total[:, -load_total.shape[1] :]
        displacement = structural_state[:, :, : self.nLoad]
        velocity = structural_state[:, :, self.nLoad :]
        first_increment = (
            torch.zeros_like(displacement[:, :1])
            if previous_scl_displacement is None
            else displacement[:, :1] - previous_scl_displacement
        )
        increment_scl = torch.cat(
            [first_increment, displacement[:, 1:] - displacement[:, :-1]],
            dim=1,
        )
        history_length = self.SCL_Module.timeTrun - 1
        next_state = {
            "lstm": tuple(
                (hidden.detach(), cell.detach())
                for hidden, cell in lstm_state
            ),
            "displacement_nl": displacement_nl[:, -1:].detach(),
            "network_level_nl": level_nl[:, -1:].detach(),
            "material": {
                key: value.detach() for key, value in material_state.items()
            },
            "scl_history": scl_input[:, -history_length:].detach(),
            "scl_displacement": displacement[:, -1:].detach(),
        }
        return {
            "dis_increment_nl": increment_nl,
            "dis_nl": displacement_nl,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
            "state": structural_state,
            "dis": displacement,
            "vel": velocity,
            "dis_increment_scl": increment_scl,
        }, next_state
