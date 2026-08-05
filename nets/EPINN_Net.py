"""E-PINN network: nonlinear displacement increments plus fixed SCL."""

from __future__ import annotations

import torch
import torch.nn as nn

from nets.common import (
    ElasticIncrementInput,
    FiberSteel02Module,
    LSTM_FC_Module,
    SCL_Module,
    force_initial_zero,
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
        input_increment_scale: float = 1.0e-1,
        output_increment_scale: float | None = None,
        hidden_size: int = 120,
        fc_size: int = 120,
    ) -> None:
        super().__init__()
        if nLoadNL != nLoad:
            raise ValueError("All five reduced DOFs carry nonlinear fiber force.")
        self.nLoad = nLoad
        self.nLoadNL = nLoadNL
        self.output_increment_scale = float(
            input_increment_scale
            if output_increment_scale is None
            else output_increment_scale
        )
        self.ElasticInput_Module = ElasticIncrementInput(
            influence_kernel, nLoad, input_increment_scale
        )
        self.LSTM_Module = LSTM_FC_Module(
            nLoad, nLoadNL, hidden_size, fc_size
        )
        # Directly accumulated increments are highly sensitive to even a
        # small persistent output-head bias over a 5000-step history.  Start
        # close to zero while retaining nonzero weights so gradients reach
        # the FC and LSTM layers from the first optimizer step.
        nn.init.xavier_uniform_(self.LSTM_Module.FC2.weight, gain=1.0e-2)
        nn.init.zeros_(self.LSTM_Module.FC2.bias)
        self.Constitutive_Module = FiberSteel02Module(
            stiffness, fiber, steel
        )
        self.SCL_Module = SCL_Module(influence_kernel)

    def forward(self, load: torch.Tensor) -> dict[str, torch.Tensor]:
        # Reference-code layout: [batch, 1, nLoad, timeLength].
        load_sequence = load.squeeze(1).transpose(1, 2)
        network_input, elastic_increment, elastic_displacement = (
            self.ElasticInput_Module(load_sequence)
        )
        increment_nl = force_initial_zero(
            self.LSTM_Module(network_input) * self.output_increment_scale
        )
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
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
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
        increment_nl, lstm_state = self.LSTM_Module(
            network_input, lstm_state, True
        )
        increment_nl = increment_nl * self.output_increment_scale
        if state is None:
            increment_nl = force_initial_zero(increment_nl)
            displacement0 = torch.zeros_like(increment_nl[:, :1])
            material_state = None
            scl_history = None
            previous_scl_displacement = None
        else:
            displacement0 = state["displacement_nl"]
            material_state = state["material"]
            scl_history = state["scl_history"]
            previous_scl_displacement = state["scl_displacement"]
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
            "elastic": elastic_state,
            "displacement_nl": displacement_nl[:, -1:].detach(),
            "material": {
                key: value.detach() for key, value in material_state.items()
            },
            "scl_history": scl_input[:, -history_length:].detach(),
            "scl_displacement": displacement[:, -1:].detach(),
        }
        return {
            "dis_increment_nl": increment_nl,
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
            "dis_nl": displacement_nl,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
            "state": structural_state,
            "dis": displacement,
            "vel": velocity,
            "dis_increment_scl": increment_scl,
        }, next_state
