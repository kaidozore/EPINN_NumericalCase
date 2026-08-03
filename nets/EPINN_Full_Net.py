"""Full-displacement E-PINN for the 5-DOF wave-load system."""

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


class EPINN_FullDis_PhyLSTM_NetBody(nn.Module):
    """Predict total displacement and enforce consistency through the SCL."""

    def __init__(
        self,
        nLoad: int,
        nLoadNL: int,
        influence_kernel: torch.Tensor,
        stiffness: torch.Tensor,
        fiber: dict[str, torch.Tensor],
        steel: dict[str, float],
        input_increment_scale: float = 1.0e-1,
        input_displacement_scale: float = 5.0e-1,
        output_displacement_scale: float = 5.0e-1,
        hidden_size: int = 120,
        fc_size: int = 120,
    ) -> None:
        super().__init__()
        if nLoadNL != nLoad:
            raise ValueError("All five reduced DOFs carry nonlinear fiber force.")
        if input_displacement_scale <= 0.0 or output_displacement_scale <= 0.0:
            raise ValueError("Displacement scales must be positive.")
        self.nLoad = int(nLoad)
        self.nLoadNL = int(nLoadNL)
        self.register_buffer(
            "input_displacement_scale",
            stiffness.new_tensor(float(input_displacement_scale)),
        )
        self.register_buffer(
            "output_displacement_scale",
            stiffness.new_tensor(float(output_displacement_scale)),
        )
        self.ElasticInput_Module = ElasticIncrementInput(
            influence_kernel, nLoad, input_increment_scale
        )
        self.LSTM_Module = LSTM_FC_Module(
            nLoad, nLoadNL, hidden_size, fc_size
        )
        nn.init.xavier_uniform_(self.LSTM_Module.FC2.weight, gain=1.0e-2)
        nn.init.zeros_(self.LSTM_Module.FC2.bias)
        self.Constitutive_Module = FiberSteel02Module(stiffness, fiber, steel)
        self.SCL_Module = SCL_Module(influence_kernel)

    def _structural_convolution(
        self,
        load_sequence: torch.Tensor,
        force_nonlinear: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        load_total = torch.cat((load_sequence, force_nonlinear), dim=2)
        state = self.SCL_Module(load_total)
        return state, state[:, :, : self.nLoad], state[:, :, self.nLoad :]

    def forward(self, load: torch.Tensor) -> dict[str, torch.Tensor]:
        load_sequence = load.squeeze(1).transpose(1, 2)
        _, elastic_increment, elastic_displacement = self.ElasticInput_Module(
            load_sequence
        )
        network_input = elastic_displacement / self.input_displacement_scale
        displacement_nl = force_initial_zero(
            self.LSTM_Module(network_input) * self.output_displacement_scale
        )
        force_internal, force_nonlinear = self.Constitutive_Module(
            displacement_nl
        )
        state, displacement, velocity = self._structural_convolution(
            load_sequence, force_nonlinear
        )
        return {
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
            "dis_nl": displacement_nl,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
            "state": state,
            "dis": displacement,
            "vel": velocity,
        }

    def forward_chunk(self, load: torch.Tensor, state=None):
        """Evaluate one consecutive TBPTT chunk with physical histories."""

        load_sequence = load.squeeze(1).transpose(1, 2)
        elastic_state = None if state is None else state["elastic"]
        (
            _, elastic_increment, elastic_displacement, elastic_state,
        ) = self.ElasticInput_Module.forward_chunk(load_sequence, elastic_state)
        network_input = elastic_displacement / self.input_displacement_scale
        lstm_state = None if state is None else state["lstm"]
        displacement_nl, lstm_state = self.LSTM_Module(
            network_input, lstm_state, True
        )
        displacement_nl = displacement_nl * self.output_displacement_scale
        if state is None:
            displacement_nl = force_initial_zero(displacement_nl)
            material_state = None
            scl_history = None
        else:
            material_state = state["material"]
            scl_history = state["scl_history"]
        force_internal, force_nonlinear, material_state = (
            self.Constitutive_Module.forward_chunk(
                displacement_nl, material_state
            )
        )
        load_total = torch.cat((load_sequence, force_nonlinear), dim=2)
        scl_input = (
            load_total
            if scl_history is None
            else torch.cat((scl_history, load_total), dim=1)
        )
        state_total = self.SCL_Module(scl_input)
        structural_state = state_total[:, -load_total.shape[1] :]
        displacement = structural_state[:, :, : self.nLoad]
        velocity = structural_state[:, :, self.nLoad :]
        history_length = self.SCL_Module.timeTrun - 1
        next_state = {
            "lstm": tuple(
                (hidden.detach(), cell.detach())
                for hidden, cell in lstm_state
            ),
            "elastic": elastic_state,
            "material": {
                key: value.detach() for key, value in material_state.items()
            },
            "scl_history": scl_input[:, -history_length:].detach(),
        }
        return {
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
            "dis_nl": displacement_nl,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
            "state": structural_state,
            "dis": displacement,
            "vel": velocity,
        }, next_state
