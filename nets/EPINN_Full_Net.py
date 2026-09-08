"""Full-displacement E-PINN for the 5-DOF wave-load system."""

from __future__ import annotations

import torch
import torch.nn as nn

from nets.common import (
    CausalTransformer_FC_Module,
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
        sequence_model: str = "lstm",
        stitch_mode: str = "hidden",
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        transformer_ff_size: int | None = None,
        transformer_memory_length: int = 128,
    ) -> None:
        super().__init__()
        if nLoadNL != nLoad:
            raise ValueError("All five reduced DOFs carry nonlinear fiber force.")
        if input_displacement_scale <= 0.0 or output_displacement_scale <= 0.0:
            raise ValueError("Displacement scales must be positive.")
        self.nLoad = int(nLoad)
        self.nLoadNL = int(nLoadNL)
        self.sequence_model = sequence_model.lower()
        self.stitch_mode = stitch_mode.lower()
        if self.sequence_model not in {"lstm", "transformer"}:
            raise ValueError("sequence_model must be 'lstm' or 'transformer'.")
        if self.stitch_mode not in {"hidden", "explicit-overlap"}:
            raise ValueError(
                "stitch_mode must be 'hidden' or 'explicit-overlap'."
            )
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
        temporal_input_size = nLoad + (1 if self.stitch_mode == "explicit-overlap" else 0)
        if self.sequence_model == "lstm":
            self.LSTM_Module = LSTM_FC_Module(
                temporal_input_size, nLoadNL, hidden_size, fc_size
            )
            temporal_module = self.LSTM_Module
        else:
            self.Transformer_Module = CausalTransformer_FC_Module(
                temporal_input_size,
                nLoadNL,
                hidden_size,
                fc_size,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                feedforward_size=transformer_ff_size,
                memory_length=transformer_memory_length,
            )
            temporal_module = self.Transformer_Module
        nn.init.xavier_uniform_(temporal_module.FC2.weight, gain=1.0e-2)
        nn.init.zeros_(temporal_module.FC2.bias)
        self.Constitutive_Module = FiberSteel02Module(stiffness, fiber, steel)
        self.SCL_Module = SCL_Module(influence_kernel)

    def _temporal_forward(
        self,
        network_input: torch.Tensor,
        initial_displacement: torch.Tensor | None,
        temporal_state=None,
    ):
        """Return current-step predictions and optional overlap quantities."""
        if self.stitch_mode == "explicit-overlap":
            if initial_displacement is None:
                initial_displacement = torch.zeros_like(network_input[:, :1])
            initial_token = torch.cat(
                (
                    initial_displacement / self.input_displacement_scale,
                    torch.ones_like(initial_displacement[:, :, :1]),
                ),
                dim=2,
            )
            response_tokens = torch.cat(
                (
                    network_input,
                    torch.zeros_like(network_input[:, :, :1]),
                ),
                dim=2,
            )
            tokens = torch.cat((initial_token, response_tokens), dim=1)
            # Explicit stitching deliberately resets hidden/attention memory;
            # only the physical terminal displacement is passed between spans.
            output, _ = self._temporal_module()(tokens, None, True)
            output = output * self.output_displacement_scale
            return output[:, 1:], None, output[:, :1], initial_displacement
        output, temporal_state = self._temporal_module()(
            network_input, temporal_state, True
        )
        return (
            output * self.output_displacement_scale,
            temporal_state,
            None,
            None,
        )

    def _temporal_module(self):
        return (
            self.LSTM_Module
            if self.sequence_model == "lstm"
            else self.Transformer_Module
        )

    def _detach_temporal_state(self, state):
        if state is None:
            return None
        if self.sequence_model == "lstm":
            return tuple(
                (hidden.detach(), cell.detach()) for hidden, cell in state
            )
        return {
            "memory": (
                None if state["memory"] is None else state["memory"].detach()
            ),
            "position": int(state["position"]),
        }

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
        displacement_nl, _, boundary_prediction, boundary_initial = (
            self._temporal_forward(network_input, None)
        )
        displacement_nl = force_initial_zero(displacement_nl)
        force_internal, force_nonlinear = self.Constitutive_Module(
            displacement_nl
        )
        state, displacement, velocity = self._structural_convolution(
            load_sequence, force_nonlinear
        )
        result = {
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
            "dis_nl": displacement_nl,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
            "state": state,
            "dis": displacement,
            "vel": velocity,
        }
        if boundary_prediction is not None:
            result["boundary_prediction"] = boundary_prediction
            result["boundary_initial"] = boundary_initial
        return result

    def forward_chunk(self, load: torch.Tensor, state=None):
        """Evaluate one consecutive TBPTT chunk with physical histories."""

        load_sequence = load.squeeze(1).transpose(1, 2)
        elastic_state = None if state is None else state["elastic"]
        (
            _, elastic_increment, elastic_displacement, elastic_state,
        ) = self.ElasticInput_Module.forward_chunk(load_sequence, elastic_state)
        network_input = elastic_displacement / self.input_displacement_scale
        temporal_state = None if state is None else state.get("temporal")
        explicit_initial = (
            None if state is None else state.get("explicit_displacement")
        )
        (
            displacement_nl,
            temporal_state,
            boundary_prediction,
            boundary_initial,
        ) = self._temporal_forward(
            network_input, explicit_initial, temporal_state
        )
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
            "temporal": self._detach_temporal_state(temporal_state),
            "elastic": elastic_state,
            "explicit_displacement": displacement_nl[:, -1:].detach(),
            "material": {
                key: value.detach() for key, value in material_state.items()
            },
            "scl_history": scl_input[:, -history_length:].detach(),
        }
        result = {
            "elastic_dis_increment": elastic_increment,
            "elastic_dis": elastic_displacement,
            "dis_nl": displacement_nl,
            "force_internal": force_internal,
            "force_nonlinear": force_nonlinear,
            "state": structural_state,
            "dis": displacement,
            "vel": velocity,
        }
        if boundary_prediction is not None:
            result["boundary_prediction"] = boundary_prediction
            result["boundary_initial"] = boundary_initial
        return result, next_state
