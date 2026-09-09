"""E-PINN network: nonlinear displacement increments plus fixed SCL."""

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
        input_displacement_scale: float = 5.0e-1,
        sequence_model: str = "lstm",
        stitch_mode: str = "hidden",
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        transformer_ff_size: int | None = None,
        transformer_memory_length: int = 128,
        reset_lstm_state: bool = False,
    ) -> None:
        super().__init__()
        if nLoadNL != nLoad:
            raise ValueError("All five reduced DOFs carry nonlinear fiber force.")
        self.nLoad = nLoad
        self.nLoadNL = nLoadNL
        self.sequence_model = sequence_model.lower()
        self.stitch_mode = stitch_mode.lower()
        self.reset_lstm_state = bool(reset_lstm_state)
        if self.reset_lstm_state and (self.sequence_model != "lstm" or self.stitch_mode != "hidden"):
            raise ValueError("reset_lstm_state requires lstm with hidden stitching.")
        if self.sequence_model not in {"lstm", "transformer"}:
            raise ValueError("sequence_model must be 'lstm' or 'transformer'.")
        if self.stitch_mode not in {"hidden", "explicit-overlap"}:
            raise ValueError(
                "stitch_mode must be 'hidden' or 'explicit-overlap'."
            )
        if input_displacement_scale <= 0.0:
            raise ValueError("input_displacement_scale must be positive.")
        # Plain scalar (not a state-dict buffer) keeps older increment-model
        # checkpoints loadable without migration.
        self.input_displacement_scale = float(input_displacement_scale)
        self.output_increment_scale = float(
            input_increment_scale
            if output_increment_scale is None
            else output_increment_scale
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
        # Directly accumulated increments are highly sensitive to even a
        # small persistent output-head bias over a 5000-step history.  Start
        # near the elastic response scale, while retaining a conservative
        # amplitude so the cumulative displacement does not immediately
        # drift.  The nonzero weights let gradients reach the FC and LSTM
        # layers from the first optimizer step.
        self.output_head_init_gain = 5.0e-2
        nn.init.xavier_uniform_(
            temporal_module.FC2.weight,
            gain=self.output_head_init_gain,
        )
        nn.init.zeros_(temporal_module.FC2.bias)
        self.Constitutive_Module = FiberSteel02Module(
            stiffness, fiber, steel
        )
        self.SCL_Module = SCL_Module(influence_kernel)

    def _temporal_module(self):
        return (
            self.LSTM_Module
            if self.sequence_model == "lstm"
            else self.Transformer_Module
        )

    def _temporal_forward(
        self,
        network_input: torch.Tensor,
        initial_displacement: torch.Tensor | None,
        temporal_state=None,
        previous_increment: torch.Tensor | None = None,
    ):
        """Every output is an increment, including the duplicated boundary."""
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
            output, _ = self._temporal_module()(tokens, None, True)
            boundary_prediction = output[:, :1] * self.output_increment_scale
            boundary_increment = (
                torch.zeros_like(boundary_prediction)
                if previous_increment is None else previous_increment
            )
            increment = output[:, 1:] * self.output_increment_scale
            return increment, None, boundary_prediction, boundary_increment
        increment, temporal_state = self._temporal_module()(
            network_input, temporal_state, True
        )
        return (
            increment * self.output_increment_scale,
            temporal_state,
            None,
            None,
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

    def forward(self, load: torch.Tensor) -> dict[str, torch.Tensor]:
        # Reference-code layout: [batch, 1, nLoad, timeLength].
        load_sequence = load.squeeze(1).transpose(1, 2)
        network_input, elastic_increment, elastic_displacement = (
            self.ElasticInput_Module(load_sequence)
        )
        increment_nl, _, boundary_prediction, boundary_initial = (
            self._temporal_forward(network_input, None)
        )
        increment_nl = force_initial_zero(increment_nl)
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
        result = {
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
        if boundary_prediction is not None:
            result["boundary_prediction"] = boundary_prediction
            result["boundary_initial"] = boundary_initial
        return result

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
        temporal_state = None if state is None else state.get("temporal")
        if self.reset_lstm_state:
            temporal_state = None  # Only neural memory resets; physical history persists.
        explicit_initial = (
            None if state is None else state.get("explicit_displacement")
        )
        (
            increment_nl,
            temporal_state,
            boundary_prediction,
            boundary_initial,
        ) = self._temporal_forward(
            network_input, explicit_initial, temporal_state,
            previous_increment=None if state is None else state["last_increment"],
        )
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
            "temporal": self._detach_temporal_state(temporal_state),
            "elastic": elastic_state,
            "displacement_nl": displacement_nl[:, -1:].detach(),
            "explicit_displacement": displacement_nl[:, -1:].detach(),
            "last_increment": increment_nl[:, -1:].detach(),
            "material": {
                key: value.detach() for key, value in material_state.items()
            },
            "scl_history": scl_input[:, -history_length:].detach(),
            "scl_displacement": displacement[:, -1:].detach(),
        }
        result = {
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
        }
        if boundary_prediction is not None:
            result["boundary_prediction"] = boundary_prediction
            result["boundary_initial"] = boundary_initial
        return result, next_state
