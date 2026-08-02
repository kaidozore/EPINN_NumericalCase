"""Shared LSTM, Steel02 fiber, SCL and differentiation modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nn_fun

from extensions import steel02_native


class _FiberSteel02LocalTangentFunction(torch.autograd.Function):
    """Exact Steel02 history in forward; local consistent tangent in backward."""

    @staticmethod
    def forward(ctx, displacement: torch.Tensor, module):
        internal_force, nonlinear_force, tangent = module._forward_state_commit(
            displacement
        )
        ctx.save_for_backward(tangent, module.K0)
        ctx.set_materialize_grads(False)
        return internal_force, nonlinear_force

    @staticmethod
    def backward(ctx, grad_internal, grad_nonlinear):
        tangent, stiffness = ctx.saved_tensors
        grad_displacement = None
        if grad_internal is not None:
            grad_displacement = torch.einsum(
                "btij,bti->btj", tangent, grad_internal
            )
        if grad_nonlinear is not None:
            nonlinear_tangent = tangent - stiffness[None, None, :, :]
            nonlinear_gradient = torch.einsum(
                "btij,bti->btj", nonlinear_tangent, grad_nonlinear
            )
            grad_displacement = (
                nonlinear_gradient
                if grad_displacement is None
                else grad_displacement + nonlinear_gradient
            )
        return grad_displacement, None


class _FiberSteel02ChunkFunction(torch.autograd.Function):
    """Chunked Steel02 evaluation with committed state passed between chunks."""

    @staticmethod
    def forward(ctx, displacement, module, *initial_state):
        state = dict(zip(module.state_names, initial_state))
        internal, nonlinear, tangent, final_state = (
            module._forward_state_commit(displacement, state, True)
        )
        ctx.save_for_backward(tangent, module.K0)
        ctx.state_count = len(initial_state)
        final_values = tuple(final_state[name] for name in module.state_names)
        ctx.mark_non_differentiable(*final_values)
        return internal, nonlinear, *final_values

    @staticmethod
    def backward(ctx, grad_internal, grad_nonlinear, *grad_state):
        tangent, stiffness = ctx.saved_tensors
        gradient = None
        if grad_internal is not None:
            gradient = torch.einsum("btij,bti->btj", tangent, grad_internal)
        if grad_nonlinear is not None:
            value = torch.einsum(
                "btij,bti->btj",
                tangent - stiffness[None, None],
                grad_nonlinear,
            )
            gradient = value if gradient is None else gradient + value
        return (gradient, None, *([None] * ctx.state_count))


class LSTM_FC_Module(nn.Module):
    """Three LSTM layers followed by FC-ReLU-FC.

    No activation function is inserted between the LSTM layers, in accordance
    with Section 3 of the supplied jacket-platform manuscript.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        fc_size: int | None = None,
    ) -> None:
        super().__init__()
        fc_size = hidden_size if fc_size is None else fc_size
        self.LSTM1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.LSTM2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.LSTM3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.FC1 = nn.Linear(hidden_size, fc_size)
        self.FC2 = nn.Linear(fc_size, output_size)
        self.Relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        state=None,
        return_state: bool = False,
    ):
        state = (None, None, None) if state is None else state
        x, state1 = self.LSTM1(x, state[0])
        x, state2 = self.LSTM2(x, state[1])
        x, state3 = self.LSTM3(x, state[2])
        x = self.FC1(x)
        x = self.Relu(x)
        output = self.FC2(x)
        if return_state:
            return output, (state1, state2, state3)
        return output


class FiberSteel02Module(nn.Module):
    """Differentiable reduced restoring force of the MATLAB fiber beam.

    All geometry, fiber, Steel02 and condensation quantities are supplied by
    ``wave_loads_300.mat/cfg``.  Python does not assemble or redefine M, C or
    K0.  The only operations performed here are the same kinematic recovery,
    material recurrence and virtual-work projection used by MATLAB.
    """

    state_names = (
        "eps", "sig", "Et", "kon", "epspl", "epss0", "sigs0",
        "epsr", "sigr", "epsmax", "epsmin",
    )

    def __init__(
        self,
        stiffness: torch.Tensor,
        fiber: dict[str, torch.Tensor],
        steel: dict[str, float],
    ) -> None:
        super().__init__()
        self.register_buffer("K0", stiffness)
        transformation = fiber["transformation"]
        element_dof = fiber["element_dof"].long()
        length = float(fiber["element_length"])
        gauss_xi = fiber["gauss_xi"]
        gauss_weight = fiber["gauss_weight"]

        n_element = int(element_dof.shape[0])
        n_reduced = int(transformation.shape[1])
        local_map = transformation.new_zeros(n_element, 4, n_reduced)
        for element in range(n_element):
            for local in range(4):
                matlab_dof = int(element_dof[element, local])
                if matlab_dof > 0:
                    local_map[element, local] = transformation[matlab_dof - 1]

        curvature_rows = []
        for value in gauss_xi:
            xi = (value + 1.0) / 2.0
            curvature_rows.append(
                torch.stack(
                    [
                        (-6.0 + 12.0 * xi) / length**2,
                        (-4.0 + 6.0 * xi) / length,
                        (6.0 - 12.0 * xi) / length**2,
                        (-2.0 + 6.0 * xi) / length,
                    ]
                )
            )
        beam_b = torch.stack(curvature_rows)
        curvature_map = torch.einsum("gl,elr->egr", beam_b, local_map)
        force_map = curvature_map * (gauss_weight * length / 2.0)[None, :, None]
        self.register_buffer("curvature_map", curvature_map)
        self.register_buffer("force_map", force_map)
        self.register_buffer("fiber_y", fiber["fiber_y"])
        self.register_buffer("fiber_area", fiber["fiber_area"])
        for name, value in steel.items():
            self.register_buffer(name, stiffness.new_tensor(float(value)))

    @staticmethod
    def _safe_denominator(value: torch.Tensor) -> torch.Tensor:
        one = torch.ones_like(value)
        return torch.where(torch.abs(value) > 1.0e-18, value, one)

    def _steel02_step(
        self, strain: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Vectorized port of ``Steel02_single.m`` for every section fiber."""

        eps_p = state["eps"]
        sig_p = state["sig"]
        et_p = state["Et"]
        deps = strain - eps_p
        unchanged = torch.abs(deps) <= 1.0e-14
        epsy = self.Fy / self.E0
        esh = self.b * self.E0

        epsmax = state["epsmax"]
        epsmin = state["epsmin"]
        epspl = state["epspl"]
        epss0 = state["epss0"]
        sigs0 = state["sigs0"]
        epsr = state["epsr"]
        sigr = state["sigr"]
        kon = state["kon"]

        initial = ((kon == 0) | (kon == 3)) & ~unchanged
        negative = deps < 0
        epsmax = torch.where(initial, epsy, epsmax)
        epsmin = torch.where(initial, -epsy, epsmin)
        kon = torch.where(initial & negative, kon.new_tensor(2), kon)
        kon = torch.where(initial & ~negative, kon.new_tensor(1), kon)
        epss0 = torch.where(initial & negative, -epsy, epss0)
        epss0 = torch.where(initial & ~negative, epsy, epss0)
        sigs0 = torch.where(initial & negative, -self.Fy, sigs0)
        sigs0 = torch.where(initial & ~negative, self.Fy, sigs0)
        epspl = torch.where(initial & negative, -epsy, epspl)
        epspl = torch.where(initial & ~negative, epsy, epspl)

        reverse_positive = (kon == 2) & (deps > 0) & ~unchanged
        old_epsmax = epsmax
        epsmin_pos = torch.minimum(epsmin, eps_p)
        d1_pos = (old_epsmax - epsmin_pos) / (2.0 * self.a4 * epsy)
        shift_pos = 1.0 + self.a3 * torch.pow(d1_pos, 0.8)
        epss0_pos = (
            self.Fy * shift_pos
            - esh * epsy * shift_pos
            - sig_p
            + self.E0 * eps_p
        ) / (self.E0 - esh)
        sigs0_pos = self.Fy * shift_pos + esh * (
            epss0_pos - epsy * shift_pos
        )
        kon = torch.where(reverse_positive, kon.new_tensor(1), kon)
        epsr = torch.where(reverse_positive, eps_p, epsr)
        sigr = torch.where(reverse_positive, sig_p, sigr)
        epsmin = torch.where(reverse_positive, epsmin_pos, epsmin)
        epss0 = torch.where(reverse_positive, epss0_pos, epss0)
        sigs0 = torch.where(reverse_positive, sigs0_pos, sigs0)
        epspl = torch.where(reverse_positive, old_epsmax, epspl)

        reverse_negative = (kon == 1) & (deps < 0) & ~unchanged
        old_epsmin = epsmin
        epsmax_neg = torch.maximum(epsmax, eps_p)
        d1_neg = (epsmax_neg - old_epsmin) / (2.0 * self.a2 * epsy)
        shift_neg = 1.0 + self.a1 * torch.pow(d1_neg, 0.8)
        epss0_neg = (
            -self.Fy * shift_neg
            + esh * epsy * shift_neg
            - sig_p
            + self.E0 * eps_p
        ) / (self.E0 - esh)
        sigs0_neg = -self.Fy * shift_neg + esh * (
            epss0_neg + epsy * shift_neg
        )
        kon = torch.where(reverse_negative, kon.new_tensor(2), kon)
        epsr = torch.where(reverse_negative, eps_p, epsr)
        sigr = torch.where(reverse_negative, sig_p, sigr)
        epsmax = torch.where(reverse_negative, epsmax_neg, epsmax)
        epss0 = torch.where(reverse_negative, epss0_neg, epss0)
        sigs0 = torch.where(reverse_negative, sigs0_neg, sigs0)
        epspl = torch.where(reverse_negative, old_epsmin, epspl)

        xi = torch.abs((epspl - epss0) / epsy)
        radius = self.R0 * (1.0 - self.cR1 * xi / (self.cR2 + xi))
        denominator = self._safe_denominator(epss0 - epsr)
        eps_ratio = (strain - epsr) / denominator
        dum1 = 1.0 + torch.abs(eps_ratio).pow(radius)
        dum2 = dum1.pow(1.0 / radius)
        stress = (
            self.b * eps_ratio + (1.0 - self.b) * eps_ratio / dum2
        ) * (sigs0 - sigr) + sigr
        tangent = (
            self.b + (1.0 - self.b) / (dum1 * dum2)
        ) * (sigs0 - sigr) / denominator
        stress = torch.where(unchanged, sig_p, stress)
        tangent = torch.where(unchanged, et_p, tangent)

        new_state = {
            "eps": torch.where(unchanged, eps_p, strain),
            "sig": stress,
            "Et": tangent,
            "kon": torch.where(unchanged, state["kon"], kon),
            "epspl": torch.where(unchanged, state["epspl"], epspl),
            "epss0": torch.where(unchanged, state["epss0"], epss0),
            "sigs0": torch.where(unchanged, state["sigs0"], sigs0),
            "epsr": torch.where(unchanged, state["epsr"], epsr),
            "sigr": torch.where(unchanged, state["sigr"], sigr),
            "epsmax": torch.where(unchanged, state["epsmax"], epsmax),
            "epsmin": torch.where(unchanged, state["epsmin"], epsmin),
        }
        return stress, tangent, new_state

    def initial_state(self, displacement: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = (
            displacement.shape[0],
            self.curvature_map.shape[0],
            self.curvature_map.shape[1],
            self.fiber_y.shape[-1],
        )
        zeros = displacement.new_zeros(shape)
        epsy = self.Fy / self.E0
        return {
            "eps": zeros,
            "sig": zeros,
            "Et": torch.ones_like(zeros) * self.E0,
            "kon": zeros.to(torch.int8),
            "epspl": zeros,
            "epss0": zeros,
            "sigs0": zeros,
            "epsr": zeros,
            "sigr": zeros,
            "epsmax": torch.ones_like(zeros) * epsy,
            "epsmin": -torch.ones_like(zeros) * epsy,
        }

    def _forward_state_commit(
        self,
        displacement: torch.Tensor,
        state: dict[str, torch.Tensor] | None = None,
        return_state: bool = False,
    ):
        """Return force and reduced tangent using trial/commit history updates."""

        state = self.initial_state(displacement) if state is None else state
        if steel02_native.available(displacement.device):
            packed_state = torch.stack(
                [state[name].to(displacement.dtype) for name in self.state_names],
                dim=-1,
            )
            parameters = torch.stack(
                [
                    self.E0, self.Fy, self.b, self.R0, self.cR1, self.cR2,
                    self.a1, self.a2, self.a3, self.a4,
                ]
            )
            internal_force, reduced_tangent, packed_final = steel02_native.forward(
                displacement,
                self.curvature_map,
                self.force_map,
                self.fiber_y.reshape(-1),
                self.fiber_area.reshape(-1),
                parameters,
                packed_state,
            )
            final_state = {
                name: packed_final[..., index]
                for index, name in enumerate(self.state_names)
            }
            final_state["kon"] = final_state["kon"].to(torch.int8)
            elastic_force = torch.einsum(
                "ij,btj->bti", self.K0, displacement
            )
            result = (
                internal_force,
                internal_force - elastic_force,
                reduced_tangent,
            )
            return (*result, final_state) if return_state else result

        curvature = torch.einsum(
            "btr,egr->bteg", displacement, self.curvature_map
        )
        strain = -curvature.unsqueeze(-1) * self.fiber_y
        force_history = []
        tangent_history = []
        area_y = self.fiber_area * self.fiber_y
        area_y2 = self.fiber_area * self.fiber_y.square()
        for index in range(displacement.shape[1]):
            stress, material_tangent, state = self._steel02_step(
                strain[:, index], state
            )
            moment = -torch.sum(stress * area_y, dim=-1)
            force = torch.einsum("beg,egr->br", moment, self.force_map)
            section_tangent = torch.sum(material_tangent * area_y2, dim=-1)
            tangent = torch.einsum(
                "beg,egr,egs->brs",
                section_tangent,
                self.force_map,
                self.curvature_map,
            )
            force_history.append(force)
            tangent_history.append(tangent)
        internal_force = torch.stack(force_history, dim=1)
        reduced_tangent = torch.stack(tangent_history, dim=1)
        elastic_force = torch.einsum("ij,btj->bti", self.K0, displacement)
        result = (internal_force, internal_force - elastic_force, reduced_tangent)
        return (*result, state) if return_state else result

    def forward(
        self, displacement: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(fint, fint-K0*u)`` with a memory-bounded backward pass."""

        return _FiberSteel02LocalTangentFunction.apply(displacement, self)

    def forward_chunk(self, displacement, state=None):
        state = self.initial_state(displacement) if state is None else state
        values = _FiberSteel02ChunkFunction.apply(
            displacement,
            self,
            *(state[name] for name in self.state_names),
        )
        final_state = dict(zip(self.state_names, values[2:]))
        return values[0], values[1], final_state


class SCL_Module(nn.Module):
    """Fixed structural-convolution layer built from the MATLAB ETDM kernel."""

    def __init__(
        self,
        influence_kernel: torch.Tensor,
    ) -> None:
        super().__init__()
        if influence_kernel.ndim != 3:
            raise ValueError("influence_kernel must have shape [lag, input, state].")
        # conv1d performs cross-correlation; reverse the lag axis for a causal
        # convolution after left padding.
        weight = influence_kernel.flip(0).permute(2, 1, 0).contiguous()
        self.register_buffer("ResWeight", weight)
        self.timeTrun = int(influence_kernel.shape[0])

    def forward(self, load: torch.Tensor) -> torch.Tensor:
        """Map ``[batch,time,input]`` to ``[batch,time,state]``."""

        x = load.transpose(1, 2)
        x = nn_fun.pad(x, (self.timeTrun - 1, 0))
        state = nn_fun.conv1d(x, self.ResWeight)
        return state.transpose(1, 2)


def force_initial_zero(increment: torch.Tensor) -> torch.Tensor:
    """Impose the known zero-displacement initial condition exactly."""

    return torch.cat(
        [torch.zeros_like(increment[:, :1, :]), increment[:, 1:, :]], dim=1
    )


def increment_from_bounded_level(
    level: torch.Tensor,
    previous_level: torch.Tensor | None = None,
) -> torch.Tensor:
    """Form increments by differencing a bounded network state.

    Directly accumulating 5000 independently predicted increments turns a
    tiny output bias into an unbounded displacement drift.  Differencing a
    learned level retains displacement-increment output and residual offsets,
    while making the cumulative displacement telescope to a bounded state.
    """

    if level.ndim != 3 or level.shape[1] < 1:
        raise ValueError("level must have shape [batch,time,dof].")
    if previous_level is None:
        first = torch.zeros_like(level[:, :1])
    else:
        if previous_level.shape != level[:, :1].shape:
            raise ValueError("previous_level must have shape [batch,1,dof].")
        first = level[:, :1] - previous_level
    return torch.cat((first, level[:, 1:] - level[:, :-1]), dim=1)


def central_difference(
    response: torch.Tensor, delta_t: float
) -> torch.Tensor:
    """Differentiable first derivative with second-order boundary formulas."""

    if response.shape[1] < 3:
        raise ValueError("At least three time points are required.")
    first = (
        -3.0 * response[:, 0, :]
        + 4.0 * response[:, 1, :]
        - response[:, 2, :]
    ) / (2.0 * delta_t)
    middle = (
        response[:, 2:, :] - response[:, :-2, :]
    ) / (2.0 * delta_t)
    last = (
        3.0 * response[:, -1, :]
        - 4.0 * response[:, -2, :]
        + response[:, -3, :]
    ) / (2.0 * delta_t)
    return torch.cat(
        [first.unsqueeze(1), middle, last.unsqueeze(1)], dim=1
    )


def newmark_average_acceleration_kinematics(
    displacement_increment: torch.Tensor,
    delta_t: float,
    initial_velocity: torch.Tensor | None = None,
    initial_acceleration: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover velocity/acceleration for Newmark beta=1/4, gamma=1/2.

    Alternating signs express the Newmark recurrences as cumulative sums, so
    the calculation remains vectorized and preserves exact TBPTT continuity.
    """

    if displacement_increment.ndim != 3:
        raise ValueError("Expected displacement increments [batch,time,dof].")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive.")
    batch, steps, dof = displacement_increment.shape
    zero = displacement_increment.new_zeros((batch, 1, dof))
    velocity0 = zero if initial_velocity is None else initial_velocity
    acceleration0 = zero if initial_acceleration is None else initial_acceleration
    if velocity0.shape != zero.shape or acceleration0.shape != zero.shape:
        raise ValueError("Initial Newmark states must have shape [batch,1,dof].")

    indices = torch.arange(steps, device=displacement_increment.device)
    signs = torch.where(
        indices.remainder(2) == 0,
        displacement_increment.new_tensor(-1.0),
        displacement_increment.new_tensor(1.0),
    ).reshape(1, steps, 1)

    velocity_rhs = 2.0 * displacement_increment / float(delta_t)
    velocity = signs * (
        velocity0 + torch.cumsum(signs * velocity_rhs, dim=1)
    )
    previous_velocity = torch.cat((velocity0, velocity[:, :-1]), dim=1)
    acceleration_rhs = 2.0 * (
        velocity - previous_velocity
    ) / float(delta_t)
    acceleration = signs * (
        acceleration0 + torch.cumsum(signs * acceleration_rhs, dim=1)
    )
    return velocity, acceleration
