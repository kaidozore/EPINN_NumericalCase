"""Increment equilibrium loss for the increment-output PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN_MDOFSys_Increment_PhyLoss(nn.Module):
    """MSE of predicted increments plus equilibrium-derived increments."""

    def __init__(
        self,
        mass: torch.Tensor,
        damping: torch.Tensor,
        stiffness: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("M", mass)
        self.register_buffer("C", damping)
        self.register_buffer("K_inv", torch.linalg.inv(stiffness))

    @staticmethod
    def _matrix_product(
        matrix: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("ij,btj->bti", matrix, response)

    def forward(
        self,
        load: torch.Tensor,
        prediction: dict[str, torch.Tensor],
        previous_equilibrium_displacement: torch.Tensor | None = None,
        return_state: bool = False,
        return_metrics: bool = False,
    ):
        load_sequence = load.squeeze(1).transpose(1, 2)
        # From K0*u + M*a + C*v + Rnl - F = 0, define
        # q = inv(K0)*(M*a + C*v + Rnl - F). The exact solution obeys
        # u + q = 0 and therefore delta_u + delta_q = 0.
        force_without_elastic = (
            self._matrix_product(self.M, prediction["acc"])
            + self._matrix_product(self.C, prediction["vel"])
            + prediction["force_nonlinear"]
            - load_sequence
        )
        equilibrium_displacement = self._matrix_product(
            self.K_inv, force_without_elastic
        )
        if previous_equilibrium_displacement is None:
            first = equilibrium_displacement[:, :1]
        else:
            expected = equilibrium_displacement[:, :1].shape
            if previous_equilibrium_displacement.shape != expected:
                raise ValueError(
                    "previous_equilibrium_displacement has an invalid shape."
                )
            first = (
                equilibrium_displacement[:, :1]
                - previous_equilibrium_displacement
            )
        equilibrium_increment = torch.cat(
            [
                first,
                equilibrium_displacement[:, 1:]
                - equilibrium_displacement[:, :-1],
            ],
            dim=1,
        )
        increment_residual = (
            prediction["dis_increment"] + equilibrium_increment
        )
        loss = torch.mean(increment_residual.pow(2))
        state = equilibrium_displacement[:, -1:].detach()
        if not return_metrics:
            if return_state:
                return loss, state
            return loss

        # The exact full displacement is -q. Correlation of the accumulated
        # network displacement with -q is a monitoring metric only; it does
        # not participate in the increment loss or its gradients.
        equilibrium_full_displacement = -equilibrium_displacement
        prediction_centered = prediction["dis"] - torch.mean(
            prediction["dis"], dim=1, keepdim=True
        )
        equilibrium_centered = equilibrium_full_displacement - torch.mean(
            equilibrium_full_displacement, dim=1, keepdim=True
        )
        numerator = torch.sum(
            prediction_centered * equilibrium_centered, dim=1
        )
        prediction_energy = torch.sum(prediction_centered.pow(2), dim=1)
        equilibrium_energy = torch.sum(equilibrium_centered.pow(2), dim=1)
        denominator = torch.sqrt(prediction_energy * equilibrium_energy)
        epsilon = torch.finfo(denominator.dtype).eps
        valid = denominator > epsilon
        correlation_values = (
            numerator / denominator.clamp_min(epsilon)
        ).clamp(-1.0, 1.0)
        if torch.any(valid):
            mean_full_displacement_correlation = torch.mean(
                correlation_values[valid]
            )
        else:
            mean_full_displacement_correlation = loss.new_zeros(())
        metrics = {
            "full_displacement_correlation": (
                mean_full_displacement_correlation.detach()
            )
        }
        if return_state:
            return loss, state, metrics
        return loss, metrics
