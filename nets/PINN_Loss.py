"""Label-assisted equation-of-motion loss for the conventional PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN_MDOFSys_DisIncrement_PhyLoss(nn.Module):
    """Correlation-weighted physics RMSE plus labelled displacement MSE."""

    def __init__(
        self,
        mass: torch.Tensor,
        damping: torch.Tensor,
        stiffness: torch.Tensor,
        label_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("M", mass)
        self.register_buffer("C", damping)
        self.register_buffer("K", stiffness)
        self.register_buffer("K_inv", torch.linalg.inv(stiffness))
        self.label_weight = float(label_weight)
        if self.label_weight < 0.0:
            raise ValueError("label_weight must be non-negative.")

    @staticmethod
    def _matrix_product(
        matrix: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("ij,btj->bti", matrix, response)

    def forward(
        self,
        load: torch.Tensor,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor] | None = None,
    ):
        load_sequence = load.squeeze(1).transpose(1, 2)
        response_without_elastic_force = (
            self._matrix_product(self.M, prediction["acc"])
            + self._matrix_product(self.C, prediction["vel"])
            + prediction["force_nonlinear"]
            - load_sequence
        )
        equilibrium_displacement = -self._matrix_product(
            self.K_inv, response_without_elastic_force
        )
        displacement_residual = (
            prediction["dis"] - equilibrium_displacement
        )
        equilibrium_rmse = torch.sqrt(
            torch.mean(displacement_residual.pow(2))
            + torch.finfo(displacement_residual.dtype).eps
        )

        # Pearson correlation is evaluated independently for every sample and
        # DOF along the time axis.  The 1.1 offset retains 0.1*RMSE even when
        # two histories have correlation one but different amplitudes.
        prediction_centered = prediction["dis"] - torch.mean(
            prediction["dis"], dim=1, keepdim=True
        )
        equilibrium_centered = equilibrium_displacement - torch.mean(
            equilibrium_displacement, dim=1, keepdim=True
        )
        numerator = torch.sum(
            prediction_centered * equilibrium_centered, dim=1
        )
        prediction_energy = torch.sum(prediction_centered.pow(2), dim=1)
        equilibrium_energy = torch.sum(equilibrium_centered.pow(2), dim=1)
        denominator = torch.sqrt(prediction_energy * equilibrium_energy)
        valid = denominator > torch.finfo(denominator.dtype).eps
        correlation_values = numerator / denominator.clamp_min(
            torch.finfo(denominator.dtype).eps
        )
        correlation_values = correlation_values.clamp(-1.0, 1.0)
        if torch.any(valid):
            mean_correlation = torch.mean(correlation_values[valid])
        else:
            mean_correlation = equilibrium_rmse.new_zeros(())
        physics_loss = equilibrium_rmse * (1.1 - mean_correlation)
        if target is None:
            return physics_loss

        labelled = target["labelled"].bool().reshape(-1)
        if torch.any(labelled):
            label_loss = torch.mean(
                (
                    prediction["dis"][labelled]
                    - target["dis"][labelled]
                ).pow(2)
            )
            labelled_count = int(torch.count_nonzero(labelled).item())
        else:
            label_loss = physics_loss.new_zeros(())
            labelled_count = 0
        total_loss = physics_loss + self.label_weight * label_loss
        return total_loss, {
            "physics": physics_loss.detach(),
            "equilibrium_rmse": equilibrium_rmse.detach(),
            "correlation": mean_correlation.detach(),
            "label": label_loss.detach(),
            "labelled_count": labelled_count,
        }
