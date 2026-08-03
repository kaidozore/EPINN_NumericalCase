"""Label-assisted equation-of-motion loss for the conventional PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN_MDOFSys_DisIncrement_PhyLoss(nn.Module):
    """Physics MSE on all samples plus displacement MSE on labelled samples."""

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
        equilibrium_residual = (
            self._matrix_product(self.M, prediction["acc"])
            + self._matrix_product(self.C, prediction["vel"])
            + self._matrix_product(self.K, prediction["dis"])
            + prediction["force_nonlinear"]
            - load_sequence
        )
        displacement_residual = self._matrix_product(
            self.K_inv, equilibrium_residual
        )
        physics_loss = torch.mean(displacement_residual.pow(2))
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
            "label": label_loss.detach(),
            "labelled_count": labelled_count,
        }
