"""Equation-of-motion residual loss for the conventional PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN_MDOFSys_DisIncrement_PhyLoss(nn.Module):
    """MSE of the mass-normalized equation-of-motion residual."""

    def __init__(
        self,
        mass: torch.Tensor,
        damping: torch.Tensor,
        stiffness: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("M_inv", torch.linalg.inv(mass))
        self.register_buffer("C", damping)
        self.register_buffer("K", stiffness)

    @staticmethod
    def _matrix_product(
        matrix: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("ij,btj->bti", matrix, response)

    def forward(
        self,
        load: torch.Tensor,
        prediction: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        load_sequence = load.squeeze(1).transpose(1, 2)
        force_residual = (
            self._matrix_product(self.C, prediction["vel"])
            + self._matrix_product(self.K, prediction["dis"])
            + prediction["force_nonlinear"]
            - load_sequence
        )
        acceleration_residual = prediction["acc"] + self._matrix_product(
            self.M_inv, force_residual
        )
        return torch.mean(acceleration_residual.pow(2))
