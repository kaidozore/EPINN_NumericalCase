"""Equation-of-motion residual loss for the conventional PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN_MDOFSys_DisIncrement_PhyLoss(nn.Module):
    """MSE of ``M*a + C*v + fint - Fwave`` in physical force units."""

    def __init__(self, mass: torch.Tensor, damping: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("M", mass)
        self.register_buffer("C", damping)

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
        residual = (
            self._matrix_product(self.M, prediction["acc"])
            + self._matrix_product(self.C, prediction["vel"])
            + prediction["force_internal"]
            - load_sequence
        )
        return torch.mean(residual.pow(2))
