"""Equation-of-motion residual loss for the conventional PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN_MDOFSys_DisIncrement_PhyLoss(nn.Module):
    """MSE of the force-equilibrium residual with one fixed force scale."""

    def __init__(
        self,
        mass: torch.Tensor,
        damping: torch.Tensor,
        force_scale: float = 1.0e5,
    ) -> None:
        super().__init__()
        if force_scale <= 0.0:
            raise ValueError("force_scale must be positive.")
        self.register_buffer("M", mass)
        self.register_buffer("C", damping)
        self.register_buffer("force_scale", mass.new_tensor(float(force_scale)))

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
        ) / self.force_scale
        return torch.mean(residual.pow(2))
