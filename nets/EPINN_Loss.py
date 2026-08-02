"""Physics loss for the displacement-increment E-PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class EPINN_MDOFSys_DisIncrement_PhyLoss(nn.Module):
    """Consistency between LSTM and SCL increments at all five DOFs."""

    def __init__(self, increment_scale: float = 1.0e-4) -> None:
        super().__init__()
        self.increment_scale = float(increment_scale)

    def forward(self, prediction: dict[str, torch.Tensor]) -> torch.Tensor:
        residual = (
            prediction["dis_increment_nl"]
            - prediction["dis_increment_scl"]
        ) / self.increment_scale
        return torch.mean(residual[:, 1:, :].pow(2))
