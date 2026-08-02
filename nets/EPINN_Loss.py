"""Physics loss for the displacement-increment E-PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class EPINN_MDOFSys_DisIncrement_PhyLoss(nn.Module):
    """Consistency between LSTM and SCL increments at all five DOFs."""

    def forward(self, prediction: dict[str, torch.Tensor]) -> torch.Tensor:
        residual = (
            prediction["dis_increment_nl"]
            - prediction["dis_increment_scl"]
        )
        return torch.mean(residual.pow(2))
