"""Physics loss for the full-displacement E-PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class EPINN_MDOFSys_FullDis_PhyLoss(nn.Module):
    """MSE between LSTM and fixed-SCL total displacements."""

    def forward(self, prediction: dict[str, torch.Tensor]) -> torch.Tensor:
        residual = prediction["dis_nl"] - prediction["dis"]
        return torch.mean(residual.pow(2))
