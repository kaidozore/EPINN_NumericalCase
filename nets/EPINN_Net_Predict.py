"""Checkpoint-backed E-PINN prediction helper."""

from __future__ import annotations

import torch


class EPINN_Net_predict:
    def __init__(self, model: torch.nn.Module) -> None:
        self.net = model.eval()

    @torch.no_grad()
    def predict(self, load: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(load)

