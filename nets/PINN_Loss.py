"""Label and equation-of-motion losses for the conventional PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN_MDOFSys_DisIncrement_LabPhyLoss(nn.Module):
    """Displacement/increment data loss plus physics on every sample."""

    def __init__(
        self,
        mass: torch.Tensor,
        damping: torch.Tensor,
        stiffness: torch.Tensor,
        displacement_scale=2.0e-3,
        force_scale=1.0e2,
        increment_scale=1.0e-4,
        increment_weight: float = 1.0,
        displacement_weight: float = 1.0,
        physics_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("M", mass)
        self.register_buffer("C", damping)
        self.register_buffer("K", stiffness)
        dtype = mass.dtype
        device = mass.device
        def scale_tensor(value):
            result = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
            if result.numel() == 1:
                result = result.expand(mass.shape[0]).clone()
            return result.reshape(1, 1, -1)
        self.register_buffer("displacement_scale", scale_tensor(displacement_scale))
        self.register_buffer("force_scale", scale_tensor(force_scale))
        self.register_buffer("increment_scale", scale_tensor(increment_scale))
        self.increment_weight = float(increment_weight)
        self.displacement_weight = float(displacement_weight)
        self.physics_weight = float(physics_weight)
        self.current_physics_weight = float(physics_weight)

    def set_physics_fraction(self, fraction: float) -> None:
        self.current_physics_weight = self.physics_weight * float(fraction)

    @staticmethod
    def _matrix_product(
        matrix: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("ij,btj->bti", matrix, response)

    def forward(
        self,
        load: torch.Tensor,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        compute_physics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if compute_physics:
            load_sequence = load.squeeze(1).transpose(1, 2)
            residual = (
                self._matrix_product(self.M, prediction["acc"])
                + self._matrix_product(self.C, prediction["vel"])
                + self._matrix_product(self.K, prediction["dis"])
                + prediction["force_nonlinear"]
                - load_sequence
            ) / self.force_scale
            physics_loss = torch.mean(residual.pow(2))
        else:
            physics_loss = prediction["dis"].new_zeros(())

        labelled = target["labelled"].bool()
        if torch.any(labelled):
            increment_error = (
                prediction["dis_increment"][labelled]
                - target["dis_increment"][labelled]
            ) / self.increment_scale
            increment_loss = torch.mean(increment_error.pow(2))
            displacement_error_physical = (
                prediction["dis"][labelled] - target["dis"][labelled]
            )
            dis_loss = torch.mean(
                (displacement_error_physical / self.displacement_scale).pow(2)
            )
            increment_mse_physical = torch.mean(
                (
                    prediction["dis_increment"][labelled]
                    - target["dis_increment"][labelled]
                ).pow(2)
            )
            displacement_mse_physical = torch.mean(
                displacement_error_physical.pow(2)
            )
            data_loss = (
                self.increment_weight * increment_loss
                + self.displacement_weight * dis_loss
            )
        else:
            increment_loss = physics_loss.new_zeros(())
            dis_loss = physics_loss.new_zeros(())
            increment_mse_physical = physics_loss.new_zeros(())
            displacement_mse_physical = physics_loss.new_zeros(())
            data_loss = physics_loss.new_zeros(())
        total = data_loss + self.current_physics_weight * physics_loss
        return total, {
            "data": data_loss.detach(),
            "physics": physics_loss.detach(),
            "increment": increment_loss.detach(),
            "displacement": dis_loss.detach(),
            "increment_mse_physical": increment_mse_physical.detach(),
            "displacement_mse_physical": displacement_mse_physical.detach(),
            "physics_fraction": physics_loss.new_tensor(
                self.current_physics_weight
            ),
        }
