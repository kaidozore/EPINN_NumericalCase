"""Full-response-first MSE loss for the increment-output E-PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class EPINN_MDOFSys_DisIncrement_PhyLoss(nn.Module):
    """Match full LSTM/SCL responses, with increment consistency auxiliary."""

    reports_metrics = True

    def __init__(
        self,
        increment_scale: float | torch.Tensor = 1.0,
        displacement_scale: float | torch.Tensor = 1.0,
        increment_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        increment_scale = torch.as_tensor(increment_scale).reshape(-1)
        displacement_scale = torch.as_tensor(displacement_scale).reshape(-1)
        if increment_scale.numel() < 1 or torch.any(increment_scale <= 0.0):
            raise ValueError("increment_scale must contain positive values.")
        if (
            displacement_scale.numel() < 1
            or torch.any(displacement_scale <= 0.0)
        ):
            raise ValueError(
                "displacement_scale must contain positive values."
            )
        self.register_buffer(
            "increment_scale", increment_scale.reshape(1, 1, -1)
        )
        self.register_buffer(
            "displacement_scale", displacement_scale.reshape(1, 1, -1)
        )
        self.increment_loss_weight = float(increment_loss_weight)
        if self.increment_loss_weight < 0.0:
            raise ValueError("increment_loss_weight must be non-negative.")

    @staticmethod
    def _mse(error: torch.Tensor) -> torch.Tensor:
        return torch.mean(error.pow(2))

    @staticmethod
    def _mean_time_correlation(
        prediction: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        prediction_centered = prediction - torch.mean(
            prediction, dim=1, keepdim=True
        )
        reference_centered = reference - torch.mean(
            reference, dim=1, keepdim=True
        )
        numerator = torch.sum(
            prediction_centered * reference_centered, dim=1
        )
        prediction_energy = torch.sum(prediction_centered.pow(2), dim=1)
        reference_energy = torch.sum(reference_centered.pow(2), dim=1)
        denominator = torch.sqrt(prediction_energy * reference_energy)
        epsilon = torch.finfo(denominator.dtype).eps
        valid = denominator > epsilon
        correlation = (
            numerator / denominator.clamp_min(epsilon)
        ).clamp(-1.0, 1.0)
        if torch.any(valid):
            return torch.mean(correlation[valid])
        return prediction.new_zeros(())

    @staticmethod
    def _scale_for(
        stored_scale: torch.Tensor,
        response: torch.Tensor,
    ) -> torch.Tensor:
        scale = stored_scale.to(
            dtype=response.dtype, device=response.device
        )
        if scale.shape[-1] == 1:
            return scale
        if scale.shape[-1] != response.shape[-1]:
            raise ValueError(
                "A physical scale must be scalar or match the DOF count."
            )
        return scale

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor] | None = None,
        return_metrics: bool = False,
    ):
        predicted_increment = prediction["dis_increment_nl"]
        scl_increment = prediction["dis_increment_scl"]
        predicted_displacement = prediction["dis_nl"]
        scl_displacement = prediction["dis"]

        increment_scale = self._scale_for(
            self.increment_scale, predicted_increment
        )
        displacement_scale = self._scale_for(
            self.displacement_scale, predicted_displacement
        )

        # Keep the first point because, after the first TBPTT chunk, it is the
        # physical increment across a chunk boundary.  Only the single global
        # initial point is trivially zero, which has negligible mean weight.
        increment_error = (
            predicted_increment - scl_increment
        ) / increment_scale
        increment_mse = self._mse(increment_error)

        full_error = (
            predicted_displacement - scl_displacement
        ) / displacement_scale
        full_mse = self._mse(full_error)
        # The accumulated displacement is the primary fixed-point quantity.
        # With TBPTT the gradient is deliberately detached at chunk boundaries,
        # so the auxiliary increment term supplies local, well-conditioned
        # training information without replacing the global response target.
        weighted_increment = self.increment_loss_weight * increment_mse
        total_loss = full_mse + weighted_increment
        if not return_metrics:
            return total_loss

        with torch.no_grad():
            scl_error = predicted_displacement - scl_displacement
            scl_increment_error = predicted_increment - scl_increment
            scl_rmse = torch.sqrt(torch.mean(scl_error.pow(2)))
            scl_increment_rmse = torch.sqrt(
                torch.mean(scl_increment_error.pow(2))
            )
            scl_correlation = self._mean_time_correlation(
                predicted_displacement, scl_displacement
            )
            metrics = {
                "increment_mse": increment_mse.detach(),
                "full_mse": full_mse.detach(),
                "weighted_increment_mse": weighted_increment.detach(),
                "scl_increment_rmse_m": scl_increment_rmse,
                "scl_displacement_rmse_m": scl_rmse,
                "scl_displacement_correlation": scl_correlation,
            }
            if target is not None:
                true_displacement = target["dis"]
                true_error = predicted_displacement - true_displacement
                true_rmse = torch.sqrt(torch.mean(true_error.pow(2)))
                true_rms = torch.sqrt(torch.mean(true_displacement.pow(2)))
                metrics.update(
                    {
                        "true_displacement_rmse_m": true_rmse,
                        "true_displacement_rrmse_percent": (
                            100.0
                            * true_rmse
                            / true_rms.clamp_min(
                                torch.finfo(true_rms.dtype).eps
                            )
                        ),
                        "true_displacement_correlation": (
                            self._mean_time_correlation(
                                predicted_displacement, true_displacement
                            )
                        ),
                    }
                )
        return total_loss, metrics
