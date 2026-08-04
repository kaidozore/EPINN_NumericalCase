"""Full-displacement consistency loss for the increment-output E-PINN."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EPINN_MDOFSys_DisIncrement_PhyLoss(nn.Module):
    """Train accumulated increments against the fixed-SCL displacement."""

    reports_metrics = True

    def __init__(
        self,
        displacement_scale: float | torch.Tensor = 1.0,
        trend_window_size: int = 26,
        trend_weight: float = 0.05,
    ) -> None:
        super().__init__()
        scale = torch.as_tensor(displacement_scale).reshape(-1)
        if scale.numel() < 1 or torch.any(scale <= 0.0):
            raise ValueError("displacement_scale must contain positive values.")
        self.register_buffer("displacement_scale", scale.reshape(1, 1, -1))
        self.trend_window_size = int(trend_window_size)
        self.trend_weight = float(trend_weight)
        if self.trend_window_size < 2:
            raise ValueError("trend_window_size must be at least two.")
        if self.trend_weight < 0.0:
            raise ValueError("trend_weight must be non-negative.")

    @staticmethod
    def _stable_log_cosh(error: torch.Tensor) -> torch.Tensor:
        absolute = torch.abs(error)
        return absolute + F.softplus(-2.0 * absolute) - math.log(2.0)

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

    def _scale_for(self, displacement: torch.Tensor) -> torch.Tensor:
        scale = self.displacement_scale.to(
            dtype=displacement.dtype, device=displacement.device
        )
        if scale.shape[-1] == 1:
            return scale
        if scale.shape[-1] != displacement.shape[-1]:
            raise ValueError(
                "displacement_scale must be scalar or match the DOF count."
            )
        return scale

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor] | None = None,
        return_metrics: bool = False,
    ):
        predicted_displacement = prediction["dis_nl"]
        scl_displacement = prediction["dis"]
        scale = self._scale_for(predicted_displacement)
        predicted_normalized = predicted_displacement / scale
        scl_normalized = scl_displacement / scale

        full_error = predicted_normalized - scl_normalized
        full_log_cosh = torch.mean(self._stable_log_cosh(full_error))

        sequence_length = predicted_displacement.shape[1]
        window_count = sequence_length // self.trend_window_size
        if window_count > 0 and self.trend_weight > 0.0:
            valid_length = window_count * self.trend_window_size
            shape = (
                predicted_displacement.shape[0],
                window_count,
                self.trend_window_size,
                predicted_displacement.shape[2],
            )
            predicted_windows = predicted_normalized[:, :valid_length].reshape(
                shape
            )
            scl_windows = scl_normalized[:, :valid_length].reshape(shape)
            predicted_trend = (
                predicted_windows - predicted_windows[:, :, :1]
            )
            scl_trend = scl_windows - scl_windows[:, :, :1]
            trend_log_cosh = torch.mean(
                self._stable_log_cosh(predicted_trend - scl_trend)
            )
        else:
            trend_log_cosh = full_log_cosh.new_zeros(())

        weighted_trend = self.trend_weight * trend_log_cosh
        total_loss = full_log_cosh + weighted_trend
        if not return_metrics:
            return total_loss

        with torch.no_grad():
            scl_error = predicted_displacement - scl_displacement
            scl_rmse = torch.sqrt(torch.mean(scl_error.pow(2)))
            scl_correlation = self._mean_time_correlation(
                predicted_displacement, scl_displacement
            )
            metrics = {
                "full_log_cosh": full_log_cosh.detach(),
                "weighted_trend_log_cosh": weighted_trend.detach(),
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
