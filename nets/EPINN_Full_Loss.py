"""Physics loss and common displacement metrics for full E-PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class EPINN_MDOFSys_FullDis_PhyLoss(nn.Module):
    """MSE between LSTM and SCL displacements with common monitoring."""

    reports_metrics = True

    def __init__(self, continuity_loss_weight: float = 1.0, label_weight: float = 0.1) -> None:
        super().__init__()
        self.label_weight = float(label_weight)
        if self.label_weight < 0.0:
            raise ValueError("label_weight must be non-negative.")
        self.continuity_loss_weight = float(continuity_loss_weight)
        if self.continuity_loss_weight < 0.0:
            raise ValueError("continuity_loss_weight must be non-negative.")

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

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor] | None = None,
        return_metrics: bool = False,
    ):
        predicted_displacement = prediction["dis_nl"]
        # Fixed-target (Picard) training: retain the nonlinear SCL forward
        # response while preventing the target branch from co-adapting in the
        # same backward pass as the LSTM prediction.
        scl_displacement = prediction["dis"].detach()
        scl_error = predicted_displacement - scl_displacement
        full_mse = torch.mean(scl_error.pow(2))
        continuity_mse = predicted_displacement.new_zeros(())
        if "boundary_prediction" in prediction:
            continuity_mse = torch.mean(
                (
                    prediction["boundary_prediction"]
                    - prediction["boundary_initial"].detach()
                ).pow(2)
            )
        weighted_continuity = (
            self.continuity_loss_weight * continuity_mse
        )
        label_mse = predicted_displacement.new_zeros(())
        if target is not None and "labelled" in target:
            labelled = target["labelled"].to(device=predicted_displacement.device, dtype=torch.bool).reshape(-1)
            if torch.any(labelled):
                label_mse = (predicted_displacement[labelled] - target["dis"][labelled]).square().mean()
        loss = full_mse + weighted_continuity + self.label_weight * label_mse
        if not return_metrics:
            return loss

        with torch.no_grad():
            metrics = {
                "full_mse": full_mse.detach(),
                "label_mse": label_mse.detach(),
                "weighted_label_mse": (self.label_weight * label_mse).detach(),
                "continuity_mse": continuity_mse.detach(),
                "weighted_continuity_mse": weighted_continuity.detach(),
                "scl_displacement_rmse_m": torch.sqrt(
                    torch.mean(scl_error.pow(2))
                ),
                "scl_displacement_correlation": (
                    self._mean_time_correlation(
                        predicted_displacement, scl_displacement
                    )
                ),
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
        return loss, metrics
