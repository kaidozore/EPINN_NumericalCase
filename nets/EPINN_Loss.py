"""Increment-primary, local-cumulative MSE loss for increment E-PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class EPINN_MDOFSys_DisIncrement_PhyLoss(nn.Module):
    """Use increment consistency plus reset local cumulative trajectories."""

    reports_metrics = True

    def __init__(
        self,
        increment_scale: float | torch.Tensor = 1.0,
        displacement_scale: float | torch.Tensor = 1.0,
        increment_loss_weight: float = 1.0,
        local_cumsum_loss_weight: float = 0.05,
        local_cumsum_window: int = 32,
        label_increment_loss_weight: float = 0.2,
        label_local_cumsum_loss_weight: float = 0.01,
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
        self.local_cumsum_loss_weight = float(local_cumsum_loss_weight)
        if self.local_cumsum_loss_weight < 0.0:
            raise ValueError(
                "local_cumsum_loss_weight must be non-negative."
            )
        self.local_cumsum_window = int(local_cumsum_window)
        if not 20 <= self.local_cumsum_window <= 50:
            raise ValueError(
                "local_cumsum_window must be between 20 and 50 steps."
            )
        self.label_increment_loss_weight = float(
            label_increment_loss_weight
        )
        self.label_local_cumsum_loss_weight = float(
            label_local_cumsum_loss_weight
        )
        if self.label_increment_loss_weight < 0.0:
            raise ValueError(
                "label_increment_loss_weight must be non-negative."
            )
        if self.label_local_cumsum_loss_weight < 0.0:
            raise ValueError(
                "label_local_cumsum_loss_weight must be non-negative."
            )

    @staticmethod
    def _mse(error: torch.Tensor) -> torch.Tensor:
        return torch.mean(error.pow(2))

    def _local_cumulative_mse(
        self,
        increment_error: torch.Tensor,
        displacement_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate only inside reset, non-overlapping short windows."""
        window = self.local_cumsum_window
        valid_length = (increment_error.shape[1] // window) * window
        if valid_length == 0:
            local_error = torch.cumsum(increment_error, dim=1)
        else:
            trimmed = increment_error[:, :valid_length]
            batch, _, dof = trimmed.shape
            blocks = trimmed.reshape(batch, -1, window, dof)
            local_error = torch.cumsum(blocks, dim=2)
        return self._mse(local_error / displacement_scale)

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
        # Treat the SCL response as a fixed Picard target during each
        # optimizer update.  Its forward value still contains the current
        # Steel02/SCL physics, but gradients do not co-adapt the target branch.
        scl_increment = prediction["dis_increment_scl"].detach()
        predicted_displacement = prediction["dis_nl"]
        scl_displacement = prediction["dis"].detach()

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

        # Reset cumsum at every short window.  This constrains local response
        # trajectories without backpropagating through a full response history.
        local_cumsum_mse = self._local_cumulative_mse(
            predicted_increment - scl_increment,
            displacement_scale,
        )
        weighted_increment = self.increment_loss_weight * increment_mse
        weighted_local_cumsum = (
            self.local_cumsum_loss_weight * local_cumsum_mse
        )
        label_increment_mse = predicted_increment.new_zeros(())
        label_local_cumsum_mse = predicted_increment.new_zeros(())
        labelled_fraction = predicted_increment.new_zeros(())
        if (
            target is not None
            and "labelled" in target
            and "dis_increment" in target
            and "dis" in target
        ):
            labelled = target["labelled"].to(
                device=predicted_increment.device, dtype=torch.bool
            ).reshape(-1)
            labelled_fraction = labelled.to(
                predicted_increment.dtype
            ).mean()
            if torch.any(labelled):
                true_increment = target["dis_increment"].to(
                    dtype=predicted_increment.dtype,
                    device=predicted_increment.device,
                )
                true_displacement = target["dis"].to(
                    dtype=predicted_displacement.dtype,
                    device=predicted_displacement.device,
                )
                label_increment_error = (
                    predicted_increment[labelled]
                    - true_increment[labelled]
                ) / increment_scale
                label_increment_mse = self._mse(label_increment_error)
                label_local_cumsum_mse = self._local_cumulative_mse(
                    predicted_increment[labelled]
                    - true_increment[labelled],
                    displacement_scale,
                )

        weighted_label_increment = (
            self.label_increment_loss_weight * label_increment_mse
        )
        weighted_label_local_cumsum = (
            self.label_local_cumsum_loss_weight
            * label_local_cumsum_mse
        )
        total_loss = (
            weighted_increment
            + weighted_local_cumsum
            + weighted_label_increment
            + weighted_label_local_cumsum
        )
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
                "local_cumsum_mse": local_cumsum_mse.detach(),
                "weighted_increment_mse": weighted_increment.detach(),
                "weighted_local_cumsum_mse": (
                    weighted_local_cumsum.detach()
                ),
                "label_increment_mse": label_increment_mse.detach(),
                "label_local_cumsum_mse": (
                    label_local_cumsum_mse.detach()
                ),
                "weighted_label_increment_mse": (
                    weighted_label_increment.detach()
                ),
                "weighted_label_local_cumsum_mse": (
                    weighted_label_local_cumsum.detach()
                ),
                "labelled_fraction": labelled_fraction.detach(),
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
