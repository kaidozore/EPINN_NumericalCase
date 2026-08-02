"""Compact loss logger compatible with long numerical experiments."""

from __future__ import annotations

import csv
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Mapping

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Training and CSV logging can still continue.
    plt = None


_CHECKPOINT_PATTERN = re.compile(
    r"^ep(?P<epoch>\d+)-train(?P<train>[-+0-9.eE]+)-"
    r"val(?P<val>[-+0-9.eE]+)\.pth$"
)


def save_top_k_checkpoint(
    checkpoint_dir: str | Path,
    checkpoint: dict,
    epoch: int,
    train_loss: float,
    val_loss: float,
    max_to_keep: int = 10,
) -> Path:
    """Save a checkpoint and retain the ten smallest validation losses."""

    if max_to_keep < 1:
        raise ValueError("max_to_keep must be positive.")
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"ep{epoch:04d}-train{train_loss:.6e}-val{val_loss:.6e}.pth"
    )
    output = directory / filename
    temporary = output.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)

    ranked: list[tuple[float, float, int, Path]] = []
    for candidate in directory.glob("*.pth"):
        match = _CHECKPOINT_PATTERN.match(candidate.name)
        if match is None:
            # Files from an older naming convention are not silently deleted.
            continue
        candidate_val = float(match.group("val"))
        candidate_train = float(match.group("train"))
        if not math.isfinite(candidate_val):
            candidate_val = math.inf
        if not math.isfinite(candidate_train):
            candidate_train = math.inf
        ranked.append(
            (
                candidate_val,
                candidate_train,
                int(match.group("epoch")),
                candidate,
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    removed = []
    for _, _, _, candidate in ranked[max_to_keep:]:
        candidate.unlink()
        removed.append(candidate.name)
    if removed:
        print(
            f"Checkpoint retention: removed {len(removed)} file(s); "
            f"kept the {max_to_keep} smallest val_loss values."
        )
    return output


class LossHistory:
    def __init__(self, log_dir: str | Path) -> None:
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.save_path = Path(log_dir) / ("loss_" + stamp)
        self.save_path.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.save_path / "epoch_loss.csv"
        self.plot_path = self.save_path / "epoch_loss.png"
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self._plot_warning_printed = False

    def append_loss(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        details: Mapping[str, float] | None = None,
    ) -> None:
        details = {} if details is None else dict(details)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **details,
        }
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.epochs.append(int(epoch))
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        self._save_loss_plot()

    def _save_loss_plot(self) -> None:
        if plt is None:
            if not self._plot_warning_printed:
                print(
                    "Warning: matplotlib is unavailable; epoch_loss.csv is "
                    "saved but epoch_loss.png cannot be generated."
                )
                self._plot_warning_printed = True
            return
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        axis.plot(
            self.epochs, self.train_losses, label="Train loss", linewidth=1.4
        )
        axis.plot(
            self.epochs, self.val_losses, label="Validation loss", linewidth=1.4
        )
        positive = all(
            value > 0.0 and math.isfinite(value)
            for value in self.train_losses + self.val_losses
        )
        if positive:
            axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("Training and validation loss")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        temporary = self.plot_path.with_suffix(".tmp.png")
        figure.savefig(temporary, dpi=160, format="png")
        plt.close(figure)
        temporary.replace(self.plot_path)
