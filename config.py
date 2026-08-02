"""Central configuration for the 5-DOF wave-load PINN/E-PINN example."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class CaseConfig:
    """Training and structural settings shared by PINN and E-PINN."""

    data_root: Path
    random_seed: int = 20260730
    model_sample_count: int = 200
    train_sample_count: int = 170
    labelled_sample_count: int = 40
    time_truncation: int = 600
    sequence_length: int | None = None
    batch_size: int = 10
    num_workers: int = 0
    dtype: str = "float64"
    displacement_increment_scale: float = 1.0e-1
    displacement_scale: float = 5.0e-1
    velocity_scale: float = 5.0e-1
    force_scale: float = 1.0e4

    @property
    def load_file(self) -> Path:
        return self.data_root / "wave_loads_300.mat"

    @property
    def etdm_file(self) -> Path:
        return self.data_root / "wave_responses_etdm_300.mat"

    @property
    def response_file(self) -> Path:
        return self.data_root / "wave_responses_newmark_300.mat"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["data_root"] = str(self.data_root)
        return result
