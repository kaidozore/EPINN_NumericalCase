"""Read the MATLAB v7.3 model, load, response and ETDM data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import h5py
import numpy as np
import torch

from config import CaseConfig


@dataclass
class CaseData:
    """Arrays use the Python layout ``[sample,time,feature]``."""

    load: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    internal_force: np.ndarray
    nonlinear_force: np.ndarray
    etdm_displacement: np.ndarray
    etdm_velocity: np.ndarray
    time: np.ndarray
    mass: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray
    influence_kernel: np.ndarray
    fiber: Dict[str, np.ndarray | float]
    steel: Dict[str, float]

    @property
    def delta_t(self) -> float:
        return float(self.time[1] - self.time[0])


@dataclass
class DataSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    labelled: np.ndarray


def _scalar(group: h5py.Group, name: str) -> float:
    return float(np.asarray(group[name]).reshape(-1)[0])


def _series(dataset: h5py.Dataset, length: int) -> np.ndarray:
    """MATLAB v7.3 reverses a [DOF,time,sample] array for h5py."""

    return np.asarray(dataset[:, :length, :], dtype=np.float64)


def load_case_data(config: CaseConfig) -> CaseData:
    """Load all structural and excitation quantities from MATLAB files."""

    for path in (config.load_file, config.response_file, config.etdm_file):
        if not path.exists():
            raise FileNotFoundError(path)

    with h5py.File(config.load_file, "r") as mat:
        available_length = int(mat["Fwave"].shape[1])
        length = (
            available_length
            if config.sequence_length is None
            else min(int(config.sequence_length), available_length)
        )
        if length < 3:
            raise ValueError("At least three time steps are required.")
        load = _series(mat["Fwave"], length)
        time = np.asarray(mat["time"], dtype=np.float64).reshape(-1)[:length]
        mass = np.asarray(mat["M"], dtype=np.float64)
        damping = np.asarray(mat["C"], dtype=np.float64)
        stiffness = np.asarray(mat["K0"], dtype=np.float64)

        structure = mat["cfg"]["structure"]
        fiber_group = mat["cfg"]["fiber"]
        steel_group = mat["cfg"]["steel"]
        fiber: Dict[str, np.ndarray | float] = {
            # h5py exposes MATLAB Tcond(10,5) as (5,10).
            "transformation": np.asarray(
                structure["transformation"], dtype=np.float64
            ).T,
            # MATLAB elementDOF(5,4) is exposed as (4,5).
            "element_dof": np.asarray(
                structure["elementDOF"], dtype=np.int64
            ).T,
            "element_length": _scalar(structure, "h"),
            "fiber_y": np.asarray(
                fiber_group["y"], dtype=np.float64
            ).reshape(-1),
            "fiber_area": np.asarray(
                fiber_group["area"], dtype=np.float64
            ).reshape(-1),
            "gauss_xi": np.asarray(
                fiber_group["gaussXi"], dtype=np.float64
            ).reshape(-1),
            "gauss_weight": np.asarray(
                fiber_group["gaussWeight"], dtype=np.float64
            ).reshape(-1),
        }
        steel = {
            name: _scalar(steel_group, name)
            for name in (
                "Fy", "E0", "b", "R0", "cR1", "cR2",
                "a1", "a2", "a3", "a4",
            )
        }

    with h5py.File(config.response_file, "r") as mat:
        displacement = _series(mat["U"], length)
        velocity = _series(mat["V"], length)
        acceleration = _series(mat["Acc"], length)
        internal_force = _series(mat["FintFull"], length)
        for name, reference in (("M", mass), ("C", damping), ("K0", stiffness)):
            if not np.allclose(np.asarray(mat[name]), reference):
                raise ValueError(f"{name} differs between MATLAB files.")

    with h5py.File(config.etdm_file, "r") as mat:
        nonlinear_force = _series(mat["Fnonlinear"], length)
        etdm_displacement = _series(mat["U"], length)
        etdm_velocity = _series(mat["V"], length)
        kernel_length = min(
            int(config.time_truncation), length, int(mat["ETDM"]["A"].shape[0])
        )
        influence_kernel = np.asarray(
            mat["ETDM"]["A"][:kernel_length], dtype=np.float64
        )

    n_sample, n_time, n_dof = load.shape
    expected = (n_sample, n_time, n_dof)
    for name, value in (
        ("U", displacement), ("V", velocity), ("Acc", acceleration),
        ("FintFull", internal_force), ("Fnonlinear", nonlinear_force),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} has shape {value.shape}, expected {expected}.")
    if mass.shape != (n_dof, n_dof):
        raise ValueError("The externally loaded M matrix is inconsistent.")
    if not np.all(load[:, 0, :] == 0.0):
        raise ValueError(
            "Fwave at the initial time must be exactly zero for direct Aij/SCL."
        )
    if not np.all(nonlinear_force[:, 0, :] == 0.0):
        raise ValueError(
            "Fnonlinear at the initial time must be exactly zero."
        )
    if influence_kernel.shape[1:] != (2 * n_dof, 2 * n_dof):
        raise ValueError(
            "ETDM/A must contain five external and five nonlinear force inputs."
        )

    return CaseData(
        load=load,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
        internal_force=internal_force,
        nonlinear_force=nonlinear_force,
        etdm_displacement=etdm_displacement,
        etdm_velocity=etdm_velocity,
        time=time,
        mass=mass,
        damping=damping,
        stiffness=stiffness,
        influence_kernel=influence_kernel,
        fiber=fiber,
        steel=steel,
    )


def build_data_split(config: CaseConfig, n_sample: int) -> DataSplit:
    """Use the paper split for 300 samples and a safe split for smoke tests."""

    if n_sample < 2:
        raise ValueError("At least two samples are required for train/validation.")
    if n_sample >= config.model_sample_count:
        model_count = config.model_sample_count
        train_count = config.train_sample_count
        test = np.arange(model_count, n_sample, dtype=np.int64)
    else:
        # Keep one independent test sample so a complete train/predict smoke
        # test is possible before the 300-sample production data are ready.
        model_count = n_sample - 1
        train_count = max(1, min(model_count - 1, round(0.85 * model_count)))
        test = np.arange(model_count, n_sample, dtype=np.int64)
    generator = np.random.default_rng(config.random_seed)
    model_pool = generator.permutation(model_count)
    train = np.sort(model_pool[:train_count])
    validation = np.sort(model_pool[train_count:])
    labelled_count = min(config.labelled_sample_count, train.size)
    labelled = np.sort(
        generator.choice(train, size=labelled_count, replace=False)
    )
    return DataSplit(train, validation, test, labelled)


def load_scale_from_training(
    load: np.ndarray, train_indices: np.ndarray
) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(load[train_indices]), axis=(0, 1)))
    return np.where(scale > 1.0e-12, scale, 1.0)


def as_torch_case(data: CaseData, device: torch.device) -> dict:
    """Convert only MATLAB-loaded constants; no M/C/K/load is rebuilt."""

    tensor = lambda value, dtype=torch.float64: torch.as_tensor(
        value, dtype=dtype, device=device
    )
    return {
        "kernel": tensor(data.influence_kernel),
        "mass": tensor(data.mass),
        "damping": tensor(data.damping),
        "stiffness": tensor(data.stiffness),
        "fiber": {
            name: tensor(value, torch.int64 if name == "element_dof" else torch.float64)
            for name, value in data.fiber.items()
        },
        "steel": data.steel,
    }
