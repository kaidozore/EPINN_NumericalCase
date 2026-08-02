"""Train the conventional physics-informed LSTM baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import CaseConfig
from nets.PINN_Loss import PINN_MDOFSys_DisIncrement_LabPhyLoss
from nets.PINN_Net import PINN_PhyLSTM3_DisIncrement_NetBody
from utils.callbacks import LossHistory
from utils.Dataloader import DynAnaDataset, DynAna_dataset_collate
from utils.DataPreProcess import (
    as_torch_case,
    build_data_split,
    load_case_data,
    load_scale_from_training,
)
from utils.utils import seed_everything
from utils.utils_fit_PINN import (
    fitOneEpoch_PINN_DisIncrement_LabPhyLoss,
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=240)
    parser.add_argument("--fc-size", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument(
        "--tbptt-length", type=int, default=500,
        help="Consecutive steps per truncated-backpropagation chunk.",
    )
    parser.add_argument(
        "--physics-weight", type=float, default=1.0,
        help="Fixed weight of the dimensionless equation loss from epoch 1.",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CaseConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
    )
    seed_everything(config.random_seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data = load_case_data(config)
    split = build_data_split(config, data.load.shape[0])
    load_scale = load_scale_from_training(data.load, split.train)
    training_displacement = data.displacement[split.train]
    training_increment = np.diff(
        training_displacement,
        axis=1,
        prepend=np.zeros_like(training_displacement[:, :1]),
    )

    def rms_scale(value: np.ndarray) -> np.ndarray:
        return np.maximum(
            np.sqrt(np.mean(np.square(value), axis=(0, 1))), 1.0e-12
        )

    increment_scale = rms_scale(training_increment)
    displacement_scale = rms_scale(training_displacement)
    velocity_scale = rms_scale(data.velocity[split.train])
    external_force_scale = rms_scale(data.load[split.train])
    # The first equation contains large K*u and nonlinear-force terms which
    # cancel each other.  Scaling only by external load hides this cancellation
    # and makes its gradient much larger than the supervised gradient.  Form a
    # fixed per-DOF equation scale from the 40 labelled training samples only.
    labelled = split.labelled
    equation_terms = (
        np.einsum("ij,btj->bti", data.mass, data.acceleration[labelled]),
        np.einsum("ij,btj->bti", data.damping, data.velocity[labelled]),
        np.einsum("ij,btj->bti", data.stiffness, data.displacement[labelled]),
        data.nonlinear_force[labelled],
        data.load[labelled],
    )
    physics_scale = np.sqrt(
        sum(np.mean(np.square(term), axis=(0, 1)) for term in equation_terms)
    )
    physics_scale = np.maximum(physics_scale, 1.0e-12)
    tensors = as_torch_case(data, device)
    train_dataset = DynAnaDataset(data, split.train, split.labelled)
    # Validation responses are never used by the optimizer, but they must be
    # labelled here so checkpoint selection reflects predictive accuracy.
    val_dataset = DynAnaDataset(data, split.validation, split.validation)
    common_loader = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": DynAna_dataset_collate,
    }
    genTrain = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **common_loader
    )
    genVal = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **common_loader
    )

    model = PINN_PhyLSTM3_DisIncrement_NetBody(
        nLoad=data.load.shape[2],
        nDOF=data.displacement.shape[2],
        delta_t=data.delta_t,
        stiffness=tensors["stiffness"],
        fiber=tensors["fiber"],
        steel=tensors["steel"],
        load_scale=torch.as_tensor(load_scale, dtype=torch.float64),
        increment_scale=torch.as_tensor(increment_scale, dtype=torch.float64),
        hidden_size=args.hidden_size,
        fc_size=args.fc_size,
    ).double().to(device)
    modelLoss = PINN_MDOFSys_DisIncrement_LabPhyLoss(
        mass=tensors["mass"],
        damping=tensors["damping"],
        stiffness=tensors["stiffness"],
        displacement_scale=torch.as_tensor(displacement_scale, dtype=torch.float64),
        velocity_scale=torch.as_tensor(velocity_scale, dtype=torch.float64),
        force_scale=torch.as_tensor(physics_scale, dtype=torch.float64),
        increment_scale=torch.as_tensor(increment_scale, dtype=torch.float64),
        physics_weight=args.physics_weight,
    ).double().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=5.0e-4
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=8, gamma=0.98
    )

    log_root = Path(__file__).resolve().parent / "logs" / "PINN_PhyLSTM3"
    lossHistory = LossHistory(log_root)
    checkpoint_dir = lossHistory.save_path / "checkpoints"
    checkpoint_data = {
        "method": "PINN",
        "case_config": config.to_dict(),
        "load_scale": load_scale.tolist(),
        "hidden_size": args.hidden_size,
        "fc_size": args.fc_size,
        "n_load": int(data.load.shape[2]),
        "n_dof": int(data.displacement.shape[2]),
        "delta_t": data.delta_t,
        "tbptt_length": args.tbptt_length,
        "increment_scale": increment_scale.tolist(),
        "displacement_scale": displacement_scale.tolist(),
        "velocity_scale": velocity_scale.tolist(),
        "external_force_scale": external_force_scale.tolist(),
        "physics_scale": physics_scale.tolist(),
        "physics_weight": args.physics_weight,
        "gradient_clip": args.gradient_clip,
        "labelled_indices": split.labelled.tolist(),
    }

    print(
        f"Device: {device}; train/validation/test = "
        f"{len(split.train)}/{len(split.validation)}/{len(split.test)}; "
        f"labelled training samples = {len(split.labelled)}"
    )
    print(
        "Dimensionless scales: load RMS = "
        f"{np.array2string(load_scale, precision=3)}; equation RMS = "
        f"{np.array2string(physics_scale, precision=3)}"
    )
    start_time = time.time()
    for epoch in range(args.epochs):
        fitOneEpoch_PINN_DisIncrement_LabPhyLoss(
            model=model,
            modelLoss=modelLoss,
            lossHistory=lossHistory,
            optimizer=optimizer,
            epoch=epoch,
            genTrain=genTrain,
            genVal=genVal,
            endEpoch=args.epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
            checkpoint_data=checkpoint_data,
            tbptt_length=args.tbptt_length,
            gradient_clip=args.gradient_clip,
        )
        lr_scheduler.step()
    print(f"PINN training time: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main()
