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
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-5)
    parser.add_argument(
        "--supervision",
        choices=("full", "partial", "none"),
        default="full",
        help="Response labels used by PINN; physics always uses all train samples.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument(
        "--tbptt-length", type=int, default=500,
        help="Consecutive steps per truncated-backpropagation chunk.",
    )
    parser.add_argument(
        "--physics-weight", type=float, default=1.0e-6,
        help="Fixed weight of the scaled equation loss from epoch 1.",
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
    # Fixed engineering reference quantities are shared by every sample and
    # every DOF.  No sample-wise, DOF-wise or dataset-derived RMS scaling is
    # used, so proportional changes in load amplitude remain proportional at
    # the network input and in the equation residual.
    n_dof = data.displacement.shape[2]
    increment_scale = np.full(
        n_dof, config.displacement_increment_scale, dtype=np.float64
    )
    displacement_scale = np.full(
        n_dof, config.displacement_scale, dtype=np.float64
    )
    physics_scale = np.full(n_dof, config.force_scale, dtype=np.float64)
    validation_displacement = data.displacement[split.validation]
    validation_increment = np.diff(
        validation_displacement,
        axis=1,
        prepend=np.zeros_like(validation_displacement[:, :1]),
    )
    zero_response_baseline = {
        "val_loss": float(
            np.mean(np.square(validation_increment / config.displacement_increment_scale))
            + np.mean(np.square(validation_displacement / config.displacement_scale))
        ),
        "val_physics": float(
            np.mean(np.square(data.load[split.validation] / config.force_scale))
        ),
        "val_displacement_rmse_m": float(
            np.sqrt(np.mean(np.square(validation_displacement)))
        ),
    }
    tensors = as_torch_case(data, device)
    # Full-data diagnostic: all 170 training responses are supervised while
    # every one of them also retains the equation-of-motion loss.  This makes
    # each shuffled batch fully labelled and removes label-count fluctuations.
    supervised_indices = {
        "full": split.train,
        "partial": split.labelled,
        "none": np.empty(0, dtype=np.int64),
    }[args.supervision]
    train_dataset = DynAnaDataset(data, split.train, supervised_indices)
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
        influence_kernel=tensors["kernel"],
        fiber=tensors["fiber"],
        steel=tensors["steel"],
        input_increment_scale=torch.as_tensor(
            increment_scale, dtype=torch.float64
        ),
        displacement_scale=torch.as_tensor(
            displacement_scale, dtype=torch.float64
        ),
        hidden_size=args.hidden_size,
        fc_size=args.fc_size,
    ).double().to(device)
    elastic_increment_sse = 0.0
    elastic_displacement_sse = 0.0
    elastic_increment_count = 0
    elastic_displacement_count = 0
    with torch.no_grad():
        for validation_load, validation_target in genVal:
            load_sequence = (
                validation_load.to(device).squeeze(1).transpose(1, 2)
            )
            _, elastic_increment, elastic_displacement = (
                model.ElasticInput_Module(load_sequence)
            )
            increment_target = validation_target["dis_increment"].to(device)
            displacement_target = validation_target["dis"].to(device)
            elastic_increment_sse += float(
                torch.sum((elastic_increment - increment_target).pow(2))
            )
            elastic_displacement_sse += float(
                torch.sum((elastic_displacement - displacement_target).pow(2))
            )
            elastic_increment_count += increment_target.numel()
            elastic_displacement_count += displacement_target.numel()
    elastic_response_baseline = {
        "val_loss": (
            elastic_increment_sse
            / elastic_increment_count
            / config.displacement_increment_scale**2
            + elastic_displacement_sse
            / elastic_displacement_count
            / config.displacement_scale**2
        ),
        "val_increment_rmse_m": float(
            np.sqrt(elastic_increment_sse / elastic_increment_count)
        ),
        "val_displacement_rmse_m": float(
            np.sqrt(elastic_displacement_sse / elastic_displacement_count)
        ),
    }
    modelLoss = PINN_MDOFSys_DisIncrement_LabPhyLoss(
        mass=tensors["mass"],
        damping=tensors["damping"],
        stiffness=tensors["stiffness"],
        displacement_scale=torch.as_tensor(displacement_scale, dtype=torch.float64),
        force_scale=torch.as_tensor(physics_scale, dtype=torch.float64),
        increment_scale=torch.as_tensor(increment_scale, dtype=torch.float64),
        physics_weight=args.physics_weight,
    ).double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
    )

    log_root = Path(__file__).resolve().parent / "logs" / "PINN_PhyLSTM3"
    lossHistory = LossHistory(log_root)
    checkpoint_dir = lossHistory.save_path / "checkpoints"
    checkpoint_data = {
        "method": "PINN",
        "case_config": config.to_dict(),
        "network_input": "elastic_displacement_increment_from_fixed_SCL",
        "input_increment_scale": increment_scale.tolist(),
        "hidden_size": args.hidden_size,
        "fc_size": args.fc_size,
        "n_load": int(data.load.shape[2]),
        "n_dof": int(data.displacement.shape[2]),
        "delta_t": data.delta_t,
        "tbptt_length": args.tbptt_length,
        "increment_scale": increment_scale.tolist(),
        "displacement_scale": displacement_scale.tolist(),
        "network_level_scale": displacement_scale.tolist(),
        "increment_scale_fixed": float(config.displacement_increment_scale),
        "displacement_scale_fixed": float(config.displacement_scale),
        "physics_scale": physics_scale.tolist(),
        "physics_weight": args.physics_weight,
        "gradient_clip": args.gradient_clip,
        "labelled_indices": supervised_indices.tolist(),
        "supervision": args.supervision,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "patience": args.lr_patience,
            "factor": args.lr_factor,
            "min_lr": args.min_learning_rate,
        },
        "zero_response_baseline": zero_response_baseline,
        "elastic_response_baseline": elastic_response_baseline,
    }

    print(
        f"Device: {device}; train/validation/test = "
        f"{len(split.train)}/{len(split.validation)}/{len(split.test)}; "
        f"labelled training samples = {len(supervised_indices)} "
        f"(supervision={args.supervision})"
    )
    print(
        "Fixed scales: raw wave load enters SCL in N; elastic/input and target "
        "displacement increment = "
        f"{config.displacement_increment_scale:.3e} m; displacement = "
        f"{config.displacement_scale:.3e} m"
    )
    print(
        "Zero-response validation baseline: loss = "
        f"{zero_response_baseline['val_loss']:.6e}; physics = "
        f"{zero_response_baseline['val_physics']:.6e}; u_RMSE = "
        f"{zero_response_baseline['val_displacement_rmse_m']:.6e} m"
    )
    print(
        "Elastic-response validation baseline: loss = "
        f"{elastic_response_baseline['val_loss']:.6e}; du_RMSE = "
        f"{elastic_response_baseline['val_increment_rmse_m']:.6e} m; "
        "u_RMSE = "
        f"{elastic_response_baseline['val_displacement_rmse_m']:.6e} m"
    )
    start_time = time.time()
    selection_metric = "physics" if args.supervision == "none" else "data"
    for epoch in range(args.epochs):
        _, validation_loss = fitOneEpoch_PINN_DisIncrement_LabPhyLoss(
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
            selection_metric=selection_metric,
        )
        lr_scheduler.step(validation_loss)
    print(f"PINN training time: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main()
