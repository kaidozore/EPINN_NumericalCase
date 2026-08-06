"""Train the displacement-increment E-PINN for the MATLAB MDOF case."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import CaseConfig
from nets.EPINN_Loss import EPINN_MDOFSys_DisIncrement_PhyLoss
from nets.EPINN_Net import EPINN_PhyLSTM_NetBody
from utils.callbacks import LossHistory, save_training_configuration
from utils.Dataloader import DynAnaDataset, DynAna_dataset_collate
from utils.DataPreProcess import (
    as_torch_case,
    build_data_split,
    load_case_data,
)
from utils.utils import seed_everything
from utils.utils_fit_EPINN import fitOneEpoch_EPINN_PhyLoss


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=120)
    parser.add_argument("--fc-size", type=int, default=120)
    parser.add_argument("--time-truncation", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--lr-patience", type=int, default=15)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--lr-threshold", type=float, default=1.0e-3)
    parser.add_argument("--lr-cooldown", type=int, default=2)
    parser.add_argument(
        "--increment-loss-weight", type=float, default=1.0,
        help="Weight of the primary normalized increment MSE term.",
    )
    parser.add_argument(
        "--local-cumsum-loss-weight", type=float, default=0.05,
        help="Weight of the reset local cumulative-trajectory MSE term.",
    )
    parser.add_argument(
        "--local-cumsum-window", type=int, default=32,
        help="Reset window in steps for local cumsum; must be 20--50.",
    )
    parser.add_argument(
        "--labelled-samples", type=int, default=10,
        help=(
            "Number of training samples carrying MATLAB response labels; "
            "use 0 for physics-only training."
        ),
    )
    parser.add_argument(
        "--label-increment-loss-weight", type=float, default=0.2,
        help="Weight of labelled displacement-increment MSE.",
    )
    parser.add_argument(
        "--label-local-cumsum-loss-weight", type=float, default=0.01,
        help="Weight of labelled reset local cumulative-trajectory MSE.",
    )
    parser.add_argument(
        "--gradient-clip", type=float, default=1.0,
        help="Maximum global gradient norm; use 0 to disable clipping.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument(
        "--tbptt-length", type=int, default=500,
        help="Consecutive steps per truncated-backpropagation chunk.",
    )
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
        time_truncation=args.time_truncation,
        sequence_length=args.sequence_length,
        labelled_sample_count=args.labelled_samples,
    )
    seed_everything(config.random_seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data = load_case_data(config)
    split = build_data_split(config, data.load.shape[0])
    # Use one fixed physical reference force for all samples and DOFs.  A
    # doubled load therefore remains doubled after scaling.
    tensors = as_torch_case(data, device)
    train_dataset = DynAnaDataset(
        data, split.train, labelled_indices=split.labelled
    )
    val_dataset = DynAnaDataset(data, split.validation)
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

    model = EPINN_PhyLSTM_NetBody(
        nLoad=data.load.shape[2],
        nLoadNL=data.load.shape[2],
        influence_kernel=tensors["kernel"],
        stiffness=tensors["stiffness"],
        fiber=tensors["fiber"],
        steel=tensors["steel"],
        input_increment_scale=config.displacement_increment_scale,
        output_increment_scale=config.displacement_increment_scale,
        hidden_size=args.hidden_size,
        fc_size=args.fc_size,
    ).double().to(device)
    modelLoss = EPINN_MDOFSys_DisIncrement_PhyLoss(
        increment_scale=config.displacement_increment_scale,
        displacement_scale=config.displacement_scale,
        increment_loss_weight=args.increment_loss_weight,
        local_cumsum_loss_weight=args.local_cumsum_loss_weight,
        local_cumsum_window=args.local_cumsum_window,
        label_increment_loss_weight=args.label_increment_loss_weight,
        label_local_cumsum_loss_weight=(
            args.label_local_cumsum_loss_weight
        ),
    ).double().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=5.0e-4
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        threshold_mode="rel",
        cooldown=args.lr_cooldown,
        min_lr=args.min_learning_rate,
    )

    log_root = Path(__file__).resolve().parent / "logs" / "EPINN_PhyLSTM"
    lossHistory = LossHistory(log_root)
    checkpoint_dir = lossHistory.save_path / "checkpoints"
    checkpoint_data = {
        "method": "EPINN",
        "case_config": config.to_dict(),
        "data_split": {
            "train_indices": split.train.tolist(),
            "validation_indices": split.validation.tolist(),
            "test_indices": split.test.tolist(),
            "labelled_train_indices": split.labelled.tolist(),
            "priority_nonlinear_indices": (
                split.priority_nonlinear.tolist()
            ),
        },
        "network_input": "elastic_displacement_increment_from_fixed_SCL",
        "network_output": "nonlinear_total_displacement_increment",
        "scl_target_detached": True,
        "loss": (
            "increment_loss_weight*MSE((LSTM_increment-SCL_increment)/"
            "fixed_increment_scale) + local_cumsum_loss_weight*MSE("
            "local_cumsum(LSTM_increment-SCL_increment)/"
            "fixed_displacement_scale) + label_increment_loss_weight*"
            "MSE((LSTM_increment-labelled_increment)/fixed_increment_scale) "
            "+ label_local_cumsum_loss_weight*MSE(local_cumsum("
            "LSTM_increment-labelled_increment)/fixed_displacement_scale)"
        ),
        "input_increment_scale": float(config.displacement_increment_scale),
        "hidden_size": args.hidden_size,
        "fc_size": args.fc_size,
        "n_load": int(data.load.shape[2]),
        "n_dof": int(data.displacement.shape[2]),
        "delta_t": data.delta_t,
        "tbptt_length": args.tbptt_length,
        "gradient_clip": args.gradient_clip,
        "increment_scale": float(config.displacement_increment_scale),
        "displacement_scale": float(config.displacement_scale),
        "increment_loss_weight": args.increment_loss_weight,
        "local_cumsum_loss_weight": args.local_cumsum_loss_weight,
        "local_cumsum_window": args.local_cumsum_window,
        "labelled_samples": args.labelled_samples,
        "label_increment_loss_weight": args.label_increment_loss_weight,
        "label_local_cumsum_loss_weight": (
            args.label_local_cumsum_loss_weight
        ),
        "output_increment_scale": float(config.displacement_increment_scale),
        "output_head_init_gain": float(model.output_head_init_gain),
    }
    configuration_path = save_training_configuration(
        lossHistory.save_path,
        {
            "script": Path(__file__).name,
            "command_line_arguments": sys.argv[1:],
            "parsed_arguments": vars(args),
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "data_split": {
                "train_indices": split.train.tolist(),
                "validation_indices": split.validation.tolist(),
                "test_indices": split.test.tolist(),
                "labelled_train_indices": split.labelled.tolist(),
                "priority_nonlinear_indices": (
                    split.priority_nonlinear.tolist()
                ),
            },
            "model_and_loss": checkpoint_data,
            "optimizer": {
                "name": "Adam",
                "learning_rate": args.learning_rate,
                "weight_decay": 5.0e-4,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
            },
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "monitor": "val_loss",
                "mode": "min",
                "factor": args.lr_factor,
                "patience": args.lr_patience,
                "threshold": args.lr_threshold,
                "threshold_mode": "rel",
                "cooldown": args.lr_cooldown,
                "min_learning_rate": args.min_learning_rate,
            },
        },
    )

    print(
        f"Device: {device}; train/validation/test = "
        f"{len(split.train)}/{len(split.validation)}/{len(split.test)}"
    )
    print(
        f"Labelled training samples: {len(split.labelled)}; "
        f"indices = {split.labelled.tolist()}"
    )
    print(
        "Priority nonlinear training samples (MATLAB indices): "
        f"{(split.priority_nonlinear + 1).tolist()}"
    )
    print(f"Training configuration: {configuration_path}")
    print(
        "E-PINN: elastic displacement increments -> LSTM -> nonlinear total "
        "increments; reset local cumsum windows constrain local trajectories."
    )
    start_time = time.time()
    for epoch in range(args.epochs):
        _, validation_loss = fitOneEpoch_EPINN_PhyLoss(
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
            gradient_clip=(
                None if args.gradient_clip <= 0.0 else args.gradient_clip
            ),
        )
        lr_scheduler.step(validation_loss)
    print(f"E-PINN training time: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main()
