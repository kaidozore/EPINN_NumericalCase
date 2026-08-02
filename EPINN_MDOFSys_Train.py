"""Train the displacement-increment E-PINN for the MATLAB MDOF case."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import CaseConfig
from nets.EPINN_Loss import EPINN_MDOFSys_DisIncrement_PhyLoss
from nets.EPINN_Net import EPINN_PhyLSTM_NetBody
from utils.callbacks import LossHistory
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
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
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
    train_dataset = DynAnaDataset(data, split.train)
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
        displacement_scale=config.displacement_scale,
        hidden_size=args.hidden_size,
        fc_size=args.fc_size,
    ).double().to(device)
    modelLoss = EPINN_MDOFSys_DisIncrement_PhyLoss(
        config.displacement_increment_scale
    ).double().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=5.0e-4
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.98
    )

    log_root = Path(__file__).resolve().parent / "logs" / "EPINN_PhyLSTM"
    lossHistory = LossHistory(log_root)
    checkpoint_dir = lossHistory.save_path / "checkpoints"
    checkpoint_data = {
        "method": "EPINN",
        "case_config": config.to_dict(),
        "network_input": "elastic_displacement_increment_from_fixed_SCL",
        "input_increment_scale": float(config.displacement_increment_scale),
        "hidden_size": args.hidden_size,
        "fc_size": args.fc_size,
        "n_load": int(data.load.shape[2]),
        "n_dof": int(data.displacement.shape[2]),
        "delta_t": data.delta_t,
        "tbptt_length": args.tbptt_length,
        "network_level_scale": float(config.displacement_scale),
    }

    print(
        f"Device: {device}; train/validation/test = "
        f"{len(split.train)}/{len(split.validation)}/{len(split.test)}"
    )
    start_time = time.time()
    for epoch in range(args.epochs):
        fitOneEpoch_EPINN_PhyLoss(
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
        )
        lr_scheduler.step()
    print(f"E-PINN training time: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main()
