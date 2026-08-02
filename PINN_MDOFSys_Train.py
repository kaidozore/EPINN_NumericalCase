"""Train the conventional physics-informed LSTM baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import CaseConfig
from nets.PINN_Loss import PINN_MDOFSys_DisIncrement_PhyLoss
from nets.PINN_Net import PINN_PhyLSTM3_DisIncrement_NetBody
from utils.callbacks import LossHistory
from utils.Dataloader import DynAnaDataset, DynAna_dataset_collate
from utils.DataPreProcess import as_torch_case, build_data_split, load_case_data
from utils.utils import seed_everything
from utils.utils_fit_PINN import fitOneEpoch_PINN_DisIncrement_PhyLoss


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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument(
        "--tbptt-length", type=int, default=500,
        help="Consecutive steps per truncated-backpropagation chunk.",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
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
    tensors = as_torch_case(data, device)
    train_dataset = DynAnaDataset(data, split.train)
    val_dataset = DynAnaDataset(data, split.validation)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": DynAna_dataset_collate,
    }
    genTrain = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_options
    )
    genVal = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_options
    )

    model = PINN_PhyLSTM3_DisIncrement_NetBody(
        nLoad=data.load.shape[2],
        nDOF=data.displacement.shape[2],
        delta_t=data.delta_t,
        stiffness=tensors["stiffness"],
        influence_kernel=tensors["kernel"],
        fiber=tensors["fiber"],
        steel=tensors["steel"],
        input_increment_scale=config.displacement_increment_scale,
        input_displacement_scale=config.displacement_scale,
        output_displacement_scale=config.displacement_scale,
        hidden_size=args.hidden_size,
        fc_size=args.fc_size,
    ).double().to(device)
    modelLoss = PINN_MDOFSys_DisIncrement_PhyLoss(
        tensors["mass"], tensors["damping"], force_scale=1.0e5
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
        "network_input": "elastic_displacement_history_from_fixed_SCL",
        "network_output": "nonlinear_total_displacement_history",
        "kinematics": "second_order_central_difference",
        "loss": "mean(((M*a+C*v+Fint-Fwave)/1e5)^2)",
        "force_scale_n": 1.0e5,
        "input_increment_scale": float(config.displacement_increment_scale),
        "input_displacement_scale": float(config.displacement_scale),
        "output_displacement_scale": float(config.displacement_scale),
        "hidden_size": args.hidden_size,
        "fc_size": args.fc_size,
        "n_load": int(data.load.shape[2]),
        "n_dof": int(data.displacement.shape[2]),
        "delta_t": data.delta_t,
        "tbptt_length": args.tbptt_length,
    }

    print(
        f"Device: {device}; train/validation/test = "
        f"{len(split.train)}/{len(split.validation)}/{len(split.test)}"
    )
    print(
        "PINN: elastic displacement history -> LSTM -> nonlinear total "
        "displacement history; loss = "
        "MSE((M*a + C*v + Fint - Fwave) / 1e5 N)."
    )
    start_time = time.time()
    for epoch in range(args.epochs):
        _, validation_loss = fitOneEpoch_PINN_DisIncrement_PhyLoss(
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
        lr_scheduler.step(validation_loss)
    print(f"PINN training time: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main()
