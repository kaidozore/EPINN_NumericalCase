"""Check all LSTM/Transformer and hidden/explicit stitching variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import CaseConfig
from nets.EPINN_Full_Loss import EPINN_MDOFSys_FullDis_PhyLoss
from nets.EPINN_Full_Net import EPINN_FullDis_PhyLSTM_NetBody
from nets.EPINN_Loss import EPINN_MDOFSys_DisIncrement_PhyLoss
from nets.EPINN_Net import EPINN_PhyLSTM_NetBody
from utils.DataPreProcess import as_torch_case, load_case_data


def _finite_trainable_gradients(model: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return bool(gradients) and all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--check-steps", type=int, default=24)
    args = parser.parse_args()
    steps = max(8, int(args.check_steps))
    split_at = steps // 2
    config = CaseConfig(data_root=args.data_root, time_truncation=steps)
    data = load_case_data(config)
    tensors = as_torch_case(data, torch.device("cpu"))
    load = torch.as_tensor(
        data.load[:1, :steps].transpose(0, 2, 1)[:, None],
        dtype=torch.float64,
    )
    target_displacement = torch.as_tensor(
        data.displacement[:1, :steps], dtype=torch.float64
    )
    target_increment = torch.cat(
        (
            torch.zeros_like(target_displacement[:, :1]),
            target_displacement[:, 1:] - target_displacement[:, :-1],
        ),
        dim=1,
    )

    for response_form in ("increment", "full"):
        for sequence_model in ("lstm", "transformer"):
            for stitch_mode in ("hidden", "explicit-overlap"):
                common = {
                    "nLoad": data.load.shape[2],
                    "nLoadNL": data.load.shape[2],
                    "influence_kernel": tensors["kernel"],
                    "stiffness": tensors["stiffness"],
                    "fiber": tensors["fiber"],
                    "steel": tensors["steel"],
                    "input_increment_scale": config.displacement_increment_scale,
                    "hidden_size": 8,
                    "fc_size": 8,
                    "sequence_model": sequence_model,
                    "stitch_mode": stitch_mode,
                    "transformer_layers": 1,
                    "transformer_heads": 2,
                    "transformer_ff_size": 16,
                    "transformer_memory_length": 8,
                }
                if response_form == "increment":
                    model = EPINN_PhyLSTM_NetBody(
                        input_displacement_scale=config.displacement_scale,
                        output_increment_scale=(
                            config.displacement_increment_scale
                        ),
                        **common,
                    ).double()
                    criterion = EPINN_MDOFSys_DisIncrement_PhyLoss(
                        increment_scale=config.displacement_increment_scale,
                        displacement_scale=config.displacement_scale,
                        local_cumsum_window=20,
                        label_increment_loss_weight=0.0,
                        label_local_cumsum_loss_weight=0.0,
                        continuity_loss_weight=1.0,
                    ).double()
                else:
                    model = EPINN_FullDis_PhyLSTM_NetBody(
                        input_displacement_scale=config.displacement_scale,
                        output_displacement_scale=config.displacement_scale,
                        **common,
                    ).double()
                    criterion = EPINN_MDOFSys_FullDis_PhyLoss(
                        continuity_loss_weight=1.0
                    ).double()

                state = None
                predictions = []
                total_loss = load.new_zeros(())
                for start, stop in ((0, split_at), (split_at, steps)):
                    prediction, next_state = model.forward_chunk(
                        load[..., start:stop], state
                    )
                    if prediction["dis_nl"].shape[1] != stop - start:
                        raise AssertionError("A duplicated boundary leaked into output.")
                    has_boundary = "boundary_prediction" in prediction
                    if has_boundary != (stitch_mode == "explicit-overlap"):
                        raise AssertionError("Boundary output does not match stitch mode.")
                    if state is not None and has_boundary:
                        expected = state["last_increment"] if response_form == "increment" else state["explicit_displacement"]
                        if not torch.equal(prediction["boundary_initial"], expected):
                            raise AssertionError("Overlap target was not passed correctly.")
                    if response_form == "increment":
                        start_dis = torch.zeros_like(prediction["dis_nl"][:, :1]) if state is None else state["displacement_nl"]
                        torch.testing.assert_close(prediction["dis_nl"], start_dis + prediction["dis_increment_nl"].cumsum(dim=1))
                        torch.testing.assert_close(next_state["last_increment"], prediction["dis_increment_nl"][:, -1:])
                    target = {
                        "dis": target_displacement[:, start:stop],
                        "dis_increment": target_increment[:, start:stop],
                        "labelled": torch.zeros(1, dtype=torch.bool),
                    }
                    loss = criterion(prediction, target)
                    total_loss = total_loss + loss * ((stop - start) / steps)
                    predictions.append(prediction["dis_nl"])
                    state = next_state
                total_loss.backward()
                if not torch.isfinite(total_loss):
                    raise AssertionError("Non-finite loss.")
                if not _finite_trainable_gradients(model):
                    raise AssertionError("Missing or non-finite gradient.")
                if torch.cat(predictions, dim=1).shape[1] != steps:
                    raise AssertionError("Stitched response length changed.")
                print(
                    "[PASS] "
                    f"{response_form}/{sequence_model}/{stitch_mode}: "
                    f"loss={float(total_loss.detach()):.6e}, steps={steps}."
                )
    print("All stitching variants passed; no optimizer step was executed.")


if __name__ == "__main__":
    main()
