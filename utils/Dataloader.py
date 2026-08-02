"""PyTorch dataset using the layout of the supplied ES_Case2 code."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.DataPreProcess import CaseData


class DynAnaDataset(Dataset):
    def __init__(
        self,
        data: CaseData,
        indices: Iterable[int],
        labelled_indices: Iterable[int] = (),
    ) -> None:
        self.data = data
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.labelled = set(int(i) for i in labelled_indices)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        # load: [1,nLoad,timeLength], matching the reference project.
        load = np.ascontiguousarray(
            self.data.load[index].T[None, :, :]
        )
        target = {
            "dis": np.ascontiguousarray(self.data.displacement[index]),
            "dis_increment": np.ascontiguousarray(
                np.diff(
                    self.data.displacement[index],
                    axis=0,
                    prepend=np.zeros_like(self.data.displacement[index, :1]),
                )
            ),
            "vel": np.ascontiguousarray(self.data.velocity[index]),
            "acc": np.ascontiguousarray(self.data.acceleration[index]),
            "force_internal": np.ascontiguousarray(
                self.data.internal_force[index]
            ),
            "force_nonlinear": np.ascontiguousarray(
                self.data.nonlinear_force[index]
            ),
            "labelled": np.asarray(index in self.labelled),
            "sample_index": np.asarray(index, dtype=np.int64),
        }
        return load, target


def DynAna_dataset_collate(batch):
    loads = torch.as_tensor(
        np.stack([item[0] for item in batch]), dtype=torch.float64
    )
    targets = {}
    for key in batch[0][1]:
        targets[key] = torch.as_tensor(
            np.stack([item[1][key] for item in batch])
        )
    return loads, targets
