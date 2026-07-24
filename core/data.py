from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Broj DataLoader radnika ogranicen brojem CPU jezgara
NW: int = min(8, (os.cpu_count() or 2))


def channel_stats(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # Per-channel mean i std, oblika (1, C, 1, 1). Zvati SAMO nad trening skupom tekuceg
    # folda, inace curi u validaciju.
    uzorak = np.stack(
        [np.load(p) for p in random.sample(paths, min(400, len(paths)))]
    ).astype("float32")
    mean = uzorak.mean((0, 2, 3), keepdims=True).astype("float32")
    std  = uzorak.std( (0, 2, 3), keepdims=True).astype("float32") + 1e-6
    return mean, std


class Settlements(Dataset):
    # Dataset za jedan .npy po naselju; frame treba kolone path i y.

    def __init__(
        self,
        frame,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.frame   = frame.reset_index(drop=True)
        self.mean    = mean
        self.std     = std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i):
        red = self.frame.iloc[i]
        x = (np.load(red.path).astype("float32") - self.mean[0]) / self.std[0]
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
            x = np.rot90(x, np.random.randint(4), axes=(1, 2))
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.tensor([red.y], dtype=torch.float32),
        )


def seed_worker(wid: int, seed: int = 42) -> None:
    # worker_init_fn za DataLoader; seed se prenosi kroz closure.
    s = seed + wid
    np.random.seed(s)
    random.seed(s)


def make_loaders(
    train_frame,
    val_frame,
    batch_size: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    # (train_dl, val_dl); normalizacija se racuna samo iz trening folda. Kolonu y postavlja
    # pozivalac, npr. np.log1p(frame["pop"]).
    mean, std = channel_stats(train_frame.path.tolist())

    def _sw(wid: int) -> None:
        seed_worker(wid, seed)

    gen = torch.Generator().manual_seed(seed)
    tdl = DataLoader(
        Settlements(train_frame, mean, std, augment=True),
        batch_size=batch_size,
        shuffle=True,
        # poslednji batch od tacno 1 uzorka ruši BatchNorm u trening modu
        # ("Expected more than 1 value per channel"); tada ga preskoci
        drop_last=len(train_frame) % batch_size == 1,
        generator=gen,
        worker_init_fn=_sw if NW else None,
        num_workers=NW,
        pin_memory=True,
        persistent_workers=NW > 0,
        prefetch_factor=4 if NW else None,
    )
    vdl = DataLoader(
        Settlements(val_frame, mean, std),
        batch_size=batch_size,
        num_workers=NW,
        pin_memory=True,
        persistent_workers=NW > 0,
        prefetch_factor=4 if NW else None,
    )
    return tdl, vdl
