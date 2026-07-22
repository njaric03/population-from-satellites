"""
Normalizacija po opsegu, Dataset i DataLoader pomocnici.

Zajednicki za pristupe sa jednim .npy fajlom po naselju (Sentinel-2 cutout-i,
footprint otisci). tiles_train koristi sopstveni NaseljaTiles dataset i
napravi_loadere zbog collate_fn za agregacionu loss, ali moze koristiti
stats_po_opsegu tako sto prosledi listu putanja plocica.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Broj DataLoader radnika ogranicen brojem CPU jezgara
NW: int = min(8, (os.cpu_count() or 2))


def stats_po_opsegu(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean i std iz uzorka .npy fajlova.

    Pozivati SAMO nad trening skupom tekuceg folda (bez curenja u validaciju).

    Args:
        paths: lista putanja ka .npy fajlovima (cutout-i ili otisci zgrada ili
               plocice - bilo koja lista; tiles_train prosledjuje uniju svih
               putanja plocica trening naselja tekuceg folda).

    Returns:
        mean, std - oba oblika (1, C, 1, 1), dtype float32.
    """
    uzorak = np.stack(
        [np.load(p) for p in random.sample(paths, min(400, len(paths)))]
    ).astype("float32")
    mean = uzorak.mean((0, 2, 3), keepdims=True).astype("float32")
    std  = uzorak.std( (0, 2, 3), keepdims=True).astype("float32") + 1e-6
    return mean, std


class Naselja(Dataset):
    """PyTorch Dataset za pristupe sa jednim .npy fajlom po naselju.

    ``frame`` mora imati kolone:

    * ``path``  - apsolutna putanja ka .npy fajlu
    * ``y``     - float32 ciljna velicina (npr. ``log1p(pop)``)
    """

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
    """Inicijalizator radnika DataLoader-a za reproduktivnost.

    Prosledjuje se kao ``worker_init_fn`` u DataLoader.
    Beleska: da bismo preneli seed, bice kreiran closure - videti
    ``napravi_loadere`` za primer.
    """
    s = seed + wid
    np.random.seed(s)
    random.seed(s)


def napravi_loadere(
    train_frame,
    val_frame,
    batch_size: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Pravi ``(train_dl, val_dl)`` iz DataFrame-ova koji vec imaju kolone
    ``path`` i ``y``.

    Normalizacija se racuna SAMO iz ``train_frame`` ovog folda
    (nema curenja u validaciju). Briga o koloni ``y`` je na pozivaocu:
    npr. ``frame["y"] = np.log1p(frame["pop"]).astype("float32")``
    pre poziva ove funkcije.

    Args:
        train_frame:  trening DataFrame; mora imati ``path`` i ``y``.
        val_frame:    validacioni DataFrame; mora imati ``path`` i ``y``.
        batch_size:   velicina batch-a.
        seed:         seed za DataLoader Generator i radnike.

    Returns:
        ``(train_dl, val_dl)``
    """
    mean, std = stats_po_opsegu(train_frame.path.tolist())

    def _sw(wid: int) -> None:
        seed_worker(wid, seed)

    gen = torch.Generator().manual_seed(seed)
    tdl = DataLoader(
        Naselja(train_frame, mean, std, augment=True),
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
        Naselja(val_frame, mean, std),
        batch_size=batch_size,
        num_workers=NW,
        pin_memory=True,
        persistent_workers=NW > 0,
        prefetch_factor=4 if NW else None,
    )
    return tdl, vdl
