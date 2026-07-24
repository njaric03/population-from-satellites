from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    # Seed za random, numpy, torch i cuda; zvati pre bilo kakvog uzorkovanja.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prodji(
    net: nn.Module,
    loader,
    treniraj: bool,
    loss_fn: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    freeze_bn: bool = False,
    device: str = "cpu",
) -> tuple[float, np.ndarray, np.ndarray]:
    # Jedan prolaz kroz DataLoader; vraca (avg_loss, predikcije, ciljevi). Za ulaze oblika
    # (x, y). tiles_train ima svoj prodji sa scatter_add agregacionim loss-om i ne koristi
    # ovaj.
    net.train(treniraj)
    if freeze_bn:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    cast_kwargs: dict = {"enabled": use_amp and device.startswith("cuda")}
    if amp_dtype is not None:
        cast_kwargs["dtype"] = amp_dtype

    ukupno, P, Y = 0.0, [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(treniraj):
            with torch.autocast("cuda", **cast_kwargs):
                out  = net(x)
                loss = loss_fn(out, y)
            if treniraj:
                optim.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()

        ukupno += loss.item() * len(x)
        P.append(out.detach().float().cpu().numpy())
        Y.append(y.cpu().numpy())

    return ukupno / len(loader.dataset), np.concatenate(P).ravel(), np.concatenate(Y).ravel()


def dvofazni_trening(
    net: nn.Module,
    epoha_fn: Callable[[torch.optim.Optimizer, int, bool], float],
    epochs_head: int,
    epochs_finetune: int,
    head_lr: float,
    finetune_lr: float,
    head_prefix: str = "fc",
) -> tuple[float, dict]:
    # Faza 1 uci samo glavu (backbone zamrznut), faza 2 fine-tune ceo model. epoha_fn(opt,
    # korak, freeze_bn) je closure iz notebooka: odradi jedan trening i val prolaz, loguje
    # MLflow metrike i vrati val R2. Vraca (best_val_r2, best_state_dict); state_dict je CPU
    # kopija.
    best_r2, best_state = -1e9, None

    # Faza 1: samo glava, backbone zamrznut
    for naziv, p in net.named_parameters():
        p.requires_grad = naziv.startswith(head_prefix)
    opt1 = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=head_lr
    )
    for e in range(epochs_head):
        epoha_fn(opt1, e, True)

    # Faza 2: ceo model, cosine LR
    for p in net.parameters():
        p.requires_grad = True
    opt2  = torch.optim.AdamW(net.parameters(), lr=finetune_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_finetune)
    for e in range(epochs_finetune):
        r2 = epoha_fn(opt2, epochs_head + e, False)
        sched.step()
        if r2 > best_r2:
            best_r2    = r2
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    return best_r2, best_state
