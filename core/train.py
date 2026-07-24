from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    """Postavlja seed za random, numpy, torch i cuda.

    Pozivati jednom na pocetku config celije u svakom notebooku pre bilo kakvog
    slucajnog uzorkovanja ili inicijalizacije modela.
    """
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
    """Jedan prolaz kroz DataLoader (trening ili evaluacija).

    Za cutout i footprint pristupe koji vracaju ``(x, y)`` parove.
    **Ne koristiti** za tiles_train - tamo je ``prodji`` lokalno definisan
    sa scatter_add agregacionom loss-om.

    Args:
        net:       PyTorch model.
        loader:    DataLoader koji vraca ``(x, y)`` parove.
        treniraj:  ``True`` = trening mod, ``False`` = evaluacija.
        loss_fn:   kriterijum gubitka (npr. ``nn.HuberLoss()``).
        optim:     optimizer - obavezan kada ``treniraj=True``.
        scaler:    ``torch.amp.GradScaler`` ili ``None``; ako je ``None``
                   koristi se obicni ``.backward()``.
        use_amp:   aktivira ``torch.autocast``.
        amp_dtype: ``torch.float16`` ili ``torch.bfloat16``
                   (``None`` = default autocast dtype). bf16 (A10) ne zahteva
                   GradScaler pa prosledjivati ``scaler=None`` tada.
        freeze_bn: drzati BatchNorm u eval modu - za fazu 1 ucenja samo glave.
        device:    uredjaj na koji se podaci salju (``"cuda"`` ili ``"cpu"``).

    Returns:
        ``(avg_loss, predictions, targets)`` - predictions i targets su 1D
        numpy float32 nizovi poravnati sa redosledom loadera.
    """
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
    """Dvofazni trening koji je identican u svim notebucima.

    * **Faza 1** - zamrznut backbone (BN u eval), uci se samo glava
      (parametri ciji naziv pocinje sa ``head_prefix``) ``epochs_head``
      epoha sa AdamW i ``head_lr``.
    * **Faza 2** - odmrznut ceo model, fine-tuning sa CosineAnnealingLR
      i ``finetune_lr``. Cuva se best checkpoint po val R2.

    Args:
        net:             PyTorch model (na DEVICE-u).
        epoha_fn:        closure ``epoha(opt, korak, freeze_bn=False) -> float``
                         koji pokrece jedan trening+val prolaz, loguje MLflow
                         metrike i vraca val R2. Definise se lokalno u
                         ``treniraj_fold`` svakog notebooka (zatvara train_dl,
                         val_dl, loss_fn, scaler, device, MLflow kontekst itd.).
        epochs_head:     broj epoha faze 1.
        epochs_finetune: broj epoha faze 2.
        head_lr:         learning rate za fazu 1.
        finetune_lr:     learning rate za fazu 2.
        head_prefix:     prefiks naziva parametara glave; ``"fc"`` za timm
                         resnet18 sa num_classes=1 (podrazumevano),
                         ``"head"`` za multimodalni model sa sopstvenom glavom.

    Returns:
        ``(best_val_r2, best_state_dict)`` - best_state_dict je recnik tezina
        (CPU kopija) koji treba ucitati nazad u model pre OOF predikcije.
    """
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
