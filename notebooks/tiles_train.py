# Databricks notebook source
# MAGIC %md
# MAGIC # Procena stanovnika preko plocica (agregaciona loss)
# MAGIC
# MAGIC Svako naselje je pokriveno disjunktnim 2.24km Sentinel plocicama. ResNet-18 daje broj stanovnika
# MAGIC po plocici (softplus, nenegativno); suma plocica jednog naselja je predikcija za to naselje.
# MAGIC Loss poredi log1p(sumu) sa log1p(popisa). Time slika i labela odgovaraju i velikim naseljima.
# MAGIC Podela po opstinama (GroupKFold). Pracenje kroz MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instalacija

# COMMAND ----------

# MAGIC %pip install -q timm mlflow
# MAGIC try:
# MAGIC     dbutils.library.restartPython()
# MAGIC except NameError:
# MAGIC     pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Konfiguracija

# COMMAND ----------

# DBTITLE 1,Postavke hiperparametara
import sys
sys.path.insert(0, "/Workspace/Users/korisnik/du-procena-stanovnistva/src")

import os, glob, zipfile
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm, mlflow
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from procena import (
    seed_everything, seed_worker, dvofazni_trening,
    stats_po_opsegu, NW,
    napravi_foldove, oof_metrics, cv_summary_figure,
)

# Colab: "/content/tiles_data" (raspakuje tiles_upload.zip). Databricks: UC Volume (isti "data" folder kao ostali pristupi).
DATA_DIR = "/content/tiles_data" if os.path.isdir("/content") else "/Volumes/katalog/deep_learning/raw_data/data"
if os.path.isdir("/content") and not os.path.isdir(DATA_DIR + "/tiles"):
    zipfile.ZipFile("/content/tiles_upload.zip").extractall(DATA_DIR)
TILES = DATA_DIR + "/tiles"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# svi hiperparametri na jednom mestu (jedini izvor istine)
CFG = {
    "num_bands": 6,                  # broj Sentinel-2 opsega / ulaznih kanala
    "img_px": 224,                   # dimenzija plocice u pikselima
    "naselja_po_batchu": 8,           # batch = 8 naselja (promenljiv broj plocica po naselju)
    "epochs_head": 3,
    "epochs_finetune": 40,
    "head_lr": 1e-3,
    "finetune_lr": 3e-4,
    "seed": 42,                      # za reproduktivnost (random/numpy/torch/cuda + DataLoader radnici)
}
seed_everything(CFG["seed"])
print("device:", DEVICE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Podaci i podela (po naseljima, grupisano po opstini)

# COMMAND ----------

# DBTITLE 1,Ucitavanje podataka / podesavanje CV
idx = pd.read_csv(DATA_DIR + "/tiles_index.csv")
idx["fpath"] = idx.path.map(lambda p: TILES + "/" + p)
tab = (
    pd.read_parquet(DATA_DIR + "/naselje_table.parquet")[["naselje_maticni_broj", "opstina_maticni_broj"]]
)
nasel = (
    idx
    .groupby("naselje_maticni_broj")
    .agg(pop=("pop", "first"), n_tiles=("path", "size")).reset_index()
    .merge(tab, on="naselje_maticni_broj", how="left")
)
tiles_by = {mb: g.fpath.tolist() for mb, g in idx.groupby("naselje_maticni_broj")}
print(f"plocica {len(idx)} | naselja {len(nasel)} | plocica/naselje med {int(nasel.n_tiles.median())} max {int(nasel.n_tiles.max())}")

FOLDS = napravi_foldove(nasel)
N_FOLDS = len(FOLDS)
broj_opstina = nasel["opstina_maticni_broj"].nunique()
print(f"naselja {len(nasel)} | opstina {broj_opstina} | foldova {N_FOLDS}")
for i, (t, v) in enumerate(FOLDS):
    print(f"  fold {i}: trening {len(t)} / val {len(v)} naselja ({v['opstina_maticni_broj'].nunique()} opstina)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalizacija i dataset

# COMMAND ----------

# DBTITLE 1,Normalizacija i dataset
# stats_po_opsegu i seed_worker su uvezeni iz procena.data;
# NaseljaTiles, collate i napravi_loadere su tiles-specificni pa ostaju ovde.

class NaseljaTiles(Dataset):
    def __init__(self, frame, mean, std, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        red = self.frame.iloc[i]
        paths = tiles_by[red.naselje_maticni_broj]
        xs = []
        for p in paths:
            x = (np.load(p).astype("float32") - self.mean[0]) / self.std[0]
            if self.augment:
                if np.random.rand() < 0.5: x = x[:, :, ::-1]
                if np.random.rand() < 0.5: x = x[:, ::-1, :]
                x = np.rot90(x, np.random.randint(4), axes=(1, 2))
            xs.append(np.ascontiguousarray(x))
        return torch.from_numpy(np.stack(xs)), float(red["pop"])

def collate(batch):
    tiles = torch.cat([b[0] for b in batch], 0)                       # [sumT, 6, 224, 224]
    pops = torch.tensor([b[1] for b in batch], dtype=torch.float32)   # [B]
    nid = torch.cat([torch.full((b[0].shape[0],), i, dtype=torch.long) for i, b in enumerate(batch)])
    return tiles, nid, pops

def napravi_loadere(train_nasel_frame, val_nasel_frame):
    """Vrati (train_dl, val_dl) sa normalizacijom racunatom iz trening plocica ovog folda."""
    svi_tr = [p for mb in train_nasel_frame.naselje_maticni_broj for p in tiles_by[mb]]
    mean, std = stats_po_opsegu(svi_tr)   # koristimo zajednicki stats_po_opsegu iz procena.data
    gen = torch.Generator().manual_seed(CFG["seed"])
    _sw = (lambda wid: seed_worker(wid, CFG["seed"])) if NW else None
    tdl = DataLoader(NaseljaTiles(train_nasel_frame, mean, std, augment=True),
                     batch_size=CFG["naselja_po_batchu"], shuffle=True,
                     collate_fn=collate, generator=gen, worker_init_fn=_sw,
                     num_workers=NW, pin_memory=True, persistent_workers=NW > 0,
                     prefetch_factor=4 if NW else None)
    vdl = DataLoader(NaseljaTiles(val_nasel_frame, mean, std),
                     batch_size=CFG["naselja_po_batchu"], collate_fn=collate,
                     num_workers=NW, pin_memory=True, persistent_workers=NW > 0,
                     prefetch_factor=4 if NW else None)
    return tdl, vdl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model i agregaciona loss

# COMMAND ----------

# timm prosiruje prvi konvolucioni sloj sa 3 na 6 kanala (in_chans); model se pravi po foldu u treniraj_fold
loss_fn = nn.HuberLoss()          # log-prostor: Huber(delta=1) na log1p skali
POP_SCALE = 1000.0                # skala za linearni loss (delta ~ 1000 stanovnika; gradijenti uporedivi log-prostoru)
lin_loss_fn = nn.HuberLoss()      # linearni prostor: Huber na (nc/POP_SCALE) vs (pops/POP_SCALE)
LOSS_SPACE = "log"                # "log" (podrazumevano) ili "linear"; menja ga run(loss_space=...)
use_amp = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

def prodji(net, loader, treniraj, optim=None, freeze_bn=False):
    net.train(treniraj)
    if freeze_bn:                                   # faza 1: telo zamrznuto -> ne azuriraj BN statistiku
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
    tot, n, PRED, TRUE = 0.0, 0, [], []
    for tiles, nid, pops in loader:
        tiles, nid, pops = tiles.to(DEVICE, non_blocking=True), nid.to(DEVICE), pops.to(DEVICE)
        with torch.set_grad_enabled(treniraj), torch.autocast("cuda", enabled=use_amp):
            raw = net(tiles).squeeze(1)                       # [sumT]
        counts = F.softplus(raw.float())                        # fp32, nenegativno
        nc = torch.zeros(len(pops), device=DEVICE).scatter_add(0, nid, counts)  # suma po naselju (fp32)
        if LOSS_SPACE == "log":
            loss = loss_fn(torch.log1p(nc), torch.log1p(pops))
        else:  # linearni prostor: Huber u stanovnicima (skaliran radi stabilnih gradijenata)
            loss = lin_loss_fn(nc / POP_SCALE, pops / POP_SCALE)
        if treniraj:
            optim.zero_grad(); scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        tot += loss.item() * len(pops); n += len(pops)
        PRED.append(nc.detach().cpu().numpy()); TRUE.append(pops.detach().cpu().numpy())
    return tot / n, np.concatenate(PRED), np.concatenate(TRUE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trening (sa MLflow pracenjem)

# COMMAND ----------

# DBTITLE 1,Trening + evaluacija (5-struka CV)
def r2log(true, pred):
    return r2_score(np.log1p(true), np.log1p(np.clip(pred, 0, None)))

def treniraj_fold(train_nasel_frame, val_nasel_frame):
    """Istrenira jedan fold (glava pa fine-tuning) i vrati (best_state, best_val_r2, oof_pred_pop, net)."""
    train_dl, val_dl = napravi_loadere(train_nasel_frame, val_nasel_frame)
    net = timm.create_model("resnet18", pretrained=False, in_chans=CFG["num_bands"], num_classes=1).to(DEVICE)
    import math
    import torchvision.models as tv_models
    tv_sd = tv_models.resnet18(weights="DEFAULT").state_dict()
    w = tv_sd["conv1.weight"].float()
    rep = math.ceil(CFG["num_bands"] / 3)
    tv_sd["conv1.weight"] = w.repeat(1, rep, 1, 1)[:, :CFG["num_bands"], :, :] * (3 / CFG["num_bands"])
    net_sd = net.state_dict()
    tv_sd = {k: v for k, v in tv_sd.items() if k in net_sd and v.shape == net_sd[k].shape}
    net_sd.update(tv_sd)
    net.load_state_dict(net_sd)
    def epoha(opt, korak, freeze_bn=False):
        tl, _, _ = prodji(net, train_dl, True, opt, freeze_bn=freeze_bn)
        vl, P, Y = prodji(net, val_dl, False)
        r2 = r2log(Y, P)
        mlflow.log_metrics({"train_loss": tl, "val_loss": vl, "val_r2": r2}, step=korak)
        return r2

    best_r2, best_state = dvofazni_trening(
        net, epoha,
        CFG["epochs_head"], CFG["epochs_finetune"],
        CFG["head_lr"],     CFG["finetune_lr"],
    )

    net.load_state_dict(best_state)     # najbolja tezina po validaciji ovog folda
    _, P, _ = prodji(net, val_dl, False)
    oof_pred_pop = np.clip(P, 0, None)   # OOF predikcija (poravnata sa val_nasel_frame redosledom)
    return best_state, best_r2, oof_pred_pop, net


def run(loss_space="log"):
    """Puna k-struka GroupKFold CV (plocice + agregaciona loss).
    Roditeljski MLflow run + ugnjezdeni run po foldu. OOF predikcije (svako naselje u validaciji
    tacno jednom) se skupljaju i daju jedinstvenu procenu na celom skupu.

    loss_space: "log" (Huber na log1p) ili "linear" (Huber u prostoru stanovnika) — za poredjenje."""
    global LOSS_SPACE
    LOSS_SPACE = loss_space
    mlflow.set_experiment("/Users/korisnik/procena_stanovnika")
    oof = pd.Series(np.nan, index=nasel.naselje_maticni_broj.values, dtype="float32")
    fold_r2 = []
    loss_opis = "HuberLoss(log1p suma plocica)" if loss_space == "log" else "HuberLoss(suma plocica / POP_SCALE)"

    with mlflow.start_run(run_name=f"tiles-agregacija-{loss_space}-cv{len(FOLDS)}"):
        mlflow.log_params(CFG)
        mlflow.log_params({
            "pristup": "plocice_agregacija", "backbone": "resnet18", "pretrained": True,
            "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "loss": loss_opis,
            "loss_space": loss_space,
            "cv": f"GroupKFold(opstina) x{len(FOLDS)}",
            "n_naselja": len(nasel), "n_opstina": int(broj_opstina),
        })

        for fold, (train_nasel_frame, val_nasel_frame) in enumerate(FOLDS):
            with mlflow.start_run(run_name=f"tiles-agregacija-{loss_space}-fold{fold}", nested=True):
                mlflow.log_params({**CFG, "fold": fold, "loss_space": loss_space,
                                   "n_train": len(train_nasel_frame), "n_val": len(val_nasel_frame)})
                best_state, best_r2, oof_pred_pop, net = treniraj_fold(train_nasel_frame, val_nasel_frame)
                mlflow.log_metric("best_val_r2", best_r2)
                oof.loc[val_nasel_frame.naselje_maticni_broj.values] = oof_pred_pop
                put = f"/Volumes/katalog/deep_learning/raw_data/tiles_agregacija_{loss_space}_fold{fold}.pt"
                torch.save(best_state, put); mlflow.log_artifact(put)
                mlflow.pytorch.log_model(net, name=f"model_tiles_{loss_space}_fold{fold}")   # servabilan model artefakt
                fold_r2.append(best_r2)
                print(f"[fold {fold}] best val R2 {best_r2:.3f}")

        # === agregacija preko svih foldova (OOF: svako naselje predvidjeno tacno jednom) ===
        oof_pred = oof.loc[nasel.naselje_maticni_broj.values].values.astype("float32")
        stvarno  = nasel["pop"].values.astype("float32")
        agg = {
            "cv_mean_val_r2": float(np.mean(fold_r2)),
            "cv_std_val_r2":  float(np.std(fold_r2)),
            **oof_metrics(stvarno, oof_pred, nasel),
        }
        mlflow.log_metrics(agg)
        fig = cv_summary_figure(fold_r2, agg, stvarno, oof_pred, nasel, label=f"tiles-{loss_space}")
        plt.show()
        mlflow.log_figure(fig, f"cv_evaluacija_tiles_{loss_space}.png")

    print(f"[tiles-{loss_space}] CV R2 {agg['cv_mean_val_r2']:.3f} \u00b1 {agg['cv_std_val_r2']:.3f}"
          f" | OOF opstina R2 {agg['oof_opstina_r2']:.3f} | OOF MAE(st) {agg['oof_mae_stanovnici']:.0f}")
    return {"pristup": f"tiles_{loss_space}", **agg}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluacija

# COMMAND ----------

# DBTITLE 1,Pokreni CV
# puna k-struka GroupKFold CV (plocice + agregaciona loss)
_orig_log_model = mlflow.pytorch.log_model

def _log_model_pickle(*args, **kwargs):
    kwargs.setdefault("serialization_format", "pickle")
    return _orig_log_model(*args, **kwargs)

mlflow.pytorch.log_model = _log_model_pickle
try:
    rezultat = run()
finally:
    mlflow.pytorch.log_model = _orig_log_model

print("\n=== Rezultat (plocice + agregacija) ===")
display(pd.DataFrame([rezultat]).set_index("pristup"))

# COMMAND ----------

# DBTITLE 1,Poredjenje log vs linear loss
# === Poredjenje: linearni Huber (loss u stanovnicima) vs log-Huber baseline ===
# Isti hiperparametri i GroupKFold; jedina razlika je prostor u kome se racuna loss.
# Cilj: da li trening u linearnom prostoru popravlja agregaciju po opstini (linearni R2).
mlflow.pytorch.log_model = _log_model_pickle
try:
    rezultat_lin = run(loss_space="linear")
finally:
    mlflow.pytorch.log_model = _orig_log_model

poredjenje = (
    pd.DataFrame([rezultat, rezultat_lin])
    .set_index("pristup")[
        ["cv_mean_val_r2", "oof_r2_log", "oof_opstina_r2", "oof_opstina_r2_log",
         "oof_mae_stanovnici", "oof_rmse_stanovnici"]
    ]
)
print("\n=== Poredjenje log vs linear (agregaciona loss) ===")
display(poredjenje)

# COMMAND ----------

# === Poredjenje: linearni Huber (loss u stanovnicima) vs log-Huber baseline ===
# Isti hiperparametri i GroupKFold; jedina razlika je prostor u kome se racuna loss.
# Cilj: da li trening u linearnom prostoru popravlja agregaciju po opstini (linearni R2).
mlflow.pytorch.log_model = _log_model_pickle
try:
    rezultat_lin = run(loss_space="linear")
finally:
    mlflow.pytorch.log_model = _orig_log_model

poredjenje = (
    pd.DataFrame([rezultat, rezultat_lin])
    .set_index("pristup")[
        ["cv_mean_val_r2", "oof_r2_log", "oof_opstina_r2", "oof_opstina_r2_log",
         "oof_mae_stanovnici", "oof_rmse_stanovnici"]
    ]
)
print("\n=== Poredjenje log vs linear (agregaciona loss) ===")
display(poredjenje)

# COMMAND ----------

