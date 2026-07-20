# Databricks notebook source
# MAGIC %md
# MAGIC # Multimodalni model: Sentinel-2 + otisci zgrada (zajednicki trening)
# MAGIC
# MAGIC Fuzija dogovorena u prepisci sa profesorom: jedna grana obradjuje satelitski
# MAGIC snimak, druga rasterizovane otiske zgrada, a njihove reprezentacije se spajaju
# MAGIC u zajednicku procenu. Dva ResNet-18 trupa (ImageNet, 6 odnosno 2 ulazna kanala)
# MAGIC daju po 512-dim embedding; konkatenacija ide u zajednicku regresionu glavu.
# MAGIC Cilj je `log1p(broj_stanovnika)`; trening end-to-end (glava pa fine-tuning oba
# MAGIC trupa). Ista GroupKFold podela po opstinama i isti OOF protokol kao u ostalim
# MAGIC notebucima, pa su rezultati direktno uporedivi (i sa stacking fuzijom).

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
import sys, os
# src modul: Databricks Workspace ili klon repozitorijuma na Colabu (/content)
for _src in ("/Workspace/Users/korisnik/du-procena-stanovnistva/src",
             "/content/du-procena-stanovnistva/src"):
    if os.path.isdir(_src):
        sys.path.insert(0, _src); break

import glob, zipfile
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm, mlflow
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from procena import (
    seed_everything, dvofazni_trening,
    stats_po_opsegu, NW, seed_worker,
    napravi_foldove, oof_metrics, cv_summary_figure,
    podesi_mlflow, izlazni_dir, sacuvaj_oof,
)

# Colab: "/content/data" (raspakuje oba zip-a). Databricks: UC Volume (isti "data" folder kao ostali pristupi).
BASE = "/content/data" if os.path.isdir("/content") else "/Volumes/katalog/deep_learning/raw_data/data"
if os.path.isdir("/content"):
    if not os.path.isdir(BASE + "/cutouts"):
        zipfile.ZipFile("/content/data_upload.zip").extractall(BASE)
    if not os.path.isdir(BASE + "/footprint_cutouts"):
        zipfile.ZipFile("/content/footprint_upload.zip").extractall(BASE)
CUT_SAT = BASE + "/cutouts"
CUT_FP  = BASE + "/footprint_cutouts"
OUT_DIR = izlazni_dir()   # tezine modela i OOF parquet (UC Volume ili /content/out)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# svi hiperparametri na jednom mestu (jedini izvor istine)
CFG = {
    "sat_bands": 6,                 # Sentinel-2 opsega (satelitska grana)
    "fp_bands": 2,                  # kanali otisaka: pokrivenost, zapreminska gustina
    "img_px": 224,
    "embed_hidden": 256,            # skriveni sloj zajednicke glave (1024 -> 256 -> 1)
    "dropout": 0.2,
    "epochs_head": 3,
    "epochs_finetune": 40,
    "batch_size": 48,               # dva ResNet-18 trupa -> nesto manji batch nego kod jednog
    "head_lr": 1e-3,
    "finetune_lr": 3e-4,
    "seed": 42,                     # za reproduktivnost (random/numpy/torch/cuda + DataLoader radnici)
}
seed_everything(CFG["seed"])
print("device:", DEVICE, "| sat cutouts:", len(os.listdir(CUT_SAT)), "| footprint cutouts:", len(os.listdir(CUT_FP)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Podaci i podela (presek naselja sa oba ulaza)

# COMMAND ----------

# DBTITLE 1,Ucitavanje podataka / podesavanje CV
labele = pd.read_parquet(BASE + "/naselje_table.parquet")[
    ["naselje_maticni_broj", "opstina_maticni_broj", "pop"]]

def putanje(folder):
    t = pd.DataFrame({"path": glob.glob(folder + "/*.npy")})
    t["naselje_maticni_broj"] = t.path.map(lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    return t

sat = putanje(CUT_SAT).rename(columns={"path": "path_sat"})
fp  = putanje(CUT_FP).rename(columns={"path": "path_fp"})
df = (
    sat.merge(fp, on="naselje_maticni_broj", how="inner")   # samo naselja sa OBA ulaza
    .merge(labele, on="naselje_maticni_broj", how="inner")
)
df["y"] = np.log1p(df["pop"]).astype("float32")
print(f"sat {len(sat)} | footprint {len(fp)} | presek (oba ulaza + labela) {len(df)}")

FOLDS = napravi_foldove(df)
N_FOLDS = len(FOLDS)
broj_opstina = df["opstina_maticni_broj"].nunique()
print(f"uzoraka {len(df)} | opstina {broj_opstina} | foldova {N_FOLDS}")
for i, (t, v) in enumerate(FOLDS):
    print(f"  fold {i}: trening {len(t)} / val {len(v)} naselja ({v['opstina_maticni_broj'].nunique()} opstina)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalizacija i dataset

# COMMAND ----------

# DBTITLE 1,Dataset sa dva ulaza
# stats_po_opsegu, NW, seed_worker su uvezeni iz procena.data;
# NaseljaMM je multimodal-specifican (dva .npy ulaza, ista prostorna augmentacija na oba).

class NaseljaMM(Dataset):
    """Par (satelitski cutout, footprint raster) istog naselja + log1p(pop).

    Augmentacija (flip/rotacija) se izvlaci jednom i primenjuje identicno na
    oba ulaza — rasteri su prostorno poravnati pa transformacije moraju biti iste.
    """

    def __init__(self, frame, mean_s, std_s, mean_f, std_f, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.mean_s, self.std_s = mean_s, std_s
        self.mean_f, self.std_f = mean_f, std_f
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        red = self.frame.iloc[i]
        xs = (np.load(red.path_sat).astype("float32") - self.mean_s[0]) / self.std_s[0]
        xf = (np.load(red.path_fp).astype("float32") - self.mean_f[0]) / self.std_f[0]
        if self.augment:
            if np.random.rand() < 0.5:
                xs = xs[:, :, ::-1]; xf = xf[:, :, ::-1]
            if np.random.rand() < 0.5:
                xs = xs[:, ::-1, :]; xf = xf[:, ::-1, :]
            k = np.random.randint(4)
            xs = np.rot90(xs, k, axes=(1, 2)); xf = np.rot90(xf, k, axes=(1, 2))
        return (
            torch.from_numpy(np.ascontiguousarray(xs)),
            torch.from_numpy(np.ascontiguousarray(xf)),
            torch.tensor([red.y], dtype=torch.float32),
        )

def napravi_loadere_mm(train_frame, val_frame):
    """Vrati (train_dl, val_dl); normalizacija po modalitetu iz trening skupa ovog folda."""
    mean_s, std_s = stats_po_opsegu(train_frame.path_sat.tolist())
    mean_f, std_f = stats_po_opsegu(train_frame.path_fp.tolist())
    gen = torch.Generator().manual_seed(CFG["seed"])
    _sw = (lambda wid: seed_worker(wid, CFG["seed"])) if NW else None
    tdl = DataLoader(NaseljaMM(train_frame, mean_s, std_s, mean_f, std_f, augment=True),
                     batch_size=CFG["batch_size"], shuffle=True,
                     generator=gen, worker_init_fn=_sw,
                     num_workers=NW, pin_memory=True, persistent_workers=NW > 0,
                     prefetch_factor=4 if NW else None)
    vdl = DataLoader(NaseljaMM(val_frame, mean_s, std_s, mean_f, std_f),
                     batch_size=CFG["batch_size"],
                     num_workers=NW, pin_memory=True, persistent_workers=NW > 0,
                     prefetch_factor=4 if NW else None)
    return tdl, vdl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model (dve grane + zajednicka glava)

# COMMAND ----------

# DBTITLE 1,Multimodalni model
class MultimodalniModel(nn.Module):
    """Dva ResNet-18 trupa (num_classes=0 -> 512-dim embedding po grani);
    konkatenacija embeddinga ide u zajednicku regresionu glavu.

    Parametri glave pocinju sa "head" — dvofazni_trening(head_prefix="head")
    u fazi 1 trenira samo nju, oba trupa zamrznuta."""

    def __init__(self):
        super().__init__()
        self.sat = timm.create_model("resnet18", pretrained=True,
                                     in_chans=CFG["sat_bands"], num_classes=0)
        self.fp  = timm.create_model("resnet18", pretrained=True,
                                     in_chans=CFG["fp_bands"], num_classes=0)
        d = self.sat.num_features + self.fp.num_features
        self.head = nn.Sequential(
            nn.Linear(d, CFG["embed_hidden"]),
            nn.ReLU(),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(CFG["embed_hidden"], 1),
        )

    def forward(self, xs, xf):
        return self.head(torch.cat([self.sat(xs), self.fp(xf)], dim=1))


loss_fn = nn.HuberLoss()
use_amp = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

def prodji_mm(net, loader, treniraj, optim=None, freeze_bn=False):
    """Jedan prolaz; kao procena.train.prodji ali sa dva ulaza po uzorku."""
    net.train(treniraj)
    if freeze_bn:                                   # faza 1: trupovi zamrznuti -> ne azuriraj BN statistiku
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
    ukupno, P, Y = 0.0, [], []
    for xs, xf, y in loader:
        xs = xs.to(DEVICE, non_blocking=True)
        xf = xf.to(DEVICE, non_blocking=True)
        y  = y.to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(treniraj), torch.autocast("cuda", enabled=use_amp):
            out  = net(xs, xf)
            loss = loss_fn(out, y)
        if treniraj:
            optim.zero_grad(); scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        ukupno += loss.item() * len(y)
        P.append(out.detach().float().cpu().numpy()); Y.append(y.cpu().numpy())
    return ukupno / len(loader.dataset), np.concatenate(P).ravel(), np.concatenate(Y).ravel()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trening (sa MLflow pracenjem)

# COMMAND ----------

# DBTITLE 1,Trening + evaluacija (5-struka CV)
def treniraj_fold(train_frame, val_frame):
    """Istrenira jedan fold (glava pa fine-tuning oba trupa) i vrati
    (best_state, best_val_r2, oof_pred_pop, net)."""
    train_dl, val_dl = napravi_loadere_mm(train_frame, val_frame)
    net = MultimodalniModel().to(DEVICE)

    def epoha(opt, korak, freeze_bn=False):
        tl, _, _ = prodji_mm(net, train_dl, True, opt, freeze_bn=freeze_bn)
        vl, P, Y = prodji_mm(net, val_dl, False)
        r2 = r2_score(Y, P)
        mlflow.log_metrics({"train_loss": tl, "val_loss": vl, "val_r2": r2}, step=korak)
        return r2

    best_r2, best_state = dvofazni_trening(
        net, epoha,
        CFG["epochs_head"], CFG["epochs_finetune"],
        CFG["head_lr"],     CFG["finetune_lr"],
        head_prefix="head",
    )
    net.load_state_dict(best_state)     # najbolja tezina po validaciji ovog folda
    _, P, _ = prodji_mm(net, val_dl, False)
    oof_pred_pop = np.clip(np.expm1(P), 0, None)   # OOF predikcija u populaciji
    return best_state, best_r2, oof_pred_pop, net

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluacija

# COMMAND ----------

# DBTITLE 1,Pokreni CV
def run():
    """Puna k-struka GroupKFold CV (multimodalni pristup)."""
    podesi_mlflow()   # Databricks workspace, ili Colab -> Databricks preko env varijabli, ili lokalni mlruns
    oof = pd.Series(np.nan, index=df.naselje_maticni_broj.values, dtype="float32")
    fold_r2 = []

    with mlflow.start_run(run_name=f"multimodal-cv{len(FOLDS)}"):
        mlflow.log_params(CFG)
        mlflow.log_params({
            "pristup": "multimodal",
            "backbone": "2x resnet18 (sat 6ch + footprint 2ch), concat 1024 -> head",
            "pretrained": True,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": "HuberLoss",
            "target": "log1p(pop)",
            "cv": f"GroupKFold(opstina) x{len(FOLDS)}",
            "n_uzoraka": len(df),
            "n_opstina": int(broj_opstina),
        })

        for fold, (train_frame, val_frame) in enumerate(FOLDS):
            with mlflow.start_run(run_name=f"multimodal-fold{fold}", nested=True):
                mlflow.log_params({**CFG, "fold": fold,
                                   "n_train": len(train_frame), "n_val": len(val_frame)})
                best_state, best_r2, oof_pred_pop, net = treniraj_fold(train_frame, val_frame)
                mlflow.log_metric("best_val_r2", best_r2)
                oof.loc[val_frame.naselje_maticni_broj.values] = oof_pred_pop
                put = f"{OUT_DIR}/multimodal_fold{fold}.pt"
                torch.save(best_state, put); mlflow.log_artifact(put)
                mlflow.pytorch.log_model(
                    net,
                    name=f"model_multimodal_fold{fold}",
                    serialization_format="pickle"
                )
                fold_r2.append(best_r2)
                print(f"[fold {fold}] best val R2 {best_r2:.3f}")

        # === agregacija preko svih foldova (OOF: svako naselje predvidjeno tacno jednom) ===
        oof_pred = oof.loc[df.naselje_maticni_broj.values].values.astype("float32")
        stvarno  = df["pop"].values.astype("float32")
        agg = {
            "cv_mean_val_r2": float(np.mean(fold_r2)),
            "cv_std_val_r2":  float(np.std(fold_r2)),
            **oof_metrics(stvarno, oof_pred, df),
        }
        mlflow.log_metrics(agg)
        put_oof = sacuvaj_oof(df, oof_pred, "multimodal", OUT_DIR)   # i ulaz za stacking poredjenje
        mlflow.log_artifact(put_oof)
        fig = cv_summary_figure(fold_r2, agg, stvarno, oof_pred, df, label="multimodal")
        plt.show()
        mlflow.log_figure(fig, "cv_evaluacija_multimodal.png")

    print(f"[multimodal] CV R2 {agg['cv_mean_val_r2']:.3f} ± {agg['cv_std_val_r2']:.3f}"
          f" | OOF R2(log) {agg['oof_r2_log']:.3f} | medAPE {agg.get('oof_medape', float('nan')):.2f}"
          f" | wMAPE {agg.get('oof_wmape', float('nan')):.2f} | bias {agg.get('oof_bias', float('nan')):.2f}"
          f" | opstina R2(log, bez top2) {agg.get('oof_opstina_r2_log_bez_top2', float('nan')):.3f}")
    return {"pristup": "multimodal", **agg}


rezultat = run()
print("\n=== Rezultat (multimodalni model) ===")
display(pd.DataFrame([rezultat]).set_index("pristup"))
