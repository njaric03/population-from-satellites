# Databricks notebook source
# MAGIC %md
# MAGIC # Procena broja stanovnika iz satelitskih snimaka
# MAGIC
# MAGIC ResNet-18 pretreniran na ImageNet-u, fine-tuning na Sentinel-2 isečke (6 opsega) po naselju.
# MAGIC Cilj je `log1p(broj_stanovnika)` ili `log1p(gustina_naseljenosti)` u zavisnosti od postavke. Ocena ide punom 5-strukom GroupKFold CV po opštinama (svako naselje u validaciji tačno jednom),
# MAGIC da susedna naselja ne cure između skupova.
# MAGIC
# MAGIC Ciljna veličina je hiperparametar (`CFG["targets"]` u prvoj ćeliji): bez dupliranja koda trenira se i poredi `log1p(pop)` (baseline) i `log1p(gustina)` (populacija = gustina × površina).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Priprema

# COMMAND ----------

# DBTITLE 1,Postavke hiperparametara
import sys
sys.path.insert(0, "/Workspace/Users/korisnik/du-procena-stanovnistva/src")

import os, glob, zipfile
import numpy as np, pandas as pd
import torch, torch.nn as nn
try:
    import timm
except ImportError:
    os.system("pip -q install timm"); import timm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import mlflow
from procena import (
    seed_everything, prodji, dvofazni_trening,
    stats_po_opsegu, Naselja, NW, seed_worker,
    napravi_loadere as _base_napravi_loadere,   # alias – ovaj notebook ima sopstveni wrapper
    napravi_foldove, oof_metrics, cv_summary_figure,
)

# podaci su vec raspakovani u Volume 
BASE = "/Volumes/katalog/deep_learning/raw_data/data"
CUT = BASE + "/cutouts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# svi hiperparametri na jednom mestu (jedini izvor istine)
CFG = {
    "num_bands": 6,                 # broj Sentinel-2 opsega / ulaznih kanala
    "img_px": 224,                  # dimenzija isecka u pikselima
    "targets": ["pop", "density"],  # ciljne velicine: trenira obe i poredi u MLflow-u (bez dupliranja koda)
    "epochs_head": 3,               # epohe za fazu ucenja glave
    "epochs_finetune": 50,          # epohe za fine-tuning
    "batch_size": 64,
    "head_lr": 1e-3,                # learning rate za glavu
    "finetune_lr": 3e-4,            # learning rate za fine-tuning
    "seed": 42,                     # za reproduktivnost (random/numpy/torch/cuda + DataLoader radnici)
}
seed_everything(CFG["seed"])
print("device:", DEVICE, "| cutouts:", len(os.listdir(CUT)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Podaci i podela

# COMMAND ----------

# DBTITLE 1,Ucitavanje podataka / podesavanje CV
# ime fajla isečka = matični broj naselja; spaja se sa tabelom labela
labele = pd.read_parquet(BASE + "/naselje_table.parquet")[
    ["naselje_maticni_broj", "opstina_maticni_broj", "pop", "area_km2"]
]
df = pd.DataFrame({"path": glob.glob(CUT + "/*.npy")})
df["naselje_maticni_broj"] = df.path.map(lambda f: int(os.path.splitext(os.path.basename(f))[0]))
df = df.merge(labele, on="naselje_maticni_broj", how="inner")
df["area_km2"] = df["area_km2"].fillna(df["area_km2"].median())
df["gustina"] = df["pop"] / df["area_km2"]              # stanovnika po km2 (za density cilj)

FOLDS = napravi_foldove(df)
N_FOLDS = len(FOLDS)
broj_opstina = df["opstina_maticni_broj"].nunique()
print(f"uzoraka {len(df)} | opstina {broj_opstina} | foldova {N_FOLDS}")
for i, (t, v) in enumerate(FOLDS):
    print(f"  fold {i}: trening {len(t)} / val {len(v)} naselja ({v["opstina_maticni_broj"].nunique()} opstina)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalizacija i dataset

# COMMAND ----------

# DBTITLE 1,Ciljna velicina i DataLoader
# stats_po_opsegu, Naselja, NW, seed_worker, _base_napravi_loadere su uvezeni iz procena.

# === ciljna velicina kao hiperparametar: kako se racuna y i kako se predikcija vraca u populaciju ===
TARGET_SPECS = {
    "pop": {
        "run_name": "resnet18-baseline",
        "target": "log1p(pop)",
        "rekonstrukcija": "expm1(pred)",
        "y": lambda f: np.log1p(f["pop"]).astype("float32"),
        "u_populaciju": lambda P, f: np.clip(np.expm1(P), 0, None),
    },
    "density": {
        "run_name": "resnet18-density",
        "target": "log1p(pop/area_km2)",
        "rekonstrukcija": "expm1(pred)*area_km2",
        "y": lambda f: np.log1p(f["pop"] / f["area_km2"]).astype("float32"),
        "u_populaciju": lambda P, f: np.clip(np.expm1(P) * f["area_km2"].values, 0, None),
    },
}

def napravi_loadere(target, train_frame, val_frame):
    """Wrapper: dodaje y kolonu za dati target, pa poziva zajednicki _base_napravi_loadere."""
    spec = TARGET_SPECS[target]
    tdf = train_frame.copy(); tdf["y"] = spec["y"](tdf)
    vdf = val_frame.copy();   vdf["y"] = spec["y"](vdf)
    tdl, vdl = _base_napravi_loadere(tdf, vdf, CFG["batch_size"], CFG["seed"])
    return tdl, vdl, tdf, vdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model

# COMMAND ----------

# DBTITLE 1,AMP i trening pomocnici
# timm prosiruje prvi konvolucioni sloj sa 3 na 6 kanala (in_chans); model se pravi po cilju u run_target
# prodji i dvofazni_trening su uvezeni iz procena.train
loss_fn = nn.HuberLoss()

# GPU ubrzanja na A10 (Ampere): TF32 matmul/cudnn + autodetekcija najbrzih kernela
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# mesovita preciznost: bf16 na A10 ne zahteva GradScaler (isti opseg eksponenta kao fp32)
use_amp  = DEVICE == "cuda"
amp_dtype = torch.bfloat16
scaler   = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trening

# COMMAND ----------

# DBTITLE 1,Trening + evaluacija (5-struka CV)
def treniraj_fold(target, train_frame, val_frame):
    """Istrenira jedan fold (glava pa fine-tuning) i vrati (best_state, best_val_r2, oof_pred_pop).
    R2 se racuna u log1p(pop) prostoru radi uporedivosti pop vs density. Normalizacija je po foldu."""
    spec = TARGET_SPECS[target]
    train_dl, val_dl, _, vfr = napravi_loadere(target, train_frame, val_frame)
    net = timm.create_model("resnet18", pretrained=True, in_chans=CFG["num_bands"], num_classes=1).to(DEVICE)
    ylog_val = np.log1p(vfr["pop"].values).astype("float32")   # poredbeni prostor (populacija)
    def epoha(opt, korak, freeze_bn=False):
        tl, _, _ = prodji(net, train_dl, True, loss_fn, opt, scaler=scaler,
                          use_amp=use_amp, amp_dtype=amp_dtype, device=DEVICE, freeze_bn=freeze_bn)
        vl, P, _ = prodji(net, val_dl, False, loss_fn, device=DEVICE)
        r2 = r2_score(ylog_val, np.log1p(spec["u_populaciju"](P, vfr)))
        mlflow.log_metrics({"train_loss": tl, "val_loss": vl, "val_r2": r2}, step=korak)
        return r2

    best_r2, best_state = dvofazni_trening(
        net, epoha,
        CFG["epochs_head"], CFG["epochs_finetune"],
        CFG["head_lr"],     CFG["finetune_lr"],
    )
    net.load_state_dict(best_state)     # najbolja tezina po validaciji ovog folda
    _, P, _ = prodji(net, val_dl, False, loss_fn, device=DEVICE)
    oof_pred_pop = spec["u_populaciju"](P, vfr)   # OOF predikcija u prostoru populacije
    return best_state, best_r2, oof_pred_pop, net


def run_target(target):
    """Puna k-struka GroupKFold CV za jednu ciljnu velicinu.
    Roditeljski MLflow run + ugnjezdeni run po foldu. OOF predikcije (svako naselje u validaciji
    tacno jednom) se skupljaju i daju jedinstvenu procenu na celom skupu."""
    spec = TARGET_SPECS[target]
    mlflow.set_experiment("/Users/korisnik/procena_stanovnika")
    oof = pd.Series(np.nan, index=df.naselje_maticni_broj.values, dtype="float32")  # OOF po naselju
    fold_r2 = []

    with mlflow.start_run(run_name=f"{spec['run_name']}-cv{len(FOLDS)}"):
        mlflow.log_params(CFG)      # svi hiperparametri odjednom iz config dict-a
        mlflow.log_params({
            "backbone": "resnet18", "pretrained": True,
            "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "loss": "HuberLoss",
            "target": spec["target"], "rekonstrukcija": spec["rekonstrukcija"],
            "cv": f"GroupKFold(opstina) x{len(FOLDS)}",
            "n_uzoraka": len(df), "n_opstina": int(broj_opstina),
        })

        for fold, (train_frame, val_frame) in enumerate(FOLDS):
            with mlflow.start_run(run_name=f"{spec['run_name']}-fold{fold}", nested=True):
                mlflow.log_params({**CFG, "fold": fold,
                                   "n_train": len(train_frame), "n_val": len(val_frame)})
                best_state, best_r2, oof_pred_pop, net = treniraj_fold(target, train_frame, val_frame)
                mlflow.log_metric("best_val_r2", best_r2)
                oof.loc[val_frame.naselje_maticni_broj.values] = oof_pred_pop   # upisi OOF za ovaj fold
                put = f"/Volumes/katalog/deep_learning/raw_data/resnet18_{target}_fold{fold}.pt"
                torch.save(best_state, put); mlflow.log_artifact(put)
                mlflow.pytorch.log_model(net, name=f"model_{target}_fold{fold}")   # servabilan model artefakt
                fold_r2.append(best_r2)
                print(f"[{target} fold {fold}] best val R2 {best_r2:.3f}")

        # === agregacija preko svih foldova (OOF: svako naselje predvidjeno tacno jednom) ===
        oof_pred = oof.loc[df.naselje_maticni_broj.values].values.astype("float32")
        stvarno  = df["pop"].values.astype("float32")
        agg = {
            "cv_mean_val_r2": float(np.mean(fold_r2)),
            "cv_std_val_r2":  float(np.std(fold_r2)),
            **oof_metrics(stvarno, oof_pred, df),
        }
        mlflow.log_metrics(agg)
        fig = cv_summary_figure(fold_r2, agg, stvarno, oof_pred, df, label=target)
        plt.show()
        mlflow.log_figure(fig, f"cv_evaluacija_{target}.png")

    print(f"[{target}] CV R2 {agg['cv_mean_val_r2']:.3f} ± {agg['cv_std_val_r2']:.3f}"
          f" | OOF opstina R2 {agg['oof_opstina_r2']:.3f} | OOF MAE(st) {agg['oof_mae_stanovnici']:.0f}")
    return {"target": target, **agg}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluacija

# COMMAND ----------

# DBTITLE 1,Pokreni sve ciljeve i uporedi
# puna 5-struka GroupKFold CV za svaku ciljnu velicinu iz CFG["targets"] (svaka = roditeljski MLflow run + ugnjezdeni run po foldu) i poredi rezultate
rezultati = [run_target(t) for t in CFG["targets"]]

poredjenje = pd.DataFrame(rezultati).set_index("target")
print("\n=== Poredjenje ciljnih velicina (validacija) ===")
display(poredjenje)