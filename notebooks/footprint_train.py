# Databricks notebook source
# MAGIC %md
# MAGIC # Procena broja stanovnika iz otisaka zgrada (pristup 2)
# MAGIC
# MAGIC ResNet-18 (ImageNet) fine-tuning na rasterizovane otiske zgrada po naselju.
# MAGIC Ulaz su 2 kanala: pokrivenost (udeo celije pod zgradom) i zapreminska gustina (pokrivenost x spratnost).
# MAGIC Cilj je `log1p(broj_stanovnika)`. Podela po opstinama (GroupKFold). Eksperimenti kroz MLflow.

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
import timm, mlflow
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from procena import (
    seed_everything, prodji, dvofazni_trening,
    stats_po_opsegu, Naselja, napravi_loadere, NW,
    napravi_foldove, oof_metrics, cv_summary_figure,
    podesi_mlflow, izlazni_dir, sacuvaj_oof,
)

# Colab: "/content/fp_data" (raspakuje footprint_upload.zip). Databricks: UC Volume (isti "data" folder kao ostali pristupi).
DATA_DIR = "/content/fp_data" if os.path.isdir("/content") else "/Volumes/katalog/deep_learning/raw_data/data"
if os.path.isdir("/content") and not os.path.isdir(DATA_DIR + "/footprint_cutouts"):
    zipfile.ZipFile("/content/footprint_upload.zip").extractall(DATA_DIR)
CUT = DATA_DIR + "/footprint_cutouts"
OUT_DIR = izlazni_dir()   # tezine modela i OOF parquet (UC Volume ili /content/out)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# svi hiperparametri na jednom mestu (jedini izvor istine)
CFG = {
    "num_bands": 2,                 # 2 kanala: pokrivenost, zapreminska gustina
    "img_px": 224,
    "epochs_head": 3,
    "epochs_finetune": 50,
    "batch_size": 64,
    "head_lr": 1e-3,
    "finetune_lr": 3e-4,
    "seed": 42,                     # za reproduktivnost (random/numpy/torch/cuda + DataLoader radnici)
}
seed_everything(CFG["seed"])
print("device:", DEVICE, "| footprint cutouts:", len(os.listdir(CUT)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Podaci i podela

# COMMAND ----------

# DBTITLE 1,Ucitavanje podataka / podesavanje CV
labele = pd.read_parquet(DATA_DIR + "/naselje_table.parquet")[
    ["naselje_maticni_broj", "opstina_maticni_broj", "pop"]]
df = pd.DataFrame({"path": glob.glob(CUT + "/*.npy")})
df["naselje_maticni_broj"] = df.path.map(lambda f: int(os.path.splitext(os.path.basename(f))[0]))
df = df.merge(labele, on="naselje_maticni_broj", how="inner")
df["y"] = np.log1p(df["pop"]).astype("float32")

FOLDS = napravi_foldove(df)
N_FOLDS = len(FOLDS)
broj_opstina = df["opstina_maticni_broj"].nunique()
print(f"uzoraka {len(df)} | opstina {broj_opstina} | foldova {N_FOLDS}")
for i, (t, v) in enumerate(FOLDS):
    print(f"  fold {i}: trening {len(t)} / val {len(v)} naselja ({v['opstina_maticni_broj'].nunique()} opstina)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model

# COMMAND ----------

# DBTITLE 1,Model
# timm prilagodjava prvi konvolucioni sloj na 2 kanala (in_chans); model se pravi po foldu u treniraj_fold
# prodji i dvofazni_trening su uvezeni iz procena.train
loss_fn = nn.HuberLoss()
use_amp = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trening (sa MLflow pracenjem)

# COMMAND ----------

# DBTITLE 1,Trening + evaluacija (5-struka CV)
def treniraj_fold(train_frame, val_frame):
    """Istrenira jedan fold i vrati (best_state, best_val_r2, oof_pred_pop, net)."""
    train_dl, val_dl = napravi_loadere(train_frame, val_frame, CFG["batch_size"], CFG["seed"])
    net = timm.create_model("resnet18", pretrained=True, in_chans=CFG["num_bands"], num_classes=1).to(DEVICE)

    def epoha(opt, korak, freeze_bn=False):
        tl, _, _ = prodji(net, train_dl, True, loss_fn, opt, scaler=scaler,
                          use_amp=use_amp, device=DEVICE, freeze_bn=freeze_bn)
        vl, P, Y = prodji(net, val_dl, False, loss_fn, device=DEVICE)
        r2 = r2_score(Y, P)
        mlflow.log_metrics({"train_loss": tl, "val_loss": vl, "val_r2": r2}, step=korak)
        return r2

    best_r2, best_state = dvofazni_trening(
        net, epoha,
        CFG["epochs_head"], CFG["epochs_finetune"],
        CFG["head_lr"],     CFG["finetune_lr"],
    )
    net.load_state_dict(best_state)     # najbolja tezina po validaciji ovog folda
    _, P, _ = prodji(net, val_dl, False, loss_fn, device=DEVICE)
    oof_pred_pop = np.expm1(P)          # OOF predikcija u populaciji
    return best_state, best_r2, oof_pred_pop, net

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluacija

# COMMAND ----------

# DBTITLE 1,Pokreni CV
def run():
    """Puna k-struka GroupKFold CV (footprint pristup)."""
    podesi_mlflow()   # Databricks workspace, ili Colab -> Databricks preko env varijabli, ili lokalni mlruns
    oof = pd.Series(np.nan, index=df.naselje_maticni_broj.values, dtype="float32")
    fold_r2 = []

    with mlflow.start_run(run_name=f"footprint-cv{len(FOLDS)}"):
        mlflow.log_params(CFG)
        mlflow.log_params({
            "pristup": "footprint",
            "backbone": "resnet18",
            "pretrained": True,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": "HuberLoss",
            "cv": f"GroupKFold(opstina) x{len(FOLDS)}",
            "n_uzoraka": len(df),
            "n_opstina": int(broj_opstina),
        })

        for fold, (train_frame, val_frame) in enumerate(FOLDS):
            with mlflow.start_run(run_name=f"footprint-fold{fold}", nested=True):
                mlflow.log_params({
                    **CFG,
                    "fold": fold,
                    "n_train": len(train_frame),
                    "n_val": len(val_frame)
                })
                best_state, best_r2, oof_pred_pop, net = treniraj_fold(train_frame, val_frame)
                mlflow.log_metric("best_val_r2", best_r2)
                oof.loc[val_frame.naselje_maticni_broj.values] = oof_pred_pop
                put = f"{OUT_DIR}/footprint_fold{fold}.pt"
                torch.save(best_state, put); mlflow.log_artifact(put)
                mlflow.pytorch.log_model(
                    net,
                    name=f"model_footprint_fold{fold}",
                    serialization_format="pickle"
                )
                fold_r2.append(best_r2)
                print(f"[fold {fold}] best val R2 {best_r2:.3f}")

        oof_pred = oof.loc[df.naselje_maticni_broj.values].values.astype("float32")
        stvarno  = df["pop"].values.astype("float32")
        agg = {
            "cv_mean_val_r2": float(np.mean(fold_r2)),
            "cv_std_val_r2":  float(np.std(fold_r2)),
            **oof_metrics(stvarno, oof_pred, df),
        }
        mlflow.log_metrics(agg)
        put_oof = sacuvaj_oof(df, oof_pred, "footprint", OUT_DIR)   # ulaz za fuziju
        mlflow.log_artifact(put_oof)
        fig = cv_summary_figure(fold_r2, agg, stvarno, oof_pred, df, label="footprint")
        plt.show()
        mlflow.log_figure(fig, "cv_evaluacija_footprint.png")

    print(f"[footprint] CV R2 {agg['cv_mean_val_r2']:.3f} \u00b1 {agg['cv_std_val_r2']:.3f}"
          f" | OOF R2(log) {agg['oof_r2_log']:.3f} | medAPE {agg.get('oof_medape', float('nan')):.2f}"
          f" | wMAPE {agg.get('oof_wmape', float('nan')):.2f} | bias {agg.get('oof_bias', float('nan')):.2f}"
          f" | opstina R2(log, bez top2) {agg.get('oof_opstina_r2_log_bez_top2', float('nan')):.3f}")
    return {"pristup": "footprint", **agg}


rezultat = run()
print("\n=== Rezultat (footprint) ===")
display(pd.DataFrame([rezultat]).set_index("pristup"))

# COMMAND ----------

