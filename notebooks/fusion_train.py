# Databricks notebook source
# MAGIC %md
# MAGIC # Fuzija pristupa (stacking nad OOF predikcijama)
# MAGIC
# MAGIC Treci deo projekta: spajanje satelitskog (tiles / sentinel) i footprint pristupa.
# MAGIC Meta-model (Ridge regresija u log prostoru) uci kako da kombinuje OOF predikcije
# MAGIC osnovnih modela. Posteno je jer je svaka OOF predikcija napravljena modelom koji
# MAGIC celu opstinu tog naselja nije video u treningu, a meta-model se ocenjuje istom
# MAGIC GroupKFold podelom po opstinama (fituje se na trening opstinama, ocenjuje na
# MAGIC validacionim).
# MAGIC
# MAGIC Usput i **post-hoc kalibracija po pristupu**: stacking sa jednim ulazom je tacno
# MAGIC regresija nagib+presek na log-log skali, sto ispravlja sistematsku kompresiju
# MAGIC predikcija ka sredini (dijagnostikovanu preko `oof_kalib_nagib` metrike).
# MAGIC Na kraju tabela: sirovo vs kalibrisano po pristupu vs fuzija.
# MAGIC
# MAGIC Ulaz su `oof_<pristup>.parquet` fajlovi koje treniraci notebooki snimaju u
# MAGIC OUT_DIR; ovaj notebook ne trenira mreze i ne zahteva GPU.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instalacija

# COMMAND ----------

# MAGIC %pip install -q mlflow scikit-learn
# MAGIC try:
# MAGIC     dbutils.library.restartPython()
# MAGIC except NameError:
# MAGIC     pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Konfiguracija

# COMMAND ----------

# DBTITLE 1,Postavke
import sys, os
# src modul: Databricks Workspace ili klon repozitorijuma na Colabu (/content)
for _src in ("/Workspace/Users/korisnik/du-procena-stanovnistva/src",
             "/content/du-procena-stanovnistva/src"):
    if os.path.isdir(_src):
        sys.path.insert(0, _src); break

import numpy as np, pandas as pd
import mlflow
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from procena import (
    seed_everything,
    napravi_foldove, oof_metrics, cv_summary_figure,
    podesi_mlflow, izlazni_dir, sacuvaj_oof,
)

OUT_DIR = izlazni_dir()   # tu su oof_<pristup>.parquet iz treniracih notebooka
# svi hiperparametri na jednom mestu (jedini izvor istine)
CFG = {
    "alpha": 1.0,     # Ridge regularizacija (mala; ulaza je 1-5, uzoraka ~4.6k)
    "seed": 42,
}
seed_everything(CFG["seed"])

# kandidati za ulaz u fuziju: koristi se svaki za koji postoji OOF parquet
KANDIDATI = ["tiles_log", "tiles_linear", "footprint", "sentinel_pop", "sentinel_density"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ucitavanje OOF predikcija osnovnih modela

# COMMAND ----------

# DBTITLE 1,Ucitavanje i spajanje OOF parqueta
osnove = {}
for ime in KANDIDATI:
    put = f"{OUT_DIR}/oof_{ime}.parquet"
    if os.path.exists(put):
        osnove[ime] = pd.read_parquet(put)
        print(f"  {ime:18s} {len(osnove[ime])} naselja")
    else:
        print(f"  {ime:18s} nema ({put})")
PRISTUPI = list(osnove)
assert len(PRISTUPI) >= 2, "za fuziju trebaju bar dva OOF parqueta (pokreni trenirace notebooke)"

# spajanje na preseku naselja; pop i opstina iz prvog (identicni su u svim fajlovima)
prvi = PRISTUPI[0]
df = osnove[prvi].rename(columns={"pred": f"pred_{prvi}"})
for ime in PRISTUPI[1:]:
    t = osnove[ime].rename(columns={"pred": f"pred_{ime}"})
    df = df.merge(t[["naselje_maticni_broj", f"pred_{ime}"]], on="naselje_maticni_broj", how="inner")
print(f"presek: {len(df)} naselja | pristupi: {PRISTUPI}")

# ulazi meta-modela su log1p predikcija (isti prostor kao cilj), cilj log1p(pop)
for ime in PRISTUPI:
    df[f"x_{ime}"] = np.log1p(df[f"pred_{ime}"].clip(lower=0)).astype("float32")
df["ylog"] = np.log1p(df["pop"]).astype("float32")

FOLDS = napravi_foldove(df)
broj_opstina = df["opstina_maticni_broj"].nunique()
print(f"naselja {len(df)} | opstina {broj_opstina} | foldova {len(FOLDS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Meta-model (stacking / kalibracija)

# COMMAND ----------

# DBTITLE 1,CV petlja meta-modela
def stacking_cv(x_kolone):
    """GroupKFold CV Ridge meta-modela nad zadatim ulazima.

    Sa jednim ulazom ovo je post-hoc kalibracija tog pristupa (nagib+presek
    u log-log prostoru); sa vise ulaza je stacking fuzija. Fituje se na
    trening opstinama, predvidja na validacionim (nista se ne fituje na
    podacima na kojima se ocenjuje).

    Vrati (oof_pred_pop, fold_r2, koeficijenti) gde je oof_pred_pop poravnat
    sa df redosledom, a koeficijenti prosek Ridge tezina preko foldova.
    """
    oof = pd.Series(np.nan, index=df.naselje_maticni_broj.values, dtype="float64")
    fold_r2, koef = [], []
    for train_frame, val_frame in FOLDS:
        reg = Ridge(alpha=CFG["alpha"]).fit(train_frame[x_kolone], train_frame["ylog"])
        plog = reg.predict(val_frame[x_kolone])
        oof.loc[val_frame.naselje_maticni_broj.values] = np.clip(np.expm1(plog), 0, None)
        fold_r2.append(r2_score(val_frame["ylog"], plog))
        koef.append(np.r_[reg.coef_, reg.intercept_])
    oof_pred = oof.loc[df.naselje_maticni_broj.values].values
    koef = np.mean(koef, axis=0)
    return oof_pred, fold_r2, dict(zip([*x_kolone, "presek"], koef.round(3)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluacija: sirovo vs kalibrisano vs fuzija

# COMMAND ----------

# DBTITLE 1,Pokreni i uporedi
KLJUCNE = ["cv_mean_val_r2", "oof_r2_log", "oof_medape", "oof_wmape", "oof_bias",
           "oof_kalib_nagib", "oof_opstina_r2_log", "oof_opstina_r2_log_bez_top2"]

def run():
    """Kalibracija svakog pristupa + stacking fuzija, sve u jednom MLflow runu.

    Za svaki pristup: metrike sirovih OOF predikcija i kalibrisanih (stacking
    sa jednim ulazom). Na kraju fuzija svih pristupa; njene metrike, rezime
    grafik i OOF parquet (`oof_fuzija.parquet`) se loguju kao artefakti.
    """
    podesi_mlflow()   # Databricks workspace, ili Colab -> Databricks preko env varijabli, ili lokalni mlruns
    stvarno = df["pop"].values.astype("float32")
    redovi = []

    with mlflow.start_run(run_name=f"fuzija-stacking-cv{len(FOLDS)}"):
        mlflow.log_params({**CFG, "pristup": "fuzija_stacking", "meta_model": "Ridge(log1p)",
                           "ulazi": ",".join(PRISTUPI), "cv": f"GroupKFold(opstina) x{len(FOLDS)}",
                           "n_naselja": len(df), "n_opstina": int(broj_opstina)})

        # sirove i kalibrisane metrike po pristupu
        for ime in PRISTUPI:
            redovi.append({"varijanta": f"{ime} (sirovo)",
                           **oof_metrics(stvarno, df[f"pred_{ime}"].values, df)})
            kal_pred, kal_r2, kal_koef = stacking_cv([f"x_{ime}"])
            print(f"[kalibracija {ime}] koeficijenti {kal_koef}")
            redovi.append({"varijanta": f"{ime} (kalibrisano)",
                           "cv_mean_val_r2": float(np.mean(kal_r2)),
                           "cv_std_val_r2": float(np.std(kal_r2)),
                           **oof_metrics(stvarno, kal_pred, df)})

        # fuzija svih pristupa
        fuz_pred, fuz_r2, fuz_koef = stacking_cv([f"x_{ime}" for ime in PRISTUPI])
        print(f"[fuzija] koeficijenti {fuz_koef}")
        agg = {
            "cv_mean_val_r2": float(np.mean(fuz_r2)),
            "cv_std_val_r2":  float(np.std(fuz_r2)),
            **oof_metrics(stvarno, fuz_pred, df),
        }
        redovi.append({"varijanta": "fuzija (stacking)", **agg})

        mlflow.log_metrics(agg)
        mlflow.log_dict(fuz_koef, "fuzija_koeficijenti.json")
        put_oof = sacuvaj_oof(df, fuz_pred, "fuzija", OUT_DIR)
        mlflow.log_artifact(put_oof)
        fig = cv_summary_figure(fuz_r2, agg, stvarno, fuz_pred, df, label="fuzija")
        plt.show()
        mlflow.log_figure(fig, "cv_evaluacija_fuzija.png")

        poredjenje = pd.DataFrame(redovi).set_index("varijanta").reindex(columns=KLJUCNE)
        mlflow.log_text(poredjenje.to_string(), "poredjenje_fuzija.txt")

    print(f"[fuzija] CV R2 {agg['cv_mean_val_r2']:.3f} ± {agg['cv_std_val_r2']:.3f}"
          f" | OOF R2(log) {agg['oof_r2_log']:.3f} | medAPE {agg.get('oof_medape', float('nan')):.2f}"
          f" | wMAPE {agg.get('oof_wmape', float('nan')):.2f} | bias {agg.get('oof_bias', float('nan')):.2f}"
          f" | opstina R2(log, bez top2) {agg.get('oof_opstina_r2_log_bez_top2', float('nan')):.3f}")
    return poredjenje


poredjenje = run()
print("\n=== Poredjenje: sirovo vs kalibrisano vs fuzija ===")
display(poredjenje)
