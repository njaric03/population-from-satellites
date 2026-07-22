"""
Detekcija okruzenja (Databricks / Colab / lokalno), MLflow tracking,
izlazni direktorijum i cuvanje OOF predikcija.

Notebuci rade nepromenjeni u oba okruzenja:

* Databricks: workspace MLflow tracking (podrazumevan), izlaz na UC Volume.
* Colab: ako su postavljeni ``DATABRICKS_HOST`` i ``DATABRICKS_TOKEN``,
  metrike idu u isti Databricks eksperiment; inace lokalni ``mlruns/``
  file store. Izlaz ide u ``/content/out``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Databricks runtime uvek ima /databricks na disku; Colab ima /content.
NA_DATABRICKSU: bool = os.path.isdir("/databricks")

VOLUME_OUT = "/Volumes/katalog/deep_learning/raw_data"
EXPERIMENT = "/Users/korisnik/procena_stanovnika"


def izlazni_dir() -> str:
    """Direktorijum za tezine modela i OOF parquet fajlove.

    Databricks: UC Volume. Colab: ``/content/out``. Lokalno: ``out/``.
    Kreira direktorijum ako ne postoji.
    """
    if NA_DATABRICKSU:
        out = VOLUME_OUT
    elif os.path.isdir("/content"):
        out = "/content/out"
    else:
        out = "out"
    os.makedirs(out, exist_ok=True)
    return out


def podesi_mlflow(experiment: str = EXPERIMENT) -> None:
    """Podesi MLflow tracking prema okruzenju i otvori eksperiment.

    * Databricks: workspace tracking je vec aktivan, samo set_experiment.
    * Van Databricksa sa ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` u env:
      pise u isti Databricks eksperiment preko REST API-ja.
    * Inace: lokalni file store (``mlruns/``); ime eksperimenta je poslednji
      segment workspace putanje jer file store ne poznaje /Users/... putanje.
    """
    import mlflow   # lazy: ostatak modula (izlazni_dir, sacuvaj_oof) ne zavisi od mlflow-a

    if NA_DATABRICKSU:
        mlflow.set_experiment(experiment)
        return
    if os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(experiment)
        return
    mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
    mlflow.set_experiment(experiment.rsplit("/", 1)[-1])


def sacuvaj_oof(
    df: pd.DataFrame,
    oof_pred: np.ndarray,
    pristup: str,
    out_dir: str,
) -> str:
    """Snimi OOF predikcije u parquet i vrati putanju fajla.

    Parquet ima po jedan red za svako naselje (kolone:
    ``naselje_maticni_broj``, ``opstina_maticni_broj``, ``pop``, ``pred``)
    i sluzi kao ulaz za fuziju (stacking) i zbirno poredjenje pristupa.
    Putanju proslediti ``mlflow.log_artifact`` da ostane uz run.

    Args:
        df:        DataFrame u istom redosledu kao ``oof_pred``; mora imati
                   kolone naselje_maticni_broj, opstina_maticni_broj, pop.
        oof_pred:  OOF predikcije u prostoru populacije, 1D array.
        pristup:   oznaka pristupa za ime fajla (npr. "sentinel_pop",
                   "footprint", "tiles_log").
        out_dir:   direktorijum (videti ``izlazni_dir``).

    Returns:
        Putanja snimljenog parquet fajla.
    """
    put = os.path.join(out_dir, f"oof_{pristup}.parquet")
    (
        df[["naselje_maticni_broj", "opstina_maticni_broj", "pop"]]
        .assign(pred=np.clip(oof_pred, 0, None).astype("float32"))
        .to_parquet(put, index=False)
    )
    return put
