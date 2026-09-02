from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# koren okruzenja stoji u scripts.config, da postoji tacno jedna definicija
from scripts.config import NA_DATABRICKSU, VOLUME

# MLflow eksperiment je putanja u radnom prostoru, pa zavisi od naloga
EXPERIMENT = os.environ.get("POPULACIJA_MLFLOW_EXPERIMENT", "/Shared/procena_stanovnika")


def output_dir() -> Path:
    # tezine i OOF parquet: UC Volume na Databricksu, out/ lokalno
    izlaz = VOLUME if NA_DATABRICKSU else Path("out")
    os.makedirs(izlaz, exist_ok=True)
    return izlaz


def setup_mlflow(experiment: str = EXPERIMENT) -> None:
    # MLflow tracking: Databricks workspace, lokalno mlruns/
    import mlflow   # lazy: output_dir i save_oof ne zavise od mlflow-a

    if NA_DATABRICKSU:
        mlflow.set_experiment(experiment)
        return
    mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
    # file store ne poznaje /Users/... putanje, pa ide poslednji segment
    mlflow.set_experiment(experiment.rsplit("/", 1)[-1])


def save_oof(
    df: pd.DataFrame,
    oof_pred: np.ndarray,
    pristup: str,
    out_dir: Path,
) -> Path:
    # OOF predikcije, red po naselju; df mora biti u istom redosledu kao oof_pred
    # ulaz za fuziju, putanju proslediti mlflow.log_artifact da ostane uz run
    put = Path(out_dir) / f"oof_{pristup}.parquet"
    (
        df[["naselje_maticni_broj", "opstina_maticni_broj", "pop"]]
        .assign(pred=np.clip(oof_pred, 0, None).astype("float32"))
        .to_parquet(put, index=False)
    )
    return put
