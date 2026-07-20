"""
GroupKFold CV podela, OOF metrike i CV rezime grafik.

Sve tri funkcije su identicne u svim trima notebucima – jedine razlike su
iznosi run_name stringa i put do .pt fajla, koji ostaju lokalni.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def napravi_foldove(
    df: pd.DataFrame,
    group_col: str = "opstina_maticni_broj",
    n_splits: int = 5,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """GroupKFold podela sa proverom curenja opstina.

    Broj foldova se automatski ogranicava na broj dostupnih grupa ako je
    ``n_splits`` vece od broja opstina u ``df``.

    Args:
        df:         DataFrame sa svim uzorcima; mora imati ``group_col``.
        group_col:  kolona po kojoj se grupise (default: ``opstina_maticni_broj``).
        n_splits:   maksimalni broj foldova (default: 5).

    Returns:
        Lista ``(train_frame, val_frame)`` parova sa resetovanim indeksima.
    """
    k = min(n_splits, df[group_col].nunique())
    gkf = GroupKFold(n_splits=k)
    folds = [
        (
            df.iloc[tr].reset_index(drop=True),
            df.iloc[va].reset_index(drop=True),
        )
        for tr, va in gkf.split(df, groups=df[group_col])
    ]
    for t, v in folds:
        assert not (
            set(t[group_col]) & set(v[group_col])
        ), "curenje opstine izmedju train i val!"
    return folds


def oof_metrics(
    stvarno: np.ndarray,
    oof_pred: np.ndarray,
    df: pd.DataFrame,
    group_col: str = "opstina_maticni_broj",
) -> dict:
    """Standardni skup OOF metrika (log-prostor + stanovnici + opstina agregacija).

    Args:
        stvarno:   stvarni broj stanovnika, 1D array.
        oof_pred:  OOF predikcije u prostoru populacije (vec u stanovnicima,
                   ne u log-prostoru), 1D array.
        df:        DataFrame sa ``group_col`` kolonom (isti redosled kao gore).
        group_col: kolona za agregaciju (default: ``opstina_maticni_broj``).

    Returns:
        Dict sa kljucevima: ``oof_r2_log``, ``oof_mae_log``, ``oof_rmse_log``,
        ``oof_mae_stanovnici``, ``oof_rmse_stanovnici``, ``oof_opstina_r2``,
        ``oof_opstina_r2_log``.
        Dodajte ``cv_mean_val_r2`` i ``cv_std_val_r2`` iz ``fold_r2`` liste
        pre logovanja.
    """
    oof_pred  = np.clip(oof_pred, 0, None)
    ylog      = np.log1p(stvarno)
    plog      = np.log1p(oof_pred)
    po_grupi  = (
        df.assign(pred=oof_pred, stvarno=stvarno)
        .groupby(group_col)[["pred", "stvarno"]]
        .sum()
    )
    return {
        "oof_r2_log":          float(r2_score(ylog, plog)),
        "oof_mae_log":         float(mean_absolute_error(ylog, plog)),
        "oof_rmse_log":        float(mean_squared_error(ylog, plog) ** 0.5),
        "oof_mae_stanovnici":  float(mean_absolute_error(stvarno, oof_pred)),
        "oof_rmse_stanovnici": float(mean_squared_error(stvarno, oof_pred) ** 0.5),
        "oof_opstina_r2":      float(r2_score(po_grupi.stvarno, po_grupi.pred)),
        "oof_opstina_r2_log":  float(
            r2_score(np.log1p(po_grupi.stvarno), np.log1p(po_grupi.pred))
        ),
    }


def cv_summary_figure(
    fold_r2: list[float],
    agg: dict,
    stvarno: np.ndarray,
    oof_pred: np.ndarray,
    df: pd.DataFrame,
    group_col: str = "opstina_maticni_broj",
    label: str = "",
) -> plt.Figure:
    """Standardni 2x2 CV rezime grafik.

    Isti raspored u svim trima notebucima:

    * Gore levo:  bar grafik best val R2 po foldu sa prosecnom linijom.
    * Gore desno: tekstualni rezime metrika.
    * Dole levo:  scatter stvarno vs predvidjeno po naselju (log skala).
    * Dole desno: scatter stvarno vs predvidjeno agregirano po opstini.

    Args:
        fold_r2:   lista best val R2 po foldu.
        agg:       dict metrika (iz ``oof_metrics`` + ``cv_mean/std_val_r2``).
        stvarno:   stvarni broj stanovnika, 1D array.
        oof_pred:  OOF predikcije u prostoru populacije, 1D array.
        df:        DataFrame sa ``group_col`` kolonom.
        group_col: kolona za agregaciju (default: ``opstina_maticni_broj``).
        label:     kratka oznaka pristupa za naslove (npr. ``"footprint"``,
                   ``"tiles"``, ``"sentinel"``).

    Returns:
        ``matplotlib.figure.Figure`` – proslediti ``mlflow.log_figure``.
    """
    oof_pred = np.clip(oof_pred, 0, None)
    po_grupi = (
        df.assign(pred=oof_pred, stvarno=stvarno)
        .groupby(group_col)[["pred", "stvarno"]]
        .sum()
    )

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # Bar grafik fold R2
    ax[0, 0].bar(range(len(fold_r2)), fold_r2)
    ax[0, 0].axhline(
        agg["cv_mean_val_r2"], color="red", ls="--",
        label=f"prosek {agg['cv_mean_val_r2']:.3f}",
    )
    ax[0, 0].set_title(f"Najbolji val R2 po foldu ({label})")
    ax[0, 0].set_xlabel("fold")
    ax[0, 0].legend()

    # Tekstualni rezime
    ax[0, 1].axis("off")
    ax[0, 1].text(
        0.02, 0.5,
        f"CV R2 = {agg['cv_mean_val_r2']:.3f} \u00b1 {agg['cv_std_val_r2']:.3f}\n"
        f"OOF R2 (log-pop) = {agg['oof_r2_log']:.3f}\n"
        f"OOF opstina R2 = {agg['oof_opstina_r2']:.3f}\n"
        f"OOF opstina R2 (log) = {agg['oof_opstina_r2_log']:.3f}\n"
        f"OOF MAE (st) = {agg['oof_mae_stanovnici']:.0f}",
        fontsize=13, va="center",
    )

    # Scatter po naselju
    m = max(float(stvarno.max()), float(oof_pred.max()), 1.0)
    ax[1, 0].scatter(stvarno, oof_pred, s=12, alpha=0.4)
    ax[1, 0].plot([1, m], [1, m], "r--")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("OOF naselje: stvarno vs predvidjeno")
    ax[1, 0].set_xlabel("stvarno")
    ax[1, 0].set_ylabel("predvidjeno")

    # Scatter po opstini (agregacija)
    mm = max(float(po_grupi.stvarno.max()), float(po_grupi.pred.max()), 1.0)
    ax[1, 1].scatter(po_grupi.stvarno, po_grupi.pred, s=35)
    ax[1, 1].plot([1, mm], [1, mm], "r--")
    ax[1, 1].set_title(
        f"OOF agregacija po {group_col} (R2 {agg['oof_opstina_r2']:.2f})"
    )
    ax[1, 1].set_xlabel("stvarno")
    ax[1, 1].set_ylabel("predvidjeno")

    plt.tight_layout()
    return fig
