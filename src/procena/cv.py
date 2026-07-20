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


# Granice velicinskih stratuma (broj stanovnika) i oznake za imena metrika
STRATUM_GRANICE = [0, 500, 5_000, 50_000, np.inf]
STRATUM_OZNAKE  = ["do_500", "500_5k", "5k_50k", "50k_plus"]


def oof_metrics(
    stvarno: np.ndarray,
    oof_pred: np.ndarray,
    df: pd.DataFrame,
    group_col: str = "opstina_maticni_broj",
) -> dict:
    """Standardni skup OOF metrika.

    Pored R2/MAE/RMSE (log-prostor i stanovnici) i agregacije po opstini,
    racuna i skup metrika otpornih na jako iskosenu raspodelu populacije
    (medijana naselja ~265, maksimum ~260k — prosecne apsolutne greske i
    linearni R2 na sumama dominiraju najveci gradovi):

    * ``oof_medape``        – medijalna apsolutna procentualna greska
      (samo naselja sa pop > 0); "tipican procenat promasaja".
    * ``oof_mediana_ae``    – medijalna apsolutna greska u stanovnicima.
    * ``oof_wmape``         – sum|pred-true| / sum(true); agregatna relativna
      greska ponderisana populacijom (robusna zamena za MAE kao headline).
    * ``oof_bias``          – sum(pred) / sum(true) na celom skupu; < 1 znaci
      sistematsko potcenjivanje ukupne populacije.
    * ``oof_kalib_nagib``   – nagib regresije log1p(true) ~ log1p(pred);
      > 1 znaci kompresiju predikcija ka sredini (kandidat za post-hoc
      kalibraciju).
    * ``oof_opstina_r2_bez_top2`` / ``..._log_bez_top2`` – agregacija po
      opstini bez dve najvece opstine (Beograd, Novi Sad); razdvaja
      "model ne radi" od "megagradovi su neekstrapolabilni u GroupKFold-u".
    * ``oof_medape_<strat>``, ``oof_mae_<strat>``, ``oof_n_<strat>`` – po
      velicinskim stratumima ``do_500``, ``500_5k``, ``5k_50k``, ``50k_plus``
      (granice po stvarnoj populaciji); pokazuju gde model radi a gde ne.

    Napomena: linearni ``oof_opstina_r2`` se i dalje loguje radi kontinuiteta,
    ali na rasponu opstina 5k–1.6M tu metriku ili nose ili ruse dva grada —
    ne koristiti je kao glavnu; glavne su ``oof_r2_log``, ``oof_medape``,
    ``oof_wmape`` i ``oof_bias``.

    Args:
        stvarno:   stvarni broj stanovnika, 1D array.
        oof_pred:  OOF predikcije u prostoru populacije (vec u stanovnicima,
                   ne u log-prostoru), 1D array.
        df:        DataFrame sa ``group_col`` kolonom (isti redosled kao gore).
        group_col: kolona za agregaciju (default: ``opstina_maticni_broj``).

    Returns:
        Dict metrika; kljucevi koji bi ispali NaN/inf (npr. prazan stratum)
        se izostavljaju da MLflow logovanje ne pukne.
        Dodajte ``cv_mean_val_r2`` i ``cv_std_val_r2`` iz ``fold_r2`` liste
        pre logovanja.
    """
    stvarno   = np.asarray(stvarno, dtype="float64")
    oof_pred  = np.clip(np.asarray(oof_pred, dtype="float64"), 0, None)
    ylog      = np.log1p(stvarno)
    plog      = np.log1p(oof_pred)
    abs_gres  = np.abs(oof_pred - stvarno)
    po_grupi  = (
        df.assign(pred=oof_pred, stvarno=stvarno)
        .groupby(group_col)[["pred", "stvarno"]]
        .sum()
    )

    m = {
        "oof_r2_log":          float(r2_score(ylog, plog)),
        "oof_mae_log":         float(mean_absolute_error(ylog, plog)),
        "oof_rmse_log":        float(mean_squared_error(ylog, plog) ** 0.5),
        "oof_mae_stanovnici":  float(mean_absolute_error(stvarno, oof_pred)),
        "oof_rmse_stanovnici": float(mean_squared_error(stvarno, oof_pred) ** 0.5),
        "oof_mediana_ae":      float(np.median(abs_gres)),
        "oof_opstina_r2":      float(r2_score(po_grupi.stvarno, po_grupi.pred)),
        "oof_opstina_r2_log":  float(
            r2_score(np.log1p(po_grupi.stvarno), np.log1p(po_grupi.pred))
        ),
    }

    # relativne greske: definisane samo za pop > 0 (u skupu ima naselja sa 0)
    poz = stvarno > 0
    if poz.any():
        m["oof_medape"] = float(np.median(abs_gres[poz] / stvarno[poz]))
    if stvarno.sum() > 0:
        m["oof_wmape"] = float(abs_gres.sum() / stvarno.sum())
        m["oof_bias"]  = float(oof_pred.sum() / stvarno.sum())

    # kalibracioni nagib: log1p(true) ~ log1p(pred); degenerisan ako su
    # predikcije ~konstantne (std ~ 0)
    if float(np.std(plog)) > 1e-9:
        nagib, _ = np.polyfit(plog, ylog, 1)
        m["oof_kalib_nagib"] = float(nagib)

    # opstina agregacija bez dve najvece opstine (po stvarnoj populaciji)
    if len(po_grupi) > 4:
        bez_top2 = po_grupi.drop(po_grupi.stvarno.nlargest(2).index)
        m["oof_opstina_r2_bez_top2"] = float(
            r2_score(bez_top2.stvarno, bez_top2.pred)
        )
        m["oof_opstina_r2_log_bez_top2"] = float(
            r2_score(np.log1p(bez_top2.stvarno), np.log1p(bez_top2.pred))
        )

    # metrike po velicinskim stratumima (granice po stvarnoj populaciji)
    for lo, hi, oznaka in zip(
        STRATUM_GRANICE[:-1], STRATUM_GRANICE[1:], STRATUM_OZNAKE
    ):
        u_bin = (stvarno >= lo) & (stvarno < hi)
        n = int(u_bin.sum())
        if n == 0:
            continue
        m[f"oof_n_{oznaka}"]   = n
        m[f"oof_mae_{oznaka}"] = float(abs_gres[u_bin].mean())
        bin_poz = u_bin & poz
        if bin_poz.any():
            m[f"oof_medape_{oznaka}"] = float(
                np.median(abs_gres[bin_poz] / stvarno[bin_poz])
            )

    return {k: v for k, v in m.items() if np.isfinite(v)}


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

    # Tekstualni rezime: glavne metrike (log R2 + relativne, robusne na skew),
    # pa dijagnostika (nagib, bias) i opstina agregacija sa i bez top-2
    def _f(kljuc: str, fmt: str = ".3f") -> str:
        return format(agg[kljuc], fmt) if kljuc in agg else "n/a"

    ax[0, 1].axis("off")
    ax[0, 1].text(
        0.02, 0.5,
        f"CV R2 = {agg['cv_mean_val_r2']:.3f} \u00b1 {agg['cv_std_val_r2']:.3f}\n"
        f"OOF R2 (log-pop) = {agg['oof_r2_log']:.3f}\n"
        f"medAPE = {_f('oof_medape', '.2f')} | wMAPE = {_f('oof_wmape', '.2f')}\n"
        f"bias (sum pred/true) = {_f('oof_bias', '.2f')}\n"
        f"kalib. nagib (log-log) = {_f('oof_kalib_nagib', '.2f')}\n"
        f"medijana AE (st) = {_f('oof_mediana_ae', '.0f')}"
        f" | MAE (st) = {agg['oof_mae_stanovnici']:.0f}\n"
        f"opstina R2 (log) = {_f('oof_opstina_r2_log')}\n"
        f"opstina R2 (log, bez top2) = {_f('oof_opstina_r2_log_bez_top2')}",
        fontsize=12, va="center",
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
