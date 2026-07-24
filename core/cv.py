from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error


def napravi_foldove(
    df: pd.DataFrame,
    group_col: str = "opstina_maticni_broj",
    n_splits: int = 5,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    # GroupKFold podela na (train, val) parove, sa proverom curenja opstina.
    k = min(n_splits, df[group_col].nunique())      # ne vise foldova nego grupa
    gkf = GroupKFold(n_splits=k)
    folds = [
        (
            df.iloc[tr].reset_index(drop=True),
            df.iloc[va].reset_index(drop=True),
        )
        for tr, va in gkf.split(df, groups=df[group_col])
    ]
    for i, (t, v) in enumerate(folds):
        preklop = set(t[group_col]) & set(v[group_col])
        if preklop:
            # raise, ne assert: provera protiv curenja mora uvek da radi
            raise ValueError(
                f"curenje izmedju train i val u foldu {i}: "
                f"{len(preklop)} zajednickih vrednosti '{group_col}'"
            )
    return folds


# Granice velicinskih stratuma (broj stanovnika) i oznake za imena metrika
STRATUM_GRANICE = [0, 500, 5_000, 50_000, np.inf]
STRATUM_OZNAKE  = ["do_500", "500_5k", "5k_50k", "50k_plus"]


def _fmt(agg: dict, kljuc: str, fmt: str = ".3f") -> str:
    # Metrika za ispis, ili "n/a" ako je oof_metrics nije vratio.
    return format(agg[kljuc], fmt) if kljuc in agg else "n/a"


def oof_metrics(
    stvarno: np.ndarray,
    oof_pred: np.ndarray,
    df: pd.DataFrame,
    group_col: str = "opstina_maticni_broj",
) -> dict:
    # 13 OOF metrika; izbor i obrazlozenje su u README, odeljak Metrike. oof_pred je u
    # stanovnicima, ne u log prostoru. Kljucevi koji bi ispali NaN se izostavljaju, da
    # MLflow logovanje ne pukne.
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

    metrike = {
        "oof_r2_log":          float(r2_score(ylog, plog)),
        "oof_mae_stanovnici":  float(mean_absolute_error(stvarno, oof_pred)),
        "oof_mediana_ae":      float(np.median(abs_gres)),
        "oof_opstina_r2_log":  float(
            r2_score(np.log1p(po_grupi.stvarno), np.log1p(po_grupi.pred))
        ),
    }

    # relativne greske: definisane samo za pop > 0 (u skupu ima naselja sa 0)
    poz = stvarno > 0
    if poz.any():
        metrike["oof_medape"] = float(np.median(abs_gres[poz] / stvarno[poz]))
    if stvarno.sum() > 0:
        metrike["oof_wmape"] = float(abs_gres.sum() / stvarno.sum())
        metrike["oof_bias"]  = float(oof_pred.sum() / stvarno.sum())

    # kalibracioni nagib: log1p(true) ~ log1p(pred); degenerisan ako su
    # predikcije ~konstantne (std ~ 0)
    if float(np.std(plog)) > 1e-9:
        nagib, _ = np.polyfit(plog, ylog, 1)
        metrike["oof_kalib_nagib"] = float(nagib)

    # opstina agregacija bez dve najvece opstine (po stvarnoj populaciji)
    if len(po_grupi) > 4:
        bez_top2 = po_grupi.drop(po_grupi.stvarno.nlargest(2).index)
        metrike["oof_opstina_r2_log_bez_top2"] = float(
            r2_score(np.log1p(bez_top2.stvarno), np.log1p(bez_top2.pred))
        )

    # relativna greska po velicinskim stratumima (granice po stvarnoj populaciji);
    # broj naselja po stratumu je opis skupa a ne rezultat runa - vidi se na
    # cv_summary_figure iznad stubica
    for lo, hi, oznaka in zip(
        STRATUM_GRANICE[:-1], STRATUM_GRANICE[1:], STRATUM_OZNAKE
    ):
        bin_poz = (stvarno >= lo) & (stvarno < hi) & poz
        if bin_poz.any():
            metrike[f"oof_medape_{oznaka}"] = float(
                np.median(abs_gres[bin_poz] / stvarno[bin_poz])
            )

    return {k: v for k, v in metrike.items() if np.isfinite(v)}


# Kolone za tabele poredjenja izmedju pristupa; podskup oof_metrics kljuceva
# koji staje u sirinu ekrana i pokriva uklapanje, relativnu gresku i pristrasnost.
GLAVNE_KOLONE = [
    "cv_mean_val_r2", "oof_r2_log", "oof_medape", "oof_wmape", "oof_bias",
    "oof_kalib_nagib", "oof_opstina_r2_log", "oof_opstina_r2_log_bez_top2",
]


def metrike_runa(
    fold_r2: list[float],
    stvarno: np.ndarray,
    oof_pred: np.ndarray,
    df: pd.DataFrame,
    group_col: str = "opstina_maticni_broj",
) -> dict:
    # Po-fold CV R2 (prosek i rasipanje) plus pun skup OOF metrika.
    return {
        "cv_mean_val_r2": float(np.mean(fold_r2)),
        "cv_std_val_r2":  float(np.std(fold_r2)),
        **oof_metrics(stvarno, oof_pred, df, group_col),
    }


def rezime_linija(agg: dict, label: str) -> str:
    # Jednolinijski rezime runa za ispis na kraju notebooka.
    return (
        f"[{label}] CV R2 {agg['cv_mean_val_r2']:.3f} ± {agg['cv_std_val_r2']:.3f}"
        f" | OOF R2(log) {agg['oof_r2_log']:.3f}"
        f" | medAPE {_fmt(agg, 'oof_medape', '.2f')}"
        f" | wMAPE {_fmt(agg, 'oof_wmape', '.2f')}"
        f" | bias {_fmt(agg, 'oof_bias', '.2f')}"
        f" | opstina R2(log, bez top2) {_fmt(agg, 'oof_opstina_r2_log_bez_top2')}"
    )


def kalibracija_figure(
    stvarno: np.ndarray,
    df: pd.DataFrame,
    pristupi: list[str],
    kalibrisani: dict,
) -> plt.Figure:
    # Efekat kalibracije po pristupu: nagib levo, medAPE desno, sirovo vs kalibrisano.
    metodi = sorted({m for _, m in kalibrisani})
    serije = ["sirovo", *metodi]

    # oof_metrics jednom po (pristup, serija); ranije se zvao po metrici pa se
    # isti racun ponavljao za svaki panel
    izmereno = {}
    for ime in pristupi:
        izmereno[(ime, "sirovo")] = oof_metrics(stvarno, df[f"pred_{ime}"].values, df)
        for md in metodi:
            izmereno[(ime, md)] = oof_metrics(stvarno, kalibrisani[(ime, md)], df)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(pristupi))
    sirina = 0.8 / len(serije)

    for panel, kljuc, naslov, cilj in (
        (ax[0], "oof_kalib_nagib", "Kalibracioni nagib (log-log)", 1.0),
        (ax[1], "oof_medape", "medAPE (medijalna procentualna greska)", None),
    ):
        for i, serija in enumerate(serije):
            vrednosti = [izmereno[(ime, serija)].get(kljuc, np.nan) for ime in pristupi]
            panel.bar(x + i * sirina - 0.4 + sirina / 2, vrednosti, sirina, label=serija)
        if cilj is not None:
            panel.axhline(cilj, color="red", ls="--", lw=1, label="cilj = 1.0")
        panel.set_xticks(x)
        panel.set_xticklabels(pristupi, rotation=20, ha="right")
        panel.set_title(naslov)
        panel.legend(fontsize=9)

    plt.tight_layout()
    return fig


def cv_summary_figure(
    fold_r2: list[float],
    agg: dict,
    stvarno: np.ndarray,
    oof_pred: np.ndarray,
    df: pd.DataFrame,
    group_col: str = "opstina_maticni_broj",
    label: str = "",
) -> plt.Figure:
    # 2x2 CV rezime: R2 po foldu, metrike, scatter po naselju, medAPE po stratumima.
    oof_pred = np.clip(oof_pred, 0, None)
    stvarno = np.asarray(stvarno, dtype="float64")

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
    ax[0, 1].axis("off")
    ax[0, 1].text(
        0.02, 0.5,
        f"CV R2 = {agg['cv_mean_val_r2']:.3f} \u00b1 {agg['cv_std_val_r2']:.3f}\n"
        f"OOF R2 (log-pop) = {agg['oof_r2_log']:.3f}\n"
        f"medAPE = {_fmt(agg, 'oof_medape', '.2f')}"
        f" | wMAPE = {_fmt(agg, 'oof_wmape', '.2f')}\n"
        f"bias (sum pred/true) = {_fmt(agg, 'oof_bias', '.2f')}\n"
        f"kalib. nagib (log-log) = {_fmt(agg, 'oof_kalib_nagib', '.2f')}\n"
        f"medijana AE (st) = {_fmt(agg, 'oof_mediana_ae', '.0f')}"
        f" | MAE (st) = {agg['oof_mae_stanovnici']:.0f}\n"
        f"opstina R2 (log) = {_fmt(agg, 'oof_opstina_r2_log')}\n"
        f"opstina R2 (log, bez top2) = {_fmt(agg, 'oof_opstina_r2_log_bez_top2')}",
        fontsize=12, va="center",
    )

    # Scatter po naselju
    maks = max(float(stvarno.max()), float(oof_pred.max()), 1.0)
    ax[1, 0].scatter(stvarno, oof_pred, s=12, alpha=0.4)
    ax[1, 0].plot([1, maks], [1, maks], "r--")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("OOF naselje: stvarno vs predvidjeno")
    ax[1, 0].set_xlabel("stvarno")
    ax[1, 0].set_ylabel("predvidjeno")

    # medAPE po velicinskim stratumima: gde model radi a gde ne.
    # Agregacija po opstini je vec u tekstualnom panelu (dva broja, sa i bez
    # top-2); ovde je korisniji raspored greske po velicini naselja, jer je
    # raspodela toliko iskosena da jedan prosek nista ne kaze.
    oznake, vrednosti, brojevi = [], [], []
    for lo, hi, oznaka in zip(
        STRATUM_GRANICE[:-1], STRATUM_GRANICE[1:], STRATUM_OZNAKE
    ):
        kljuc = f"oof_medape_{oznaka}"
        if kljuc not in agg:
            continue
        oznake.append(oznaka)
        vrednosti.append(agg[kljuc])
        brojevi.append(int(((stvarno >= lo) & (stvarno < hi)).sum()))

    if oznake:
        stubici = ax[1, 1].bar(range(len(oznake)), vrednosti, color="tab:orange")
        for s, n in zip(stubici, brojevi):
            ax[1, 1].text(s.get_x() + s.get_width() / 2, s.get_height(),
                          f"n={n}", ha="center", va="bottom", fontsize=9)
        ax[1, 1].set_xticks(range(len(oznake)))
        ax[1, 1].set_xticklabels(oznake)
        ax[1, 1].set_ylim(0, max(vrednosti) * 1.18)
    ax[1, 1].set_title("medAPE po velicini naselja (stanovnika)")
    ax[1, 1].set_ylabel("medijalna relativna greska")

    plt.tight_layout()
    return fig
