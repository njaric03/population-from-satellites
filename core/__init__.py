"""core – zajednicki modul za projekat procene broja stanovnika
iz daljinskog osmatranja (Sentinel-2 / footprint / tiles).

Sve sto je identicno u vise notebooka zivi ovde; notebooci drze samo ono
sto je specificno za svoj pristup.

Upotreba u notebucima (paket je u korenu repoa, koren se dodaje na sys.path)::

    from core import (
        seed_everything, prodji, dvofazni_trening,
        stats_po_opsegu, Naselja, napravi_loadere, NW,
        napravi_foldove, oof_metrics, cv_summary_figure,
        podesi_mlflow, izlazni_dir, sacuvaj_oof,
    )
"""
from .data        import NW, Naselja, napravi_loadere, seed_worker, stats_po_opsegu
from .train       import dvofazni_trening, prodji, seed_everything
from .cv          import cv_summary_figure, napravi_foldove, oof_metrics
from .environment import NA_DATABRICKSU, izlazni_dir, podesi_mlflow, sacuvaj_oof

__all__ = [
    # data
    "NW",
    "Naselja",
    "napravi_loadere",
    "seed_worker",
    "stats_po_opsegu",
    # train
    "dvofazni_trening",
    "prodji",
    "seed_everything",
    # cv
    "cv_summary_figure",
    "napravi_foldove",
    "oof_metrics",
    # environment
    "NA_DATABRICKSU",
    "izlazni_dir",
    "podesi_mlflow",
    "sacuvaj_oof",
]
