"""procena – zajednicki modul za projekat procene broja stanovnika
iz daljinskog osmatranja (Sentinel-2 / footprint / tiles).

Upotreba u notebucima::

    import sys
    sys.path.insert(0, "/Workspace/Users/korisnik"
                       "/du-procena-stanovnistva/src")
    from procena import (
        seed_everything, prodji, dvofazni_trening,
        stats_po_opsegu, Naselja, napravi_loadere, NW,
        napravi_foldove, oof_metrics, cv_summary_figure,
    )
"""
from .data  import NW, Naselja, napravi_loadere, seed_worker, stats_po_opsegu
from .train import dvofazni_trening, prodji, seed_everything
from .cv    import cv_summary_figure, napravi_foldove, oof_metrics

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
]
