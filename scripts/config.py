"""Putanje projekta na jednom mestu.

Sve skripte u ``scripts/`` uvoze odavde (``from scripts import config``)
umesto da svaka sklapa svoje putanje. Pokrecu se iz korena repoa kao moduli::

    python -m scripts.preprocessing.build_labels

Podela direktorijuma:

* ``DATA``    – ulazi i medjukoraci; sve sto neki sledeci korak cita kao ulaz.
                Van gita (preveliko), deli se kao zip preko Drive-a.
* ``RESULTS`` – terminalne tabele i sazeci (.csv, .json); u gitu.
* ``FIGURES`` – terminalne slike (.png); u gitu.

Terminalno = niko dalje to ne konzumira kao ulaz, nego se gleda ili predaje.
Tezine modela i OOF predikcije ne idu ni u jedno od ovoga — one nastaju u
notebocima i idu u ``out/`` i uz MLflow run (videti ``core.environment``).
"""
import os

KOREN   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(KOREN, "data")
RESULTS = os.path.join(KOREN, "results")
FIGURES = os.path.join(KOREN, "figures")


def _p(*delovi: str) -> str:
    """Putanja unutar DATA (os.path.join, pa radi i van Windowsa)."""
    return os.path.join(DATA, *delovi)


# --- sirovi ulazi (preuzeti spolja) ---------------------------------------
NASELJA_GPKG   = _p("naselja", "naselje.gpkg")            # GeoSrbija RPJ
OPSTINE_GPKG   = _p("opstine", "opstine.gpkg")            # GeoSrbija RPJ
POPISNI_KRUG   = _p("popisnikrugovi", "popisni_krug.gpkg")
RZS_XLSX       = _p("rzs", "ukupno_stanovnika_naselja.xlsx")   # Popis 2022
SENTINEL_DIR   = _p("sentinel")                            # pojedinacni .tiff za EDA

# --- medjukoraci (pravi ih pipeline, cita ih sledeci korak) ---------------
NASELJE_POP    = _p("rzs", "naselje_pop_final.csv")        # preprocessing.build_labels
DATASET_DIR    = _p("dataset")
NASELJE_TABLE  = _p("dataset", "naselje_table.parquet")    # preprocessing.make_dataset_table
NASELJE_TABLE_CSV   = _p("dataset", "naselje_table.csv")
NASELJE_FOOTPRINTS  = _p("dataset", "naselje_footprints.parquet")
TILES_INDEX    = _p("dataset", "tiles_index.csv")          # sentinel.tiles

OVERTURE_OKRUG = _p("overture_okrug")   # otisci po okrugu; ulaz za rasterizaciju i plocice
OVERTURE_RURAL = _p("overture_rural")   # otisci uzorka sela (dijagnostika)
OKRUG_COMP     = _p("okrug_comp")       # Sentinel kompoziti po okrugu

CUTOUTS        = _p("cutouts")           # 1 isecak po naselju (za multimodalnu fuziju F2)
CUTOUTS_INDEX  = _p("cutouts", "index.csv")
FOOTPRINT_CUT  = _p("footprint_cutouts")  # rasterizovani otisci (pristup 2)
TILES          = _p("tiles")             # plocice 2.24 km (pristup 1)


def obezbedi(*putanje: str) -> None:
    """Napravi direktorijume ako ne postoje."""
    for p in putanje:
        os.makedirs(p, exist_ok=True)
