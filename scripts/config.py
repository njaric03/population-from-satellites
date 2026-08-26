import os
from pathlib import Path

# Databricks runtime uvek ima /databricks na disku
NA_DATABRICKSU: bool = Path("/databricks").is_dir()

# UC Volume drzi i podatke i izlaze treninga; core.environment uzima isti koren
VOLUME = Path("/Volumes/katalog/deep_learning/raw_data")

KOREN = Path(__file__).resolve().parent.parent
DATA = VOLUME / "data" if NA_DATABRICKSU else KOREN / "data"   # ulazi i medjukoraci
NOTEBOOKS = KOREN / "notebooks"                                # izvrseni .ipynb, ulaz za report
RESULTS = KOREN / "results"                                    # tabele i sazeci, u gitu
FIGURES = KOREN / "figures"                                    # slike, u gitu


def _table(ime: str) -> Path:
    # pipeline pise u DATA/dataset/, na Volume-u stoje ravno u DATA/
    ugnjezdena = DATA / "dataset" / ime
    ravna = DATA / ime
    return ravna if not ugnjezdena.exists() and ravna.exists() else ugnjezdena


# --- sirovi ulazi (preuzeti spolja) ---------------------------------------
NASELJA_GPKG = DATA / "naselja" / "naselje.gpkg"               # GeoSrbija RPJ
OPSTINE_GPKG = DATA / "opstine" / "opstine.gpkg"               # GeoSrbija RPJ
POPISNI_KRUG = DATA / "popisnikrugovi" / "popisni_krug.gpkg"
RZS_XLSX = DATA / "rzs" / "ukupno_stanovnika_naselja.xlsx"     # Popis 2022
SENTINEL_DIR = DATA / "sentinel"                               # pojedinacni .tiff za EDA

# --- medjukoraci (pravi ih pipeline, cita ih sledeci korak) ---------------
NASELJE_POP = DATA / "rzs" / "naselje_pop_final.csv"           # preprocessing.build_labels
DATASET_DIR = DATA / "dataset"
NASELJE_TABLE = _table("naselje_table.parquet")               # preprocessing.make_dataset_table
NASELJE_TABLE_CSV = _table("naselje_table.csv")
NASELJE_FOOTPRINTS = _table("naselje_footprints.parquet")     # footprint.per_naselje
NASELJE_FOOTPRINTS_CSV = _table("naselje_footprints.csv")
TILES_INDEX = _table("tiles_index.csv")                       # sentinel.tiles

OVERTURE_OKRUG = DATA / "overture_okrug"    # otisci po okrugu, ulaz za rastere i plocice
OVERTURE_RURAL = DATA / "overture_rural"    # otisci uzorka sela (dijagnostika)
OKRUG_COMP = DATA / "okrug_comp"            # Sentinel kompoziti po okrugu

CUTOUTS = DATA / "cutouts"                  # 1 isecak po naselju (fuzija F2)
CUTOUTS_INDEX = DATA / "cutouts" / "index.csv"
FOOTPRINT_CUT = DATA / "footprint_cutouts"  # rasterizovani otisci (pristup 2)
TILES = DATA / "tiles"                      # plocice 2.24 km (pristup 1)

# --- terminalni izlazi (u gitu) -------------------------------------------
RURAL_FOOTPRINTS = RESULTS / "rural_footprints.csv"            # footprint.coverage
OVERTURE_POPUNJENOST = RESULTS / "overture_popunjenost_atributa.csv"
METRIKE = RESULTS / "metrike_po_pristupu.csv"                  # scripts.report


def ensure_dirs(*putanje: Path) -> None:
    # napravi direktorijume ako ne postoje
    for putanja in putanje:
        os.makedirs(putanja, exist_ok=True)


# --- strukturirani atributi otisaka (pristup 2) ----------------------------
# footprint/per_naselje ih racuna, 03_footprint_train i 04_multimodal_train ih
# citaju. Svi su izvedeni iz geometrije; Overture opisne
# kolone se ne koriste (num_floors 0.76%, height 0.05%, subtype/class ~8.1%,
# i to neravnomerno po okruzima, pa mere gustinu mapiranja a ne izgradjenost).

# racuna ih per_naselje agregacijom po naselju
FP_AGREGIRANI = [
    "n_buildings", "roof_area_m2",                       # kolicina
    "mean_bsize", "median_bsize", "std_bsize", "p90_bsize",
    "mean_compact", "udeo_velikih",                      # oblik
    "mean_nn_dist", "median_nn_dist", "mean_n_50m",      # raspored
]

# izvedeni iz agregiranih i povrsine naselja (GeoSrbija geometrija)
FP_IZVEDENI = ["building_density", "built_fraction"]

# tezak rep pa idu kroz log1p pre standardizacije
FP_LOG = [
    "n_buildings", "roof_area_m2", "mean_bsize", "median_bsize", "std_bsize",
    "p90_bsize", "mean_nn_dist", "median_nn_dist", "mean_n_50m",
    "building_density", "area_km2",
]
# ogranicene velicine (udeo, kompaktnost), ostaju kakve jesu
FP_LIN = ["mean_compact", "udeo_velikih", "built_fraction"]

# ulaz modela: 14 atributa
FP_ATRIBUTI = FP_LOG + FP_LIN
