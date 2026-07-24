import os

# Databricks runtime uvek ima /databricks na disku.
NA_DATABRICKSU: bool = os.path.isdir("/databricks")

# UC Volume drzi i podatke i izlaze treninga; core.environment uzima isti koren
VOLUME = "/Volumes/katalog/deep_learning/raw_data"

KOREN   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ulazi i medjukoraci: Volume na Databricksu, data/ iz repoa lokalno
DATA    = (VOLUME + "/data") if NA_DATABRICKSU else os.path.join(KOREN, "data")
RESULTS = os.path.join(KOREN, "results")    # terminalne tabele i sazeci, u gitu
FIGURES = os.path.join(KOREN, "figures")    # terminalne slike, u gitu


def _p(*delovi: str) -> str:
    """Putanja unutar DATA (os.path.join, pa radi i van Windowsa)."""
    return os.path.join(DATA, *delovi)


def _tabela(ime: str) -> str:
    """Tabela iz ``dataset/``, sa tolerancijom na ravan raspored.

    Pipeline ih pise u ``DATA/dataset/``, ali na UC Volume-u stoje ravno u
    ``DATA/``. Uzima se ona koja postoji, a ako nema nijedne ona pod
    ``dataset/`` jer tamo ih pipeline pravi.
    """
    ugnjezdena = _p("dataset", ime)
    ravna = _p(ime)
    if not os.path.exists(ugnjezdena) and os.path.exists(ravna):
        return ravna
    return ugnjezdena


# --- sirovi ulazi (preuzeti spolja) ---------------------------------------
NASELJA_GPKG   = _p("naselja", "naselje.gpkg")            # GeoSrbija RPJ
OPSTINE_GPKG   = _p("opstine", "opstine.gpkg")            # GeoSrbija RPJ
POPISNI_KRUG   = _p("popisnikrugovi", "popisni_krug.gpkg")
RZS_XLSX       = _p("rzs", "ukupno_stanovnika_naselja.xlsx")   # Popis 2022
SENTINEL_DIR   = _p("sentinel")                            # pojedinacni .tiff za EDA

# --- medjukoraci (pravi ih pipeline, cita ih sledeci korak) ---------------
NASELJE_POP    = _p("rzs", "naselje_pop_final.csv")        # preprocessing.build_labels
DATASET_DIR    = _p("dataset")
NASELJE_TABLE  = _tabela("naselje_table.parquet")          # preprocessing.make_dataset_table
NASELJE_TABLE_CSV   = _tabela("naselje_table.csv")
NASELJE_FOOTPRINTS  = _tabela("naselje_footprints.parquet")   # footprint.per_naselje
NASELJE_FOOTPRINTS_CSV = _tabela("naselje_footprints.csv")
TILES_INDEX    = _tabela("tiles_index.csv")                # sentinel.tiles

OVERTURE_OKRUG = _p("overture_okrug")   # otisci po okrugu; ulaz za rasterizaciju i plocice
OVERTURE_RURAL = _p("overture_rural")   # otisci uzorka sela (dijagnostika)
OKRUG_COMP     = _p("okrug_comp")       # Sentinel kompoziti po okrugu

CUTOUTS        = _p("cutouts")           # 1 isecak po naselju (za multimodalnu fuziju F2)
CUTOUTS_INDEX  = _p("cutouts", "index.csv")
FOOTPRINT_CUT  = _p("footprint_cutouts")  # rasterizovani otisci (pristup 2)
TILES          = _p("tiles")             # plocice 2.24 km (pristup 1)

# --- terminalni izlazi (u gitu) -------------------------------------------
RURAL_FOOTPRINTS = os.path.join(RESULTS, "rural_footprints.csv")   # footprint.coverage
OVERTURE_POPUNJENOST = os.path.join(RESULTS, "overture_popunjenost_atributa.csv")


def obezbedi(*putanje: str) -> None:
    """Napravi direktorijume ako ne postoje."""
    for p in putanje:
        os.makedirs(p, exist_ok=True)


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
