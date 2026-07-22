"""Spakuj cutoute + tabele u data_upload.zip za upload na Colab.
Struktura u zip-u: cutouts/<mb>.npy, index.csv, naselje_table.parquet, naselje_footprints.parquet
Ponovo pokreni kad subset/full zavrsi da osvezi zip.
"""
import os, glob, zipfile
from scripts import config

CUT = config.CUTOUTS; DSD = config.DATASET_DIR
ZIP = config.ZIP_DATA

npys = glob.glob(CUT + r"\*.npy")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:   # npy je vec kompaktan, bez kompresije
    for f in npys:
        z.write(f, "cutouts/" + os.path.basename(f))
    z.write(config.CUTOUTS_INDEX, "index.csv")
    z.write(config.NASELJE_TABLE, "naselje_table.parquet")
    fp = config.NASELJE_FOOTPRINTS
    if os.path.exists(fp): z.write(fp, "naselje_footprints.parquet")
mb = os.path.getsize(ZIP) / 1e6
print(f"WROTE {ZIP} | cutouts: {len(npys)} | velicina: {mb:.1f} MB")

# pristup 2: footprint rasteri (ako postoje)
FCUT = config.FOOTPRINT_CUT
if os.path.isdir(FCUT):
    fnpys = glob.glob(FCUT + r"\*.npy")
    FZIP = config.ZIP_FOOTPRINT
    with zipfile.ZipFile(FZIP, "w", zipfile.ZIP_STORED) as z:
        for f in fnpys:
            z.write(f, "footprint_cutouts/" + os.path.basename(f))
        z.write(config.NASELJE_TABLE, "naselje_table.parquet")
    print(f"WROTE {FZIP} | footprint cutouts: {len(fnpys)} | velicina: {os.path.getsize(FZIP)/1e6:.1f} MB")
