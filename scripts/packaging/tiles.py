"""Spakuj plocice + index + tabelu u tiles_upload.zip (pristup B1).
Velik fajl (~GB) - za Databricks radije prebaci data/tiles na Volume umesto zip-a."""
import os, glob, zipfile
from scripts import config

TILES = config.TILES; DSD = config.DATASET_DIR
ZIP = config.ZIP_TILES

npys = glob.glob(TILES + r"\*.npy")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for f in npys:
        z.write(f, "tiles/" + os.path.basename(f))
    z.write(config.TILES_INDEX, "tiles_index.csv")
    z.write(config.NASELJE_TABLE, "naselje_table.parquet")
print(f"WROTE {ZIP} | tiles {len(npys)} | {os.path.getsize(ZIP) / 1e9:.2f} GB")
