"""Spakuj plocice + index + tabelu u tiles_upload.zip (pristup B1).
Velik fajl (~GB) - za Databricks radije prebaci data/tiles na Volume umesto zip-a."""
import os, glob, zipfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "data")
TILES = BASE + r"\tiles"; DSD = BASE + r"\dataset"
ZIP = os.path.join(ROOT, "tiles_upload.zip")

npys = glob.glob(TILES + r"\*.npy")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for f in npys:
        z.write(f, "tiles/" + os.path.basename(f))
    z.write(DSD + r"\tiles_index.csv", "tiles_index.csv")
    z.write(DSD + r"\naselje_table.parquet", "naselje_table.parquet")
print(f"WROTE {ZIP} | tiles {len(npys)} | {os.path.getsize(ZIP) / 1e9:.2f} GB")
