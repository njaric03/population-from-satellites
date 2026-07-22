"""Pristup 2 - rasterizacija otisaka zgrada (Overture) u kanale po naselju.
Isti 224px @10m prozor oko centroida kao satelitski cutout (poravnato za fuziju).
Kanali: [0] pokrivenost (udeo celije pod zgradom, 4x supersample), [1] zapreminska gustina (pokrivenost x spratnost).
Izlaz: data/footprint_cutouts/<mb>.npy (2,224,224) float32. Resumable.
"""
import os, glob
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize, MergeAlg
from rasterio.transform import from_origin

from scripts import config

OKR = config.OVERTURE_OKRUG
OUT = config.FOOTPRINT_CUT; config.obezbedi(OUT)
PX, H, SUB = 224, 1120, 4          # 224 px, pola strane 1120 m, 4x supersample (2.5 m)
FPX, RES = PX * SUB, 10.0 / SUB

tab = pd.read_parquet(config.NASELJE_TABLE)   # ima okrug_sifra
imaju_cutout = {int(os.path.splitext(os.path.basename(f))[0]) for f in glob.glob(config.CUTOUTS + r"\*.npy")}
tab = tab[tab.naselje_maticni_broj.isin(imaju_cutout)].copy()
print("naselja za rasterizaciju:", len(tab))

def floors_col(b):
    fl = b["num_floors"] if "num_floors" in b.columns else pd.Series(np.nan, index=b.index)
    h = b["height"] if "height" in b.columns else pd.Series(np.nan, index=b.index)
    return fl.fillna((h / 3).round()).fillna(1).clip(lower=1).astype("float32")

def downsample(fine):                 # (FPX,FPX) -> (PX,PX) prosek
    return fine.reshape(PX, SUB, PX, SUB).mean(axis=(1, 3)).astype("float32")

done = 0
for k in sorted(tab.okrug_sifra.dropna().unique()):
    fp = f"{OKR}\\okrug_{int(k)}.parquet"
    if not os.path.exists(fp):
        print(f"okrug {int(k)}: nema parquet, preskacem"); continue
    b = gpd.read_parquet(fp)
    if b.crs is None: b = b.set_crs(4326)
    b = b.to_crs(32634)
    b["fl"] = floors_col(b)
    sidx = b.sindex
    sub_tab = tab[tab.okrug_sifra == k]
    for _, r in sub_tab.iterrows():
        mb = int(r.naselje_maticni_broj)
        outp = f"{OUT}\\{mb}.npy"
        if os.path.exists(outp): continue
        cx, cy = float(r.cx), float(r.cy)
        west, north = cx - H, cy + H
        wbox = box(cx - H, cy - H, cx + H, cy + H)
        cand = b.iloc[list(sidx.query(wbox, predicate="intersects"))]
        tr = from_origin(west, north, RES, RES)
        if len(cand):
            cov = rasterize(((g, 1.0) for g in cand.geometry), out_shape=(FPX, FPX),
                            transform=tr, fill=0, all_touched=False, dtype="float32")
            vol = rasterize(zip(cand.geometry, cand.fl), out_shape=(FPX, FPX), transform=tr,
                            fill=0, merge_alg=MergeAlg.replace, dtype="float32")
            arr = np.stack([downsample(cov), downsample(vol)])
        else:
            arr = np.zeros((2, PX, PX), dtype="float32")
        np.save(outp, arr)
        done += 1
    print(f"okrug {int(k)}: gotovo, ukupno {done}", flush=True)

print(f"DONE: {done} footprint rastera u {OUT}")
