"""Pristup 2b - rasterizacija otisaka zgrada (Overture) u kanale po naselju.
Isti 224px @10m prozor oko centroida kao satelitski cutout (poravnato za fuziju).

Kanali: [0] pokrivenost (udeo celije pod zgradom, 4x supersample),
        [1] gustina zgrada (broj centroida po celiji).

Oba su izvedena iskljucivo iz geometrije. Raniji kanal [1] je bio pokrivenost x
spratnost, ali je num_floors popunjen u 1.17% zgrada a height u 0.04% (mereno
nad kesiranim okruzima, videti footprint/coverage.py), pa je spratnost bila 1
za skoro sve zgrade i kanal [1] je ispadao duplikat kanala [0]. Broj centroida
razdvaja mnogo malih zgrada od nekoliko velikih, sto pokrivenost sama ne vidi.

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

def downsample(fine, kako="mean"):    # (FPX,FPX) -> (PX,PX)
    blok = fine.reshape(PX, SUB, PX, SUB)
    return (blok.mean(axis=(1, 3)) if kako == "mean" else blok.sum(axis=(1, 3))).astype("float32")

done = 0
for k in sorted(tab.okrug_sifra.dropna().unique()):
    fp = f"{OKR}\\okrug_{int(k)}.parquet"
    if not os.path.exists(fp):
        print(f"okrug {int(k)}: nema parquet, preskacem"); continue
    b = gpd.read_parquet(fp)
    if b.crs is None: b = b.set_crs(4326)
    b = b.to_crs(32634)
    b["centroid"] = b.geometry.centroid
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
            # kanal 1: pokrivenost - udeo celije pod zgradom
            cov = rasterize(((g, 1.0) for g in cand.geometry), out_shape=(FPX, FPX),
                            transform=tr, fill=0, all_touched=False, dtype="float32")
            # kanal 2: gustina zgrada - broj centroida po celiji. Razdvaja mnogo malih
            # zgrada od nekoliko velikih, sto pokrivenost sama ne vidi.
            cnt = rasterize(((g, 1.0) for g in cand["centroid"]), out_shape=(FPX, FPX),
                            transform=tr, fill=0, merge_alg=MergeAlg.add, dtype="float32")
            arr = np.stack([downsample(cov, "mean"), downsample(cnt, "sum")])
        else:
            arr = np.zeros((2, PX, PX), dtype="float32")
        np.save(outp, arr)
        done += 1
    print(f"okrug {int(k)}: gotovo, ukupno {done}", flush=True)

print(f"DONE: {done} footprint rastera u {OUT}")
