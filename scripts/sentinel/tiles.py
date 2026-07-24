import os, glob
import numpy as np, pandas as pd, geopandas as gpd, pyogrio, rasterio
from rasterio.windows import from_bounds

from scripts import config

COMP = config.OKRUG_COMP; OKR = config.OVERTURE_OKRUG
OUT = config.TILES; config.obezbedi(OUT)
PX, STEP = 224, 2240.0; H = STEP / 2   # disjunktne plocice 2.24 km, suma plocica = naselje

have_comp = {int(os.path.basename(f).split("_")[1].split(".")[0]) for f in glob.glob(COMP + r"\okrug_*.tiff")}
print("okruzi sa kompozitom:", sorted(have_comp))

nas = gpd.read_file(config.NASELJA_GPKG)[["naselje_maticni_broj", "opstina_maticni_broj", "geometry"]]
ops = pyogrio.read_dataframe(config.OPSTINE_GPKG, read_geometry=False)[["opstina_maticni_broj", "okrug_sifra"]]
nas = nas.merge(ops, on="opstina_maticni_broj", how="left")
tab = pd.read_parquet(config.NASELJE_TABLE)[["naselje_maticni_broj", "cx", "cy", "pop"]]
nas = nas.merge(tab, on="naselje_maticni_broj", how="inner")
nas = nas[nas.okrug_sifra.isin(have_comp)]
print("naselja:", len(nas))

rows = []
for k in sorted(nas.okrug_sifra.unique()):
    comp = f"{COMP}\\okrug_{int(k)}.tiff"
    b = gpd.read_parquet(f"{OKR}\\okrug_{int(k)}.parquet")
    if b.crs is None: b = b.set_crs(4326)
    b = b.to_crs(32634); bsidx = b.sindex
    sub = nas[nas.okrug_sifra == k]
    with rasterio.open(comp) as ds:
        for _, r in sub.iterrows():
            mb = int(r.naselje_maticni_broj); poly = r.geometry
            cx, cy = float(r.cx), float(r.cy)
            nb = b.iloc[list(bsidx.query(poly, predicate="intersects"))]
            bc = np.array([(g.x, g.y) for g in nb.geometry.centroid]) if len(nb) else np.empty((0, 2))
            minx, miny, maxx, maxy = poly.bounds
            i_lo, i_hi = int(np.floor((minx - cx) / STEP)) - 1, int(np.ceil((maxx - cx) / STEP)) + 1
            j_lo, j_hi = int(np.floor((miny - cy) / STEP)) - 1, int(np.ceil((maxy - cy) / STEP)) + 1
            kept = 0
            for i in range(i_lo, i_hi + 1):
                for j in range(j_lo, j_hi + 1):
                    tx, ty = cx + i * STEP, cy + j * STEP
                    if len(bc):
                        # zadrzi samo plocice sa bar jednom zgradom (prazne njive preskoci)
                        inside = ((bc[:, 0] >= tx - H) & (bc[:, 0] < tx + H) &
                                  (bc[:, 1] >= ty - H) & (bc[:, 1] < ty + H)).any()
                        if not inside: continue
                    elif not (i == 0 and j == 0):
                        continue                              # bez zgrada -> samo centralna plocica
                    win = from_bounds(tx - H, ty - H, tx + H, ty + H, ds.transform)
                    a = ds.read(window=win, boundless=True, fill_value=0).astype("int16")
                    out = np.zeros((6, PX, PX), dtype="int16")
                    hh, ww = min(PX, a.shape[1]), min(PX, a.shape[2]); out[:, :hh, :ww] = a[:, :hh, :ww]
                    name = f"{mb}_{i}_{j}.npy"
                    np.save(f"{OUT}\\{name}", out)
                    rows.append({"path": name, "naselje_maticni_broj": mb, "pop": int(r["pop"])})
                    kept += 1
            if kept == 0:                                     # fallback ako nista nije zadrzano
                tx, ty = cx, cy
                win = from_bounds(tx - H, ty - H, tx + H, ty + H, ds.transform)
                a = ds.read(window=win, boundless=True, fill_value=0).astype("int16")
                out = np.zeros((6, PX, PX), dtype="int16")
                hh, ww = min(PX, a.shape[1]), min(PX, a.shape[2]); out[:, :hh, :ww] = a[:, :hh, :ww]
                np.save(f"{OUT}\\{mb}_0_0.npy", out)
                rows.append({"path": f"{mb}_0_0.npy", "naselje_maticni_broj": mb, "pop": int(r["pop"])})
    print(f"okrug {int(k)}: plocica do sada {len(rows)}", flush=True)

idx = pd.DataFrame(rows)
idx.to_csv(config.TILES_INDEX, index=False, encoding="utf-8-sig")
per = idx.groupby("naselje_maticni_broj").size()
print(f"DONE: {len(idx)} plocica / {idx.naselje_maticni_broj.nunique()} naselja "
      f"| plocica/naselje: med {int(per.median())} max {int(per.max())}")
