"""Faza 2.3 (batch) - Sentinel isečci preko openEO BATCH poslova po okrugu.
Po okrugu: jedan medijan kompozit (6 opsega, 10m, 2024, oblaci<30) -> lokalno
isecanje 224px prozora oko centroida svakog naselja. 25 poslova umesto 4720 poziva.
Pokretanje: python -m scripts.sentinel.cutouts test   (1 najmanji okrug)  |  ... subset  |  ... full
Resumable: preskace naselja iz index.csv i okruge sa postojecim kompozitom.
"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, geopandas as gpd, pyogrio, rasterio, openeo
from rasterio.windows import from_bounds, Window

from scripts import config

CUT = config.CUTOUTS; COMP = config.OKRUG_COMP
os.makedirs(CUT, exist_ok=True); os.makedirs(COMP, exist_ok=True)
IDX = config.CUTOUTS_INDEX
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]; PX = 224; H = 1120; PAD = 1300

mode = sys.argv[1] if len(sys.argv) > 1 else "test"
nas = gpd.read_file(config.NASELJA_GPKG)[["naselje_maticni_broj", "opstina_maticni_broj", "geometry"]]
ops = pyogrio.read_dataframe(config.OPSTINE_GPKG,
                             read_geometry=False)[["opstina_maticni_broj", "okrug_sifra"]]
nas = nas.merge(ops, on="opstina_maticni_broj", how="left")
tab = pd.read_parquet(config.NASELJE_TABLE)[["naselje_maticni_broj", "cx", "cy", "pop"]]
nas = nas.merge(tab, on="naselje_maticni_broj", how="inner")

okr = sorted(nas.okrug_sifra.dropna().unique().tolist())
if mode == "test":
    okr = [min(okr, key=lambda k: (lambda b: (b[2]-b[0])*(b[3]-b[1]))(nas[nas.okrug_sifra == k].total_bounds))]
elif mode == "subset":
    okr = [k for k in [0, 10, 12, 19, 22] if k in set(okr)]
print("okruzi za obradu:", [int(x) for x in okr], flush=True)

con = openeo.connect("openeo.dataspace.copernicus.eu"); con.authenticate_oidc()
done = set(pd.read_csv(IDX).naselje_maticni_broj) if os.path.exists(IDX) else set()
print("auth ok | vec gotovo naselja:", len(done), flush=True)

def valid_tiff(p):
    try:
        with rasterio.open(p) as d:
            d.read(1, window=Window(0, 0, 1, 1))
        return True
    except Exception:
        return False

total, t0 = 0, time.time()
for k in okr:
    sub = nas[nas.okrug_sifra == k]
    w, s, e, n = sub.total_bounds
    ext = {"west": w-PAD, "south": s-PAD, "east": e+PAD, "north": n+PAD, "crs": "EPSG:32634"}
    comp = f"{COMP}\\okrug_{int(k)}.tiff"; t1 = time.time()
    if os.path.exists(comp) and not valid_tiff(comp):
        print(f"okrug {int(k)}: neispravan kompozit -> brisem", flush=True); os.remove(comp)
    if not os.path.exists(comp):
        cube = con.load_collection("SENTINEL2_L2A", spatial_extent=ext,
            temporal_extent=["2024-05-01", "2024-09-30"], bands=BANDS,
            max_cloud_cover=30).reduce_dimension(dimension="t", reducer="median") \
            .resample_spatial(resolution=10, projection=32634)
        for att in range(3):                                   # retry: download moze pasti (ChunkedEncodingError)
            try:
                cube.execute_batch(comp, out_format="GTiff", title=f"okrug_{int(k)}")
                if valid_tiff(comp): break
                print(f"okrug {int(k)}: download nepotpun", flush=True)
            except Exception as ex:
                print(f"okrug {int(k)} pokusaj {att+1} pao: {str(ex)[:70]}", flush=True)
            if os.path.exists(comp): os.remove(comp)
            time.sleep(15)
        if not (os.path.exists(comp) and valid_tiff(comp)):
            print(f"okrug {int(k)} PRESKOCEN (3 pokusaja)", flush=True); continue
    print(f"okrug {int(k)}: kompozit {time.time()-t1:.0f}s, {os.path.getsize(comp)/1e6:.0f}MB", flush=True)
    orows = []
    with rasterio.open(comp) as ds:
        for _, r in sub.iterrows():
            mb = int(r.naselje_maticni_broj)
            if mb in done: continue
            cx, cy = float(r.cx), float(r.cy)
            win = from_bounds(cx-H, cy-H, cx+H, cy+H, ds.transform)
            a = ds.read(window=win, boundless=True, fill_value=0).astype("int16")
            out = np.zeros((len(BANDS), PX, PX), dtype="int16")
            hh, ww = min(PX, a.shape[1]), min(PX, a.shape[2]); out[:, :hh, :ww] = a[:, :hh, :ww]
            np.save(f"{CUT}\\{mb}.npy", out)
            orows.append({"naselje_maticni_broj": mb, "pop": int(r["pop"]),
                          "shape": "x".join(map(str, out.shape)),
                          "empty_frac": round(float((out == 0).all(axis=0).mean()), 3)})
    if orows:                                                  # upis index-a PO OKRUGU (crash-safe)
        pd.DataFrame(orows).to_csv(IDX, mode="a", header=not os.path.exists(IDX),
                                   index=False, encoding="utf-8-sig")
        done.update(d["naselje_maticni_broj"] for d in orows); total += len(orows)
    print(f"okrug {int(k)}: isečeno {len(orows)} | ukupno {total} | t={time.time()-t0:.0f}s", flush=True)
print(f"DONE: {total} novih isečaka u {time.time()-t0:.0f}s", flush=True)
