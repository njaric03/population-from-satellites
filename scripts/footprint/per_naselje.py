import sys, os, subprocess
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, geopandas as gpd, numpy as np, pyogrio
from scipy.spatial import cKDTree

from scripts import config

OUT = config.DATASET_DIR; TMP = config.OVERTURE_OKRUG; config.obezbedi(OUT, TMP)

nas = gpd.read_file(config.NASELJA_GPKG)[
    ["naselje_maticni_broj", "opstina_maticni_broj", "geometry"]]
ops = pyogrio.read_dataframe(config.OPSTINE_GPKG,
                             read_geometry=False)[["opstina_maticni_broj", "okrug_sifra"]]
nas = nas.merge(ops, on="opstina_maticni_broj", how="left")

parts = []
okruzi = sorted(nas.okrug_sifra.dropna().unique().tolist())
print("okruga:", len(okruzi))
for k in okruzi:
    sub = nas[nas.okrug_sifra == k][["naselje_maticni_broj", "geometry"]]
    w, s, e, n = sub.to_crs(4326).total_bounds
    fp = f"{TMP}\\okrug_{int(k)}.parquet"
    try:
        if not os.path.exists(fp):
            subprocess.run(["overturemaps", "download", f"--bbox={w},{s},{e},{n}",
                            "-f", "geoparquet", "--type", "building", "-o", fp],
                           check=True, capture_output=True, timeout=600)
        b = gpd.read_parquet(fp)
        if len(b) == 0:
            print(f"okrug {int(k)}: 0 zgrada"); continue
        if b.crs is None:
            b = b.set_crs(4326)
        b = b.to_crs(32634)
        b["ba"] = b.geometry.area
        b["compact"] = (4 * np.pi * b["ba"]) / b.geometry.length.clip(lower=1.0) ** 2
        b["velika"] = (b["ba"] > 200).astype("float32")   # proksi za stambeni blok

        # raspored: rastojanje do najblize zgrade i broj suseda u 50 m, racunato
        # nad SVIM zgradama okruga (susedstvo ne staje na granici naselja)
        cent = b.geometry.centroid
        xy = np.c_[cent.x.values, cent.y.values]
        drvo = cKDTree(xy)
        b["nn_dist"] = drvo.query(xy, k=2)[0][:, 1] if len(b) > 1 else 0.0
        b["n_50m"] = drvo.query_ball_point(xy, 50, return_length=True) - 1

        kol = ["ba", "compact", "velika", "nn_dist", "n_50m"]
        pts = gpd.GeoDataFrame(b[kol].copy(), geometry=cent, crs=32634)
        j = gpd.sjoin(pts, sub, predicate="within", how="inner")
        agg = j.groupby("naselje_maticni_broj").agg(
            n_buildings=("ba", "size"),
            roof_area_m2=("ba", "sum"),
            mean_bsize=("ba", "mean"),
            median_bsize=("ba", "median"),
            std_bsize=("ba", "std"),
            p90_bsize=("ba", lambda s: s.quantile(0.9)),
            mean_compact=("compact", "mean"),
            udeo_velikih=("velika", "mean"),
            mean_nn_dist=("nn_dist", "mean"),
            median_nn_dist=("nn_dist", "median"),
            mean_n_50m=("n_50m", "mean"),
        )
        parts.append(agg.reset_index())
        print(f"okrug {int(k)}: zgrada {len(b)} -> naselja pokrivena {len(agg)}")
    except Exception as ex:
        print(f"okrug {int(k)}: ERR {str(ex)[:60]}")

ATRIBUTI = config.FP_AGREGIRANI      # spisak atributa i zasto bas ti: config.py

fps = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
    columns=["naselje_maticni_broj", *ATRIBUTI])

tab = pd.read_parquet(config.NASELJE_TABLE)
m = tab.merge(fps, on="naselje_maticni_broj", how="left")
# naselja bez ijedne zgrade: kolicine su 0, a atributi oblika/rasporeda nedefinisani.
# 0 je i tu bezbedno jer n_buildings=0 nosi tu informaciju modelu.
for c in ATRIBUTI:
    m[c] = m[c].fillna(0)
m["roof_area_m2"] = m["roof_area_m2"].round(0)

# izvedeni odnosi (gustine po povrsini naselja iz GeoSrbija geometrije)
m["building_density"] = m["n_buildings"] / m["area_km2"].clip(lower=1e-6)
m["built_fraction"] = (m["roof_area_m2"] / (m["area_km2"] * 1e6).clip(lower=1.0)).clip(0, 1)
m.to_csv(os.path.join(OUT, "naselje_footprints.csv"), index=False, encoding="utf-8-sig")
m.to_parquet(config.NASELJE_FOOTPRINTS, index=False)

zero = int((m.n_buildings == 0).sum())
ok = m[m.n_buildings > 0]
cor = np.corrcoef(np.log1p(ok["pop"]), np.log1p(ok["roof_area_m2"]))[0, 1] if len(ok) else float("nan")
print("\n=== REZIME ===")
print("naselja:", len(m), "| sa 0 zgrada:", zero, "| ukupno zgrada:", int(m.n_buildings.sum()))
print("log(pop) vs log(roof_area) corr:", round(float(cor), 3))
print("bldg/cap median:", round(float((ok.n_buildings / ok["pop"].clip(lower=1)).median()), 2))
print("\nkorelacija sa log1p(pop) po atributu (naselja sa >0 zgrada):")
for c in [*ATRIBUTI, "building_density", "built_fraction"]:
    if len(ok) and ok[c].std() > 0:
        r = np.corrcoef(np.log1p(ok["pop"]), np.log1p(ok[c].clip(lower=0)))[0, 1]
        print(f"  {c:18s} {r:+.3f}")
print(f"\nWROTE {config.NASELJE_FOOTPRINTS}")
