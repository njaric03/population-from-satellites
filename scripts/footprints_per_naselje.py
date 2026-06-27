"""Faza 2.2 - footprinti po naselju (Overture), petlja po okrugu (memorija niska).
Po naselju: broj zgrada, krovna povrsina (m2), zapremina-proxy (povrsina*spratnost).
Izlaz: data/dataset/naselje_footprints.csv  (+ spaja na naselje_table).
"""
import sys, os, subprocess
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, geopandas as gpd, numpy as np, pyogrio

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT = BASE + r"\dataset"; TMP = BASE + r"\overture_okrug"; os.makedirs(TMP, exist_ok=True)

nas = gpd.read_file(BASE + r"\naselja\naselje.gpkg")[
    ["naselje_maticni_broj", "opstina_maticni_broj", "geometry"]]
ops = pyogrio.read_dataframe(BASE + r"\opstine\opstine.gpkg",
                             read_geometry=False)[["opstina_maticni_broj", "okrug_sifra"]]
nas = nas.merge(ops, on="opstina_maticni_broj", how="left")     # 32634

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
        fl = b["num_floors"] if "num_floors" in b.columns else pd.Series(np.nan, index=b.index)
        h = b["height"] if "height" in b.columns else pd.Series(np.nan, index=b.index)
        floors = fl.fillna((h / 3).round()).fillna(1).clip(lower=1)
        b["vol"] = b["ba"] * floors
        pts = gpd.GeoDataFrame(b[["ba", "vol"]].copy(), geometry=b.geometry.centroid, crs=32634)
        j = gpd.sjoin(pts, sub, predicate="within", how="inner")
        agg = j.groupby("naselje_maticni_broj").agg(
            n_buildings=("ba", "size"), roof_area_m2=("ba", "sum"), vol_proxy=("vol", "sum"))
        parts.append(agg.reset_index())
        print(f"okrug {int(k)}: zgrada {len(b)} -> naselja pokrivena {len(agg)}")
    except Exception as ex:
        print(f"okrug {int(k)}: ERR {str(ex)[:60]}")

fps = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
    columns=["naselje_maticni_broj", "n_buildings", "roof_area_m2", "vol_proxy"])

tab = pd.read_parquet(OUT + r"\naselje_table.parquet")
m = tab.merge(fps, on="naselje_maticni_broj", how="left")
for c in ["n_buildings", "roof_area_m2", "vol_proxy"]:
    m[c] = m[c].fillna(0)
m["roof_area_m2"] = m["roof_area_m2"].round(0); m["vol_proxy"] = m["vol_proxy"].round(0)
m.to_csv(OUT + r"\naselje_footprints.csv", index=False, encoding="utf-8-sig")
m.to_parquet(OUT + r"\naselje_footprints.parquet", index=False)

zero = int((m.n_buildings == 0).sum())
ok = m[m.n_buildings > 0]
cor = np.corrcoef(np.log1p(ok["pop"]), np.log1p(ok["roof_area_m2"]))[0, 1] if len(ok) else float("nan")
print("\n=== REZIME ===")
print("naselja:", len(m), "| sa 0 zgrada:", zero, "| ukupno zgrada:", int(m.n_buildings.sum()))
print("log(pop) vs log(roof_area) corr:", round(float(cor), 3))
print("bldg/cap median:", round(float((ok.n_buildings / ok["pop"].clip(lower=1)).median()), 2))
print("WROTE naselje_footprints.csv")
