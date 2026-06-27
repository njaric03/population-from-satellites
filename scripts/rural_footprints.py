"""Provera pokrivenosti footprintima u SELIMA (Overture).
Rizik: ML footprinti retki u selima -> izgradjenost nepouzdana ba za depopulacione slucajeve.
Uzorak: 6 sela pop 50-800 + 2 depopulaciona (pop 1-20). Broj zgrada, krovna povrsina, izvori.
"""
import sys, os, subprocess
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, geopandas as gpd

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT = BASE + r"\eda"; os.makedirs(OUT, exist_ok=True)
TMP = BASE + r"\overture_rural"; os.makedirs(TMP, exist_ok=True)

g = gpd.read_file(BASE + r"\naselja\naselje.gpkg")
pop = pd.read_csv(BASE + r"\rzs\naselje_pop_final.csv")
g = g.merge(pop[["naselje_maticni_broj", "pop"]], on="naselje_maticni_broj", how="inner")
g4326 = g.to_crs(4326)

samp = pd.concat([
    g[g["pop"].between(50, 800)].sample(6, random_state=42),
    g[g["pop"].between(1, 20)].sample(2, random_state=7),    # depopulacija
])

rows = []
for idx, r in samp.iterrows():
    poly = g4326.loc[idx, "geometry"]; w, s, e, n = poly.bounds
    out = f"{TMP}\\{r.naselje_maticni_broj}.parquet"
    try:
        subprocess.run(["overturemaps", "download", f"--bbox={w},{s},{e},{n}",
                        "-f", "geoparquet", "--type", "building", "-o", out],
                       check=True, capture_output=True, timeout=150)
        b = gpd.read_parquet(out)
        if len(b):
            if b.crs is None: b = b.set_crs(4326)
            b = b[b.intersects(poly)]
        nb = len(b)
        area = round(float(b.to_crs(32634).area.sum())) if nb else 0
        srcs = {}
        if nb and "sources" in b.columns:
            for val in b["sources"]:
                try:
                    for d in val:
                        ds = d.get("dataset") if isinstance(d, dict) else None
                        if ds: srcs[ds] = srcs.get(ds, 0) + 1
                except Exception: pass
        top = max(srcs, key=srcs.get) if srcs else "-"
        rows.append((r.naselje_ime, r.opstina_ime, int(r["pop"]), nb, area,
                     round(nb / max(int(r["pop"]), 1), 2), top))
    except Exception as ex:
        rows.append((r.naselje_ime, r.opstina_ime, int(r["pop"]), -1, 0, 0, str(ex)[:30]))

df = pd.DataFrame(rows, columns=["naselje", "opstina", "pop", "buildings",
                                 "roof_m2", "bldg_per_cap", "top_source"])
print(df.to_string(index=False))
df.to_csv(OUT + r"\rural_footprints.csv", index=False, encoding="utf-8-sig")
ok = df[df.buildings > 0]
print("\nzero-coverage villages:", int((df.buildings == 0).sum()), "/", len(df))
print("median bldg/cap (covered):", round(float(ok.bldg_per_cap.median()), 2) if len(ok) else "n/a")
print("source tally top:", df.top_source.value_counts().to_dict())
