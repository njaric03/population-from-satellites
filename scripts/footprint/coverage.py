"""Provera pokrivenosti footprintima u SELIMA (Overture).
Rizik: ML footprinti retki u selima -> izgradjenost nepouzdana ba za depopulacione slucajeve.
Uzorak: 6 sela pop 50-800 + 2 depopulaciona (pop 1-20). Broj zgrada, krovna povrsina, izvori.
"""
import sys, os, subprocess
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, geopandas as gpd

from scripts import config

# rezultat je terminalni artefakt (nista ga dalje ne konzumira kao ulaz) -> results/
OUT = config.RESULTS
TMP = config.OVERTURE_RURAL      # kes Overture preuzimanja
config.obezbedi(OUT, TMP)

g = gpd.read_file(config.NASELJA_GPKG)
pop = pd.read_csv(config.NASELJE_POP)
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
df.to_csv(os.path.join(OUT, "rural_footprints.csv"), index=False, encoding="utf-8-sig")
ok = df[df.buildings > 0]
print("\nzero-coverage villages:", int((df.buildings == 0).sum()), "/", len(df))
print("median bldg/cap (covered):", round(float(ok.bldg_per_cap.median()), 2) if len(ok) else "n/a")
print("source tally top:", df.top_source.value_counts().to_dict())


# === popunjenost Overture atributa po okrugu ===
# Razlog zasto per_naselje racuna atribute iskljucivo iz geometrije: opisni atributi
# su retki i, sto je gore, neravnomerno popunjeni — gusce mapirani okruzi su i
# urbaniji, pa bi popunjenost bila proksi za urbanost a ne za izgradjenost.
import glob

ATR = ["num_floors", "height", "subtype", "class", "roof_shape"]
redovi = []
for f in sorted(glob.glob(os.path.join(config.OVERTURE_OKRUG, "okrug_*.parquet"))):
    try:
        b = pd.read_parquet(f, columns=ATR)
    except Exception:
        continue
    red = {"okrug": os.path.basename(f).replace("okrug_", "").replace(".parquet", ""),
           "zgrada": len(b)}
    for a in ATR:
        red[a] = round(100 * float(b[a].notna().mean()), 2)
    redovi.append(red)

if redovi:
    atr_df = pd.DataFrame(redovi)
    uk = atr_df["zgrada"].sum()
    ukupno = {"okrug": "UKUPNO", "zgrada": int(uk)}
    for a in ATR:
        ukupno[a] = round(float((atr_df[a] / 100 * atr_df["zgrada"]).sum() / uk * 100), 2)
    atr_df = pd.concat([atr_df, pd.DataFrame([ukupno])], ignore_index=True)
    put = os.path.join(OUT, "overture_popunjenost_atributa.csv")
    atr_df.to_csv(put, index=False, encoding="utf-8-sig")
    print(f"\n=== popunjenost Overture atributa (% zgrada), {len(redovi)} okruga ===")
    print(atr_df.tail(6).to_string(index=False))
    print(f"WROTE {put}")
else:
    print("\n(nema kesiranih okruga u", config.OVERTURE_OKRUG, "- pokreni footprint.per_naselje)")
