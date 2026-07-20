# Databricks notebook source
# MAGIC %md
# MAGIC # EDA — Procena populacije / izgrađenosti iz satelitskih snimaka (Srbija)
# MAGIC
# MAGIC Provera da li podaci **pasuju** za DL projekat (CNN + transfer learning).
# MAGIC Izvori (svi besplatni): **GeoSrbija RPJ** (geometrija naselja/opština/krugova), **RZS Popis 2022**
# MAGIC (populacija po naseljima), **Overture** (footprinti), **Sentinel-2 L2A** preko openEO (snimci).
# MAGIC
# MAGIC **Pillari koje proveravamo:** geometrija · labele (populacija) · footprinti (izgrađenost) · snimci.
# MAGIC

# COMMAND ----------

import os, re
import numpy as np, pandas as pd, geopandas as gpd, pyogrio, openpyxl
import matplotlib.pyplot as plt
%matplotlib inline

BASE = "../data" if os.path.isdir("../data") else "data"
NAS = BASE + r"\naselja\naselje.gpkg"; OPS = BASE + r"\opstine\opstine.gpkg"
PK  = BASE + r"\popisnikrugovi\popisni_krug.gpkg"; XLSX = BASE + r"\rzs\ukupno_stanovnika_naselja.xlsx"
EDA = BASE + r"\eda"; os.makedirs(EDA, exist_ok=True)

def n_strip(s):
    s = re.sub(r"\s*\(.*?\)\s*", " ", str(s))
    s = re.sub(r"\s+", " ", s.upper().replace("\n", " ")).strip()
    return re.sub(r"^ГРАД\s+", "", s)
def n_plain(s):
    return re.sub(r"\s+", " ", str(s).upper().replace("\n", " ")).strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Prostorne jedinice (RPJ)
# MAGIC Broj jedinica, CRS, površina (iz geometrije), MAUP rizik. Napomena: RPJ OpenData export ima SAMO geometriju — sve statističke kolone su nule.

# COMMAND ----------

nas = gpd.read_file(NAS)
ops = pyogrio.read_dataframe(OPS, read_geometry=False)
pk  = pyogrio.read_dataframe(PK,  read_geometry=False)
nas["area_km2"] = nas.geometry.area / 1e6
print("naselje:", len(nas), "| opštine sa naseljima:", nas.opstina_maticni_broj.nunique(), "| CRS:", nas.crs)
print("opstine rows:", len(ops), "| popisni krug:", len(pk))
print("area km2 -> total %.0f (=Srbija bez KiM)  p50 %.1f  max %.1f" % (
    nas.area_km2.sum(), nas.area_km2.median(), nas.area_km2.max()))
print("\nMAUP rizik (najveća naselja = jedna labela preko ogromne površine):")
display(nas.nlargest(5, "area_km2")[["naselje_ime", "opstina_ime", "area_km2"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Labele — populacija (RZS 2022) → geometrija
# MAGIC RZS xlsx je hijerarhijski po imenu (bez šifre). Parsiramo po `indent` nivou, spajamo 2-stepno + 4 ručna ispravka. **Validacija: zbir = nacionalni total 6.646.833.**

# COMMAND ----------

ws = openpyxl.load_workbook(XLSX, data_only=True)["Sheet1"]
rows, cur = [], None
for r in range(1, ws.max_row + 1):
    a = ws.cell(r, 1); name = a.value
    if not name or not str(name).strip(): continue
    nm = str(name).split("\n")[0].strip()
    if nm in ("Градска", "Остала"): continue
    ind = a.alignment.indent or 0
    if ind == 2: cur = nm
    elif ind == 4: rows.append((cur, nm, ws.cell(r, 3).value))
rzs = pd.DataFrame(rows, columns=["opstina", "naselje", "pop"])
rzs["pop"] = pd.to_numeric(rzs["pop"].replace("-", 0), errors="coerce")
rzs["k_op"] = rzs.opstina.map(n_strip); rzs["k_na"] = rzs.naselje.map(n_plain); rzs["k_pl"] = rzs.naselje.map(n_plain)

g = pyogrio.read_dataframe(NAS, read_geometry=False)[["naselje_maticni_broj","naselje_ime","opstina_ime"]].copy()
g["k_op"] = g.opstina_ime.map(n_strip); g["k_na"] = g.naselje_ime.map(n_plain); g["k_pl"] = g.naselje_ime.map(n_plain)

g = g.merge(rzs.drop_duplicates(["k_op","k_na"])[["k_op","k_na","pop"]], on=["k_op","k_na"], how="left")
g["stage"] = g["pop"].notna().astype(int)
mk = set(zip(g.loc[g["pop"].notna(),"k_op"], g.loc[g["pop"].notna(),"k_na"]))
ru = rzs[~rzs.apply(lambda x:(x.k_op,x.k_na) in mk, axis=1)]
rc = ru.k_pl.value_counts(); uq = set(rc[rc==1].index)
gc = g.loc[g["pop"].isna(),"k_pl"].value_counts(); ug = set(gc[gc==1].index)
key2 = ru[ru.k_pl.isin(uq)].drop_duplicates("k_pl").set_index("k_pl")["pop"]
m2 = g["pop"].isna() & g.k_pl.isin(uq) & g.k_pl.isin(ug)
g.loc[m2,"pop"] = g.loc[m2,"k_pl"].map(key2); g.loc[m2,"stage"] = 2
MANUAL = {("КАЊИЖА","ЗИМОНИЋ"):175, ("СРЕМСКА МИТРОВИЦА","ЗАСАВИЦА I"):652,
          ("СРЕМСКА МИТРОВИЦА","ЗАСАВИЦА II"):532, ("ПРОКУПЉЕ","БУКОЛОРАМ"):2}
for (op,na),val in MANUAL.items():
    sel = (g["k_op"]==n_strip(op)) & (g["naselje_ime"].str.upper()==na) & (g["pop"].isna())
    g.loc[sel,"pop"] = val; g.loc[sel,"stage"] = 3
tot = int(g["pop"].notna().sum())
print("matched %d/%d = %.2f%%" % (tot, len(g), tot/len(g)*100))
print("pop sum %d  (RZS nacionalni total 6646833)  -> %s" % (
    int(g["pop"].sum()), "RECONCILED ✓" if int(g["pop"].sum())==6646833 else "check"))
out = g[g["pop"].notna()].copy(); out["pop"] = out["pop"].astype(int)
out[["naselje_maticni_broj","naselje_ime","opstina_ime","pop","stage"]].to_csv(
    BASE + r"\rzs\naselje_pop_final.csv", index=False, encoding="utf-8-sig")
print("WROTE naselje_pop_final.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Distribucija populacije
# MAGIC Jaka desna iskošenost → `log1p(pop)` cilj. Depopulacija: naselja sa ~0 stanovnika.

# COMMAND ----------

pop = pd.read_csv(BASE + r"\rzs\naselje_pop_final.csv"); p = pop["pop"]
print("median %d  mean %.0f  max %d  zeros %d  <=10 %d" % (
    p.median(), p.mean(), p.max(), (p==0).sum(), (p<=10).sum()))
print("skew raw %.2f  ->  log1p %.2f" % (p.skew(), np.log1p(p).skew()))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(p, bins=60, color="#c44"); ax[0].set_title("population raw (skew %.1f)" % p.skew())
ax[1].hist(np.log1p(p), bins=60, color="#48c"); ax[1].set_title("log1p(pop) ~ normalno")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Footprinti (Overture) — pokrivenost sela
# MAGIC Provera da li ML footprinti pokrivaju sela (rizik depopulacije). Učitava keširani `rural_footprints.csv` (generiše ga `rural_footprints.py`).

# COMMAND ----------

rf = EDA + r"\rural_footprints.csv"
if os.path.exists(rf):
    rr = pd.read_csv(rf); display(rr)
    print("zero-coverage:", int((rr.buildings==0).sum()), "/", len(rr),
          "| izvori:", rr.top_source.value_counts().to_dict())
    print("NALAZ: footprinti opstaju i posle depopulacije -> izgradjenost != populacija")
    print("  (npr. ДЕЈАНОВАЦ: pop 3, ~100 zgrada). Model mora citati ZAUZETOST iz slike, ne samo broj krovova.")
else:
    print("Pokreni rural_footprints.py da generises rural_footprints.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sentinel-2 sample (openEO)
# MAGIC Dokaz pipeline-a: bezoblačni letnji medijan, 4 opsega, isečak oko centroida. Učitava keširani GTiff; ako ga nema, povlači (auth je tih ako je token keširan).

# COMMAND ----------

import rasterio
TIF = BASE + r"\sentinel\novi_sad_s2.tiff"
if not os.path.exists(TIF):
    import openeo
    con = openeo.connect("openeo.dataspace.copernicus.eu"); con.authenticate_oidc()
    nsg = gpd.read_file(NAS); c = nsg[nsg.naselje_ime=="НОВИ САД"].iloc[0].geometry.centroid; H = 1115
    ext = {"west":c.x-H, "south":c.y-H, "east":c.x+H, "north":c.y+H, "crs":"EPSG:32634"}
    cube = con.load_collection("SENTINEL2_L2A", spatial_extent=ext,
        temporal_extent=["2024-05-01","2024-09-30"], bands=["B02","B03","B04","B08"],
        max_cloud_cover=30).reduce_dimension(dimension="t", reducer="median")
    cube.download(TIF, format="GTiff")
with rasterio.open(TIF) as ds:
    arr = ds.read().astype("float32")
print("shape", arr.shape, "| nan%", round(float(np.isnan(arr).mean()*100), 1), "| crs EPSG:32634")
def st(x):
    lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98); return np.clip((x-lo)/(hi-lo+1e-6), 0, 1)
rgb = np.dstack([st(arr[2]), st(arr[1]), st(arr[0])])
plt.figure(figsize=(4.5, 4.5)); plt.imshow(rgb); plt.axis("off")
plt.title("Novi Sad — S2 RGB median 2024 (224px @10m)"); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verdikt
# MAGIC
# MAGIC | Pillar | Status | Dokaz |
# MAGIC |---|---|---|
# MAGIC | Geometrija | ✅ | 4.721 naselja / 168 opština, EPSG:32634, površina = Srbija bez KiM |
# MAGIC | Labele | ✅ | RZS join 99.98%, zbir = 6.646.833 (nacionalni total) |
# MAGIC | Footprinti | ✅ | Overture pokriva i sela (ML), 0 praznih; izgrađenost ≠ populacija u depopulaciji |
# MAGIC | Snimci | ✅ | openEO S2 isečak, bezoblačan medijan, bez rupa |
# MAGIC
# MAGIC **Podaci na disku ≈ 0.7 GB.** Pun MVP (isečci 1 god, 6 opsega) ≈ 5–8 GB.
# MAGIC
# MAGIC **Sledeće (Faza 2 — gradnja dataseta):** centroid+label tabela → footprinti po naselju (izgrađenost cilj) →
# MAGIC Sentinel isečci loop (4.720 × 224px × 6 opsega) → GroupKFold po opštini.
# MAGIC