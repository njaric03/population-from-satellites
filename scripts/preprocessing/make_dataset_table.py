"""Faza 2.1 - master tabela naselja: centroid + labela + grupisanje.
Temelj za: Sentinel isecke (cx,cy), footprint rasterizaciju, GroupKFold (opstina/okrug).
Izlaz: data/dataset/naselje_table.parquet (+ .csv pregled).
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import geopandas as gpd, pandas as pd, pyogrio

from scripts import config

OUT = config.DATASET_DIR; config.obezbedi(OUT)

nas = gpd.read_file(config.NASELJA_GPKG)
pop = pd.read_csv(config.NASELJE_POP)
ops = pyogrio.read_dataframe(config.OPSTINE_GPKG,
                             read_geometry=False)[["opstina_maticni_broj", "okrug_sifra"]]

nas = nas.merge(pop[["naselje_maticni_broj", "pop", "stage"]], on="naselje_maticni_broj", how="inner")
nas = nas.merge(ops, on="opstina_maticni_broj", how="left")
nas["area_km2"] = (nas.geometry.area / 1e6).round(3)

cent = gpd.GeoSeries(nas.geometry.centroid, crs=nas.crs)
inside = cent.within(nas.geometry)                     # centroid van poligona? (konkavni)
rep = nas.geometry.representative_point()
cent = gpd.GeoSeries([c if i else r for c, i, r in zip(cent, inside, rep)], crs=nas.crs)
nas["cx"] = cent.x.round(1); nas["cy"] = cent.y.round(1)
c4 = cent.to_crs(4326)
nas["lon"] = c4.x.round(6); nas["lat"] = c4.y.round(6)

cols = ["naselje_maticni_broj", "naselje_ime", "opstina_maticni_broj", "opstina_ime",
        "okrug_sifra", "pop", "stage", "area_km2", "cx", "cy", "lon", "lat"]
tab = pd.DataFrame(nas[cols])
tab.to_parquet(config.NASELJE_TABLE, index=False)
tab.to_csv(config.NASELJE_TABLE_CSV, index=False, encoding="utf-8-sig")

print("rows:", len(tab), "| pop sum:", int(tab["pop"].sum()),
      "| centroid izvan poligona popravljeno:", int((~inside).sum()))
print("opstine:", tab.opstina_maticni_broj.nunique(), "| okruzi:", tab.okrug_sifra.nunique(),
      "| missing okrug:", int(tab.okrug_sifra.isna().sum()))
print("pop: min", int(tab["pop"].min()), "p50", int(tab["pop"].median()), "max", int(tab["pop"].max()))
print("\nhead:")
print(tab.head(6).to_string(index=False))
print("\nWROTE", OUT + r"\naselje_table.parquet")
