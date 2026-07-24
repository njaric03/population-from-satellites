import sys

import geopandas as gpd
import pandas as pd
import pyogrio

from scripts import config

KOORDINATE_CRS = 4326      # WGS84, za lon/lat kolone uz projektovane cx/cy


def ucitaj_naselja() -> gpd.GeoDataFrame:
    # Geometrija naselja spojena sa labelom i sifrom okruga.
    naselja = gpd.read_file(config.NASELJA_GPKG)
    labele = pd.read_csv(config.NASELJE_POP)
    okruzi = pyogrio.read_dataframe(config.OPSTINE_GPKG, read_geometry=False)[
        ["opstina_maticni_broj", "okrug_sifra"]]

    naselja = naselja.merge(labele[["naselje_maticni_broj", "pop", "stage"]],
                            on="naselje_maticni_broj", how="inner")
    naselja = naselja.merge(okruzi, on="opstina_maticni_broj", how="left")
    naselja["area_km2"] = (naselja.geometry.area / 1e6).round(3)
    return naselja


def centroidi(naselja: gpd.GeoDataFrame) -> tuple[gpd.GeoSeries, pd.Series]:
    # Centroid po naselju + maska onih kojima je pao unutar poligona. Kod konkavnog oblika
    # centroid ume da padne izvan naselja, pa bi isecak bio centriran na tudje zemljiste;
    # takvi dobijaju representative_point().
    cent = gpd.GeoSeries(naselja.geometry.centroid, crs=naselja.crs)
    unutra = cent.within(naselja.geometry)
    zamena = naselja.geometry.representative_point()
    ispravljeni = [c if u else z for c, u, z in zip(cent, unutra, zamena)]
    return gpd.GeoSeries(ispravljeni, crs=naselja.crs), unutra


def napravi_tabelu() -> tuple[pd.DataFrame, pd.Series]:
    # Master tabela: labela, grupisanje, povrsina, centroid (+ maska centroida).
    naselja = ucitaj_naselja()
    cent, unutra = centroidi(naselja)
    naselja["cx"] = cent.x.round(1)
    naselja["cy"] = cent.y.round(1)

    stepeni = cent.to_crs(KOORDINATE_CRS)
    naselja["lon"] = stepeni.x.round(6)
    naselja["lat"] = stepeni.y.round(6)

    kolone = ["naselje_maticni_broj", "naselje_ime", "opstina_maticni_broj", "opstina_ime",
              "okrug_sifra", "pop", "stage", "area_km2", "cx", "cy", "lon", "lat"]
    return pd.DataFrame(naselja[kolone]), unutra


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.obezbedi(config.DATASET_DIR)

    tab, unutra = napravi_tabelu()
    tab.to_parquet(config.NASELJE_TABLE, index=False)
    tab.to_csv(config.NASELJE_TABLE_CSV, index=False, encoding="utf-8-sig")

    print("rows:", len(tab), "| pop sum:", int(tab["pop"].sum()),
          "| centroid izvan poligona popravljeno:", int((~unutra).sum()))
    print("opstine:", tab.opstina_maticni_broj.nunique(),
          "| okruzi:", tab.okrug_sifra.nunique(),
          "| missing okrug:", int(tab.okrug_sifra.isna().sum()))
    print("pop: min", int(tab["pop"].min()),
          "p50", int(tab["pop"].median()),
          "max", int(tab["pop"].max()))
    print("\nhead:")
    print(tab.head(6).to_string(index=False))
    print("\nWROTE", config.NASELJE_TABLE)


if __name__ == "__main__":
    main()
