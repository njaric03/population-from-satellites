import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize, MergeAlg
from rasterio.transform import from_origin
from shapely.geometry import box

from scripts import config

CRS_METRI = 32634
CRS_STEPENI = 4326

# isti 224px @10m prozor oko centroida kao satelitski cutout (poravnato za fuziju)
PX = 224                  # strana rastera u pikselima
POLA_STRANE_M = 1120      # pola strane prozora u metrima (224 px * 10 m / 2)
SUPERSAMPLE = 4           # rasterizuje se 4x finije pa se sazima, da udeo bude gladak
FINI_PX = PX * SUPERSAMPLE
FINI_RES_M = 10.0 / SUPERSAMPLE


def sazmi(fini: np.ndarray, kako: str = "mean") -> np.ndarray:
    # (FINI_PX, FINI_PX) -> (PX, PX): prosek za pokrivenost, suma za brojanje.
    blok = fini.reshape(PX, SUPERSAMPLE, PX, SUPERSAMPLE)
    sazeto = blok.mean(axis=(1, 3)) if kako == "mean" else blok.sum(axis=(1, 3))
    return sazeto.astype("float32")


def rasterizuj_naselje(zgrade: gpd.GeoDataFrame, prostorni_indeks,
                       cx: float, cy: float) -> np.ndarray:
    # Dva kanala oko (cx, cy): udeo celije pod zgradom i broj centroida po celiji. Drugi
    # kanal razdvaja mnogo malih zgrada od nekoliko velikih.
    prozor = box(cx - POLA_STRANE_M, cy - POLA_STRANE_M,
                 cx + POLA_STRANE_M, cy + POLA_STRANE_M)
    kandidati = zgrade.iloc[list(prostorni_indeks.query(prozor, predicate="intersects"))]
    if not len(kandidati):
        return np.zeros((2, PX, PX), dtype="float32")

    transformacija = from_origin(cx - POLA_STRANE_M, cy + POLA_STRANE_M,
                                 FINI_RES_M, FINI_RES_M)
    pokrivenost = rasterize(((g, 1.0) for g in kandidati.geometry),
                            out_shape=(FINI_PX, FINI_PX), transform=transformacija,
                            fill=0, all_touched=False, dtype="float32")
    broj_zgrada = rasterize(((g, 1.0) for g in kandidati["centroid"]),
                            out_shape=(FINI_PX, FINI_PX), transform=transformacija,
                            fill=0, merge_alg=MergeAlg.add, dtype="float32")
    return np.stack([sazmi(pokrivenost, "mean"), sazmi(broj_zgrada, "sum")])


def naselja_za_obradu() -> pd.DataFrame:
    # Naselja sa satelitskim iseckom; bez para fuzija nema sta da spoji.
    tabela = pd.read_parquet(config.NASELJE_TABLE)
    imaju_isecak = {int(f.stem) for f in config.CUTOUTS.glob("*.npy")}
    return tabela[tabela.naselje_maticni_broj.isin(imaju_isecak)].copy()


def ucitaj_zgrade(okrug: int) -> gpd.GeoDataFrame:
    # Overture otisci jednog okruga, u metarskom CRS-u, sa centroidima.
    putanja = config.OVERTURE_OKRUG / f"okrug_{okrug}.parquet"
    zgrade = gpd.read_parquet(putanja)
    if zgrade.crs is None:
        zgrade = zgrade.set_crs(CRS_STEPENI)
    zgrade = zgrade.to_crs(CRS_METRI)
    zgrade["centroid"] = zgrade.geometry.centroid
    return zgrade


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.obezbedi(config.FOOTPRINT_CUT)

    tabela = naselja_za_obradu()
    print("naselja za rasterizaciju:", len(tabela))

    napravljeno = 0
    for sifra in sorted(tabela.okrug_sifra.dropna().unique()):
        okrug = int(sifra)
        if not (config.OVERTURE_OKRUG / f"okrug_{okrug}.parquet").exists():
            print(f"okrug {okrug}: nema parquet, preskacem")
            continue

        zgrade = ucitaj_zgrade(okrug)
        prostorni_indeks = zgrade.sindex
        for _, naselje in tabela[tabela.okrug_sifra == sifra].iterrows():
            maticni_broj = int(naselje.naselje_maticni_broj)
            izlaz = config.FOOTPRINT_CUT / f"{maticni_broj}.npy"
            if izlaz.exists():               # resumable: gotovo se ne racuna ponovo
                continue
            raster = rasterizuj_naselje(zgrade, prostorni_indeks,
                                        float(naselje.cx), float(naselje.cy))
            np.save(izlaz, raster)
            napravljeno += 1
        print(f"okrug {okrug}: gotovo, ukupno {napravljeno}", flush=True)

    print(f"DONE: {napravljeno} footprint rastera u {config.FOOTPRINT_CUT}")


if __name__ == "__main__":
    main()
