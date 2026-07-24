import glob
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.windows import from_bounds

from scripts import config

CRS_METRI = 32634
CRS_STEPENI = 4326
OPSEGA = 6                # Sentinel-2 kanala u kompozitu
PX = 224                  # strana plocice u pikselima
KORAK_M = 2240.0          # 224 px * 10 m: disjunktne plocice, suma plocica = naselje
POLA_M = KORAK_M / 2


def ucitaj_naselja(okruzi_sa_kompozitom: set[int]) -> gpd.GeoDataFrame:
    """Naselja sa geometrijom, centroidom i labelom, samo iz pokrivenih okruga."""
    naselja = gpd.read_file(config.NASELJA_GPKG)[
        ["naselje_maticni_broj", "opstina_maticni_broj", "geometry"]]
    okruzi = pyogrio.read_dataframe(config.OPSTINE_GPKG, read_geometry=False)[
        ["opstina_maticni_broj", "okrug_sifra"]]
    naselja = naselja.merge(okruzi, on="opstina_maticni_broj", how="left")

    tabela = pd.read_parquet(config.NASELJE_TABLE)[
        ["naselje_maticni_broj", "cx", "cy", "pop"]]
    naselja = naselja.merge(tabela, on="naselje_maticni_broj", how="inner")
    return naselja[naselja.okrug_sifra.isin(okruzi_sa_kompozitom)]


def iseci_plocicu(kompozit, tx: float, ty: float) -> np.ndarray:
    """Plocica (OPSEGA, PX, PX) oko (tx, ty), dopunjena nulama do pune velicine.

    Na rubu okruga prozor izlazi iz kompozita, pa ``boundless`` cita nule, a
    dopuna pokriva slucaj kada procitani blok ispadne manji od PX.
    """
    prozor = from_bounds(tx - POLA_M, ty - POLA_M, tx + POLA_M, ty + POLA_M,
                         kompozit.transform)
    procitano = kompozit.read(window=prozor, boundless=True, fill_value=0).astype("int16")
    plocica = np.zeros((OPSEGA, PX, PX), dtype="int16")
    visina, sirina = min(PX, procitano.shape[1]), min(PX, procitano.shape[2])
    plocica[:, :visina, :sirina] = procitano[:, :visina, :sirina]
    return plocica


def centroidi_zgrada(zgrade: gpd.GeoDataFrame, prostorni_indeks, poligon) -> np.ndarray:
    """Koordinate centroida zgrada koje seku dato naselje, kao (N, 2) niz."""
    u_naselju = zgrade.iloc[list(prostorni_indeks.query(poligon, predicate="intersects"))]
    if not len(u_naselju):
        return np.empty((0, 2))
    return np.array([(g.x, g.y) for g in u_naselju.geometry.centroid])


def raspon_plocica(poligon, cx: float, cy: float) -> tuple[range, range]:
    """Indeksi plocica (i, j) koji pokrivaju bounding box naselja, sa rezervom."""
    minx, miny, maxx, maxy = poligon.bounds
    i_od = int(np.floor((minx - cx) / KORAK_M)) - 1
    i_do = int(np.ceil((maxx - cx) / KORAK_M)) + 1
    j_od = int(np.floor((miny - cy) / KORAK_M)) - 1
    j_do = int(np.ceil((maxy - cy) / KORAK_M)) + 1
    return range(i_od, i_do + 1), range(j_od, j_do + 1)


def ima_zgradu(centroidi: np.ndarray, tx: float, ty: float) -> bool:
    """Da li bar jedan centroid zgrade pada u plocicu (prazne njive se preskacu)."""
    return bool(((centroidi[:, 0] >= tx - POLA_M) & (centroidi[:, 0] < tx + POLA_M) &
                 (centroidi[:, 1] >= ty - POLA_M) & (centroidi[:, 1] < ty + POLA_M)).any())


def plocice_naselja(kompozit, naselje, centroidi: np.ndarray) -> list[dict]:
    """Iseci i snimi sve plocice jednog naselja; vrati redove za indeks.

    Naselje bez ijedne zgrade (ili ono kome nijedna plocica nije prosla) dobija
    jednu centralnu plocicu, da ne ispadne iz skupa.
    """
    maticni_broj = int(naselje.naselje_maticni_broj)
    populacija = int(naselje["pop"])
    cx, cy = float(naselje.cx), float(naselje.cy)
    redovi = []

    i_opseg, j_opseg = raspon_plocica(naselje.geometry, cx, cy)
    for i in i_opseg:
        for j in j_opseg:
            tx, ty = cx + i * KORAK_M, cy + j * KORAK_M
            if len(centroidi):
                if not ima_zgradu(centroidi, tx, ty):
                    continue
            elif not (i == 0 and j == 0):
                continue                      # bez zgrada: samo centralna plocica
            ime = f"{maticni_broj}_{i}_{j}.npy"
            np.save(os.path.join(config.TILES, ime), iseci_plocicu(kompozit, tx, ty))
            redovi.append({"path": ime, "naselje_maticni_broj": maticni_broj,
                           "pop": populacija})

    if not redovi:
        ime = f"{maticni_broj}_0_0.npy"
        np.save(os.path.join(config.TILES, ime), iseci_plocicu(kompozit, cx, cy))
        redovi.append({"path": ime, "naselje_maticni_broj": maticni_broj,
                       "pop": populacija})
    return redovi


def ucitaj_zgrade(okrug: int) -> gpd.GeoDataFrame:
    """Overture otisci jednog okruga u metarskom CRS-u."""
    zgrade = gpd.read_parquet(os.path.join(config.OVERTURE_OKRUG, f"okrug_{okrug}.parquet"))
    if zgrade.crs is None:
        zgrade = zgrade.set_crs(CRS_STEPENI)
    return zgrade.to_crs(CRS_METRI)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.obezbedi(config.TILES)

    kompoziti = {int(os.path.basename(f).split("_")[1].split(".")[0])
                 for f in glob.glob(os.path.join(config.OKRUG_COMP, "okrug_*.tiff"))}
    print("okruzi sa kompozitom:", sorted(kompoziti))

    naselja = ucitaj_naselja(kompoziti)
    print("naselja:", len(naselja))

    redovi = []
    for sifra in sorted(naselja.okrug_sifra.unique()):
        okrug = int(sifra)
        zgrade = ucitaj_zgrade(okrug)
        prostorni_indeks = zgrade.sindex
        with rasterio.open(os.path.join(config.OKRUG_COMP, f"okrug_{okrug}.tiff")) as kompozit:
            for _, naselje in naselja[naselja.okrug_sifra == sifra].iterrows():
                centroidi = centroidi_zgrada(zgrade, prostorni_indeks, naselje.geometry)
                redovi.extend(plocice_naselja(kompozit, naselje, centroidi))
        print(f"okrug {okrug}: plocica do sada {len(redovi)}", flush=True)

    indeks = pd.DataFrame(redovi)
    indeks.to_csv(config.TILES_INDEX, index=False, encoding="utf-8-sig")
    po_naselju = indeks.groupby("naselje_maticni_broj").size()
    print(f"DONE: {len(indeks)} plocica / {indeks.naselje_maticni_broj.nunique()} naselja "
          f"| plocica/naselje: med {int(po_naselju.median())} max {int(po_naselju.max())}")


if __name__ == "__main__":
    main()
