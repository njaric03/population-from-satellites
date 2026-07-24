import argparse
import os
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.windows import from_bounds, Window

from scripts import config

CRS_METRI = 32634
OPSEZI = ["B02", "B03", "B04", "B08", "B11", "B12"]
PX = 224                  # strana isecka u pikselima
POLA_STRANE_M = 1120      # pola strane isecka (224 px * 10 m / 2)
REZERVA_M = 1300          # bbox kompozita se prosiri, da rubna naselja imaju pun prozor
REZOLUCIJA_M = 10

SEZONA = ["2024-05-01", "2024-09-30"]   # letnji medijan, bez snega i bez golih krosnji
MAX_OBLACI = 30                          # procenat oblacnosti po snimku
POKUSAJA = 3                             # preuzimanje ume da padne (ChunkedEncodingError)
PAUZA_S = 15

OKRUZI_SUBSET = [0, 10, 12, 19, 22]
OPENEO_HOST = "openeo.dataspace.copernicus.eu"


def ucitaj_naselja() -> gpd.GeoDataFrame:
    """Geometrija naselja sa sifrom okruga, centroidom i labelom."""
    naselja = gpd.read_file(config.NASELJA_GPKG)[
        ["naselje_maticni_broj", "opstina_maticni_broj", "geometry"]]
    okruzi = pyogrio.read_dataframe(config.OPSTINE_GPKG, read_geometry=False)[
        ["opstina_maticni_broj", "okrug_sifra"]]
    naselja = naselja.merge(okruzi, on="opstina_maticni_broj", how="left")
    tabela = pd.read_parquet(config.NASELJE_TABLE)[
        ["naselje_maticni_broj", "cx", "cy", "pop"]]
    return naselja.merge(tabela, on="naselje_maticni_broj", how="inner")


def izaberi_okruge(naselja: gpd.GeoDataFrame, rezim: str) -> list:
    """Okruzi za obradu prema rezimu: najmanji jedan, fiksni podskup ili svi."""
    sve = sorted(naselja.okrug_sifra.dropna().unique().tolist())
    if rezim == "test":
        def povrsina_bboxa(sifra):
            zapad, jug, istok, sever = naselja[naselja.okrug_sifra == sifra].total_bounds
            return (istok - zapad) * (sever - jug)
        return [min(sve, key=povrsina_bboxa)]
    if rezim == "subset":
        return [k for k in OKRUZI_SUBSET if k in set(sve)]
    return sve


def ispravan_tiff(putanja: str) -> bool:
    """Da li se GTiff otvara i cita; prekinut download ostavi fajl koji puca."""
    try:
        with rasterio.open(putanja) as dataset:
            dataset.read(1, window=Window(0, 0, 1, 1))
        return True
    except Exception:
        return False


def povezi_se():
    """openEO veza sa autentikacijom; zove se tek kad kompozit stvarno fali."""
    import openeo
    veza = openeo.connect(OPENEO_HOST)
    veza.authenticate_oidc()
    print("openEO auth ok", flush=True)
    return veza


def preuzmi_kompozit(veza, naselja_okruga: gpd.GeoDataFrame, putanja: str, okrug: int) -> bool:
    """Medijan kompozit okruga preko openEO batch posla. Vraca da li je uspelo.

    Jedan posao po okrugu umesto poziva po naselju: 25 poslova umesto 4720.
    """
    zapad, jug, istok, sever = naselja_okruga.total_bounds
    opseg = {"west": zapad - REZERVA_M, "south": jug - REZERVA_M,
             "east": istok + REZERVA_M, "north": sever + REZERVA_M,
             "crs": f"EPSG:{CRS_METRI}"}
    kocka = (
        veza.load_collection("SENTINEL2_L2A", spatial_extent=opseg,
                             temporal_extent=SEZONA, bands=OPSEZI,
                             max_cloud_cover=MAX_OBLACI)
        .reduce_dimension(dimension="t", reducer="median")
        .resample_spatial(resolution=REZOLUCIJA_M, projection=CRS_METRI)
    )

    for pokusaj in range(POKUSAJA):
        try:
            kocka.execute_batch(putanja, out_format="GTiff", title=f"okrug_{okrug}")
            if ispravan_tiff(putanja):
                return True
            print(f"okrug {okrug}: download nepotpun", flush=True)
        except Exception as greska:
            print(f"okrug {okrug} pokusaj {pokusaj + 1} pao: "
                  f"{type(greska).__name__}: {greska}", flush=True)
        if os.path.exists(putanja):
            os.remove(putanja)
        time.sleep(PAUZA_S)
    return False


def iseci_naselje(kompozit, cx: float, cy: float) -> np.ndarray:
    """Isecak (OPSEZI, PX, PX) oko centroida, dopunjen nulama do pune velicine."""
    prozor = from_bounds(cx - POLA_STRANE_M, cy - POLA_STRANE_M,
                         cx + POLA_STRANE_M, cy + POLA_STRANE_M, kompozit.transform)
    procitano = kompozit.read(window=prozor, boundless=True, fill_value=0).astype("int16")
    isecak = np.zeros((len(OPSEZI), PX, PX), dtype="int16")
    visina, sirina = min(PX, procitano.shape[1]), min(PX, procitano.shape[2])
    isecak[:, :visina, :sirina] = procitano[:, :visina, :sirina]
    return isecak


def iseci_okrug(putanja_kompozita: str, naselja_okruga: gpd.GeoDataFrame,
                gotova: set[int]) -> list[dict]:
    """Isecci za sva naselja okruga koja jos nisu u indeksu."""
    redovi = []
    with rasterio.open(putanja_kompozita) as kompozit:
        for _, naselje in naselja_okruga.iterrows():
            maticni_broj = int(naselje.naselje_maticni_broj)
            if maticni_broj in gotova:
                continue
            isecak = iseci_naselje(kompozit, float(naselje.cx), float(naselje.cy))
            np.save(os.path.join(config.CUTOUTS, f"{maticni_broj}.npy"), isecak)
            redovi.append({
                "naselje_maticni_broj": maticni_broj,
                "pop": int(naselje["pop"]),
                "shape": "x".join(map(str, isecak.shape)),
                "empty_frac": round(float((isecak == 0).all(axis=0).mean()), 3),
            })
    return redovi


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Sentinel-2 isecci po naselju, preko openEO kompozita po okrugu.")
    parser.add_argument("rezim", nargs="?", default="test",
                        choices=["test", "subset", "full"],
                        help="test = 1 najmanji okrug, subset = 5 okruga, full = svi")
    rezim = parser.parse_args().rezim

    config.obezbedi(config.CUTOUTS, config.OKRUG_COMP)
    naselja = ucitaj_naselja()
    okruzi = izaberi_okruge(naselja, rezim)
    print("okruzi za obradu:", [int(k) for k in okruzi], flush=True)

    gotova = set()
    if os.path.exists(config.CUTOUTS_INDEX):
        gotova = set(pd.read_csv(config.CUTOUTS_INDEX).naselje_maticni_broj)
    print("vec gotovo naselja:", len(gotova), flush=True)

    veza = None
    ukupno, pocetak = 0, time.time()
    for sifra in okruzi:
        okrug = int(sifra)
        naselja_okruga = naselja[naselja.okrug_sifra == sifra]
        putanja = os.path.join(config.OKRUG_COMP, f"okrug_{okrug}.tiff")
        merenje = time.time()

        if os.path.exists(putanja) and not ispravan_tiff(putanja):
            print(f"okrug {okrug}: neispravan kompozit -> brisem", flush=True)
            os.remove(putanja)
        if not os.path.exists(putanja):
            if veza is None:            # veza se otvara tek kad neki kompozit fali
                veza = povezi_se()
            if not preuzmi_kompozit(veza, naselja_okruga, putanja, okrug):
                print(f"okrug {okrug} PRESKOCEN ({POKUSAJA} pokusaja)", flush=True)
                continue
            print(f"okrug {okrug}: kompozit {time.time() - merenje:.0f}s, "
                  f"{os.path.getsize(putanja) / 1e6:.0f}MB", flush=True)

        redovi = iseci_okrug(putanja, naselja_okruga, gotova)
        if redovi:
            # indeks se dopisuje po okrugu, pa prekid ne gubi vec obradjene okruge
            pd.DataFrame(redovi).to_csv(
                config.CUTOUTS_INDEX, mode="a",
                header=not os.path.exists(config.CUTOUTS_INDEX),
                index=False, encoding="utf-8-sig")
            gotova.update(red["naselje_maticni_broj"] for red in redovi)
            ukupno += len(redovi)
        print(f"okrug {okrug}: iseceno {len(redovi)} | ukupno {ukupno} "
              f"| t={time.time() - pocetak:.0f}s", flush=True)

    print(f"DONE: {ukupno} novih isecaka u {time.time() - pocetak:.0f}s", flush=True)


if __name__ == "__main__":
    main()
