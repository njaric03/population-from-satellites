import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.spatial import cKDTree

from scripts import config

CRS_METRI = 32634          # UTM 34N, da povrsine i rastojanja budu u metrima
CRS_STEPENI = 4326         # WGS84, Overture ocekuje bbox u stepenima
VELIKA_ZGRADA_M2 = 200     # prag za "veliku" zgradu, proksi za stambeni blok
SUSEDSTVO_M = 50           # poluprecnik u kome se broje susedne zgrade
KVANTIL_KRUPNIH = 0.9      # p90 velicine zgrade
PREUZIMANJE_TIMEOUT_S = 600

ATRIBUTI = config.FP_AGREGIRANI      # spisak atributa i zasto bas ti: config.py


def ucitaj_naselja() -> gpd.GeoDataFrame:
    # Geometrija naselja sa pridruzenom sifrom okruga (petlja ide po okrugu).
    naselja = gpd.read_file(config.NASELJA_GPKG)[
        ["naselje_maticni_broj", "opstina_maticni_broj", "geometry"]]
    okruzi = pyogrio.read_dataframe(config.OPSTINE_GPKG, read_geometry=False)[
        ["opstina_maticni_broj", "okrug_sifra"]]
    return naselja.merge(okruzi, on="opstina_maticni_broj", how="left")


def preuzmi_otiske(naselja_okruga: gpd.GeoDataFrame, putanja: Path) -> None:
    # Skine Overture otiske za bbox okruga, ako fajl vec ne postoji.
    if putanja.exists():
        return
    zapad, jug, istok, sever = naselja_okruga.to_crs(CRS_STEPENI).total_bounds
    subprocess.run(
        ["overturemaps", "download", f"--bbox={zapad},{jug},{istok},{sever}",
         "-f", "geoparquet", "--type", "building", "-o", putanja],
        check=True, capture_output=True, timeout=PREUZIMANJE_TIMEOUT_S)


def geometrijski_atributi(zgrade: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Po zgradi: povrsina, kompaktnost, oznaka velike, mere rasporeda. Rastojanje i broj
    # suseda idu nad SVIM zgradama okruga, ne po naselju: inace bi zgrada tik uz granicu
    # naselja ispala usamljena.
    zgrade = zgrade.to_crs(CRS_METRI)
    zgrade["ba"] = zgrade.geometry.area
    zgrade["compact"] = (4 * np.pi * zgrade["ba"]) / zgrade.geometry.length.clip(lower=1.0) ** 2
    zgrade["velika"] = (zgrade["ba"] > VELIKA_ZGRADA_M2).astype("float32")

    centroidi = zgrade.geometry.centroid
    tacke = np.c_[centroidi.x.values, centroidi.y.values]
    drvo = cKDTree(tacke)
    zgrade["nn_dist"] = drvo.query(tacke, k=2)[0][:, 1] if len(zgrade) > 1 else 0.0
    zgrade["n_50m"] = drvo.query_ball_point(tacke, SUSEDSTVO_M, return_length=True) - 1
    zgrade["centroid"] = centroidi
    return zgrade


def agregiraj_po_naselju(zgrade: gpd.GeoDataFrame,
                         naselja_okruga: gpd.GeoDataFrame) -> pd.DataFrame:
    # Zgrade svrstane u naselja po centroidu, pa svedene na atribute naselja.
    kolone = ["ba", "compact", "velika", "nn_dist", "n_50m"]
    tacke = gpd.GeoDataFrame(zgrade[kolone].copy(),
                             geometry=zgrade["centroid"], crs=CRS_METRI)
    u_naselju = gpd.sjoin(tacke, naselja_okruga, predicate="within", how="inner")
    return u_naselju.groupby("naselje_maticni_broj").agg(
        n_buildings=("ba", "size"),
        roof_area_m2=("ba", "sum"),
        mean_bsize=("ba", "mean"),
        median_bsize=("ba", "median"),
        std_bsize=("ba", "std"),
        p90_bsize=("ba", lambda s: s.quantile(KVANTIL_KRUPNIH)),
        mean_compact=("compact", "mean"),
        udeo_velikih=("velika", "mean"),
        mean_nn_dist=("nn_dist", "mean"),
        median_nn_dist=("nn_dist", "median"),
        mean_n_50m=("n_50m", "mean"),
    ).reset_index()


def atributi_po_okruzima(naselja: gpd.GeoDataFrame) -> tuple[pd.DataFrame, list[int]]:
    # (atributi, sifre palih okruga). Ide okrug po okrug, sve zgrade ne staju u RAM.
    delovi, pali = [], []
    sifre = sorted(naselja.okrug_sifra.dropna().unique().tolist())
    print("okruga:", len(sifre))

    for sifra in sifre:
        okrug = int(sifra)
        naselja_okruga = naselja[naselja.okrug_sifra == sifra][
            ["naselje_maticni_broj", "geometry"]]
        putanja = config.OVERTURE_OKRUG / f"okrug_{okrug}.parquet"
        try:
            preuzmi_otiske(naselja_okruga, putanja)
            zgrade = gpd.read_parquet(putanja)
            if len(zgrade) == 0:
                print(f"okrug {okrug}: 0 zgrada")
                continue
            if zgrade.crs is None:
                zgrade = zgrade.set_crs(CRS_STEPENI)

            zgrade = geometrijski_atributi(zgrade)
            agregat = agregiraj_po_naselju(zgrade, naselja_okruga)
            delovi.append(agregat)
            print(f"okrug {okrug}: zgrada {len(zgrade)} -> naselja pokrivena {len(agregat)}")
        except Exception as greska:
            pali.append(okrug)
            print(f"okrug {okrug}: PAO ({type(greska).__name__}: {greska})")

    if delovi:
        return pd.concat(delovi, ignore_index=True), pali
    return pd.DataFrame(columns=["naselje_maticni_broj", *ATRIBUTI]), pali


def spoji_sa_tabelom(atributi: pd.DataFrame) -> pd.DataFrame:
    # Atributi pridruzeni master tabeli, sa izvedenim gustinama.
    tabela = pd.read_parquet(config.NASELJE_TABLE)
    spojeno = tabela.merge(atributi, on="naselje_maticni_broj", how="left")

    # Naselja bez ijedne zgrade: kolicine su 0, a atributi oblika i rasporeda
    # nedefinisani. 0 je i tu bezbedno jer n_buildings=0 nosi tu informaciju.
    for kolona in ATRIBUTI:
        spojeno[kolona] = spojeno[kolona].fillna(0)
    spojeno["roof_area_m2"] = spojeno["roof_area_m2"].round(0)

    # izvedeni odnosi (gustine po povrsini naselja iz GeoSrbija geometrije)
    spojeno["building_density"] = spojeno["n_buildings"] / spojeno["area_km2"].clip(lower=1e-6)
    spojeno["built_fraction"] = (
        spojeno["roof_area_m2"] / (spojeno["area_km2"] * 1e6).clip(lower=1.0)).clip(0, 1)
    return spojeno


def stampaj_rezime(spojeno: pd.DataFrame) -> None:
    # Korelacije atributa sa populacijom, provera da signal uopste postoji.
    bez_zgrada = int((spojeno.n_buildings == 0).sum())
    sa_zgradama = spojeno[spojeno.n_buildings > 0]
    korelacija = float("nan")
    if len(sa_zgradama):
        korelacija = np.corrcoef(np.log1p(sa_zgradama["pop"]),
                                 np.log1p(sa_zgradama["roof_area_m2"]))[0, 1]

    print("\n=== REZIME ===")
    print("naselja:", len(spojeno), "| sa 0 zgrada:", bez_zgrada,
          "| ukupno zgrada:", int(spojeno.n_buildings.sum()))
    print("log(pop) vs log(roof_area) corr:", round(float(korelacija), 3))
    if len(sa_zgradama):
        po_stanovniku = (sa_zgradama.n_buildings / sa_zgradama["pop"].clip(lower=1)).median()
        print("bldg/cap median:", round(float(po_stanovniku), 2))

    print("\nkorelacija sa log1p(pop) po atributu (naselja sa >0 zgrada):")
    for kolona in [*ATRIBUTI, *config.FP_IZVEDENI]:
        if len(sa_zgradama) and sa_zgradama[kolona].std() > 0:
            r = np.corrcoef(np.log1p(sa_zgradama["pop"]),
                            np.log1p(sa_zgradama[kolona].clip(lower=0)))[0, 1]
            print(f"  {kolona:18s} {r:+.3f}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.obezbedi(config.DATASET_DIR, config.OVERTURE_OKRUG)

    naselja = ucitaj_naselja()
    atributi, pali = atributi_po_okruzima(naselja)
    spojeno = spoji_sa_tabelom(atributi)

    spojeno.to_csv(config.NASELJE_FOOTPRINTS_CSV, index=False, encoding="utf-8-sig")
    spojeno.to_parquet(config.NASELJE_FOOTPRINTS, index=False)

    stampaj_rezime(spojeno)
    if pali:
        print("\nUPOZORENJE: okruzi bez atributa:", pali, "- rezultat je nepotpun")
    print(f"\nWROTE {config.NASELJE_FOOTPRINTS}")


if __name__ == "__main__":
    main()
