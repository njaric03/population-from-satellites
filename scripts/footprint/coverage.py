import glob
import os
import subprocess
import sys

import geopandas as gpd
import pandas as pd

from scripts import config

CRS_METRI = 32634
CRS_STEPENI = 4326
PREUZIMANJE_TIMEOUT_S = 150

# uzorak: obicna sela plus depopulaciona, da se vidi da li otisci nestaju sa ljudima
SELA_RASPON = (50, 800)
SELA_BROJ = 6
SELA_SEED = 42
DEPOPULACIJA_RASPON = (1, 20)
DEPOPULACIJA_BROJ = 2
DEPOPULACIJA_SEED = 7

# opisne kolone koje per_naselje NE koristi; ovde se meri koliko su popunjene
OPISNI_ATRIBUTI = ["num_floors", "height", "subtype", "class", "roof_shape"]


def uzorkuj_sela() -> gpd.GeoDataFrame:
    """Nasumican uzorak sela i depopulacionih naselja, sa fiksnim seed-om."""
    naselja = gpd.read_file(config.NASELJA_GPKG)
    labele = pd.read_csv(config.NASELJE_POP)
    naselja = naselja.merge(labele[["naselje_maticni_broj", "pop"]],
                            on="naselje_maticni_broj", how="inner")
    return pd.concat([
        naselja[naselja["pop"].between(*SELA_RASPON)].sample(SELA_BROJ, random_state=SELA_SEED),
        naselja[naselja["pop"].between(*DEPOPULACIJA_RASPON)].sample(
            DEPOPULACIJA_BROJ, random_state=DEPOPULACIJA_SEED),
    ])


def preuzmi_otiske(granice, putanja: str) -> None:
    """Overture otisci za bbox jednog naselja; preskace ako je vec kesirano."""
    if os.path.exists(putanja):
        return
    zapad, jug, istok, sever = granice
    subprocess.run(
        ["overturemaps", "download", f"--bbox={zapad},{jug},{istok},{sever}",
         "-f", "geoparquet", "--type", "building", "-o", putanja],
        check=True, capture_output=True, timeout=PREUZIMANJE_TIMEOUT_S)


def prebroj_izvore(zgrade: gpd.GeoDataFrame) -> dict[str, int]:
    """Koliko zgrada dolazi iz kog Overture izvora.

    Kolona ``sources`` je lista recnika po zgradi; oblik ume da varira izmedju
    izdanja, pa se preskace sve sto nije recnik sa kljucem ``dataset``.
    """
    if not len(zgrade) or "sources" not in zgrade.columns:
        return {}
    tally: dict[str, int] = {}
    for vrednost in zgrade["sources"]:
        if vrednost is None:
            continue
        for stavka in vrednost:
            if isinstance(stavka, dict) and stavka.get("dataset"):
                izvor = stavka["dataset"]
                tally[izvor] = tally.get(izvor, 0) + 1
    return tally


def pokrivenost_sela() -> pd.DataFrame:
    """Po uzorkovanom naselju: broj zgrada, krovna povrsina i glavni izvor.

    Provera rizika za pristup 2: ako su ML-generisani otisci retki u selima,
    izgradjenost je nepouzdan signal bas tamo gde je populacija najmanja.
    """
    uzorak = uzorkuj_sela()
    u_stepenima = uzorak.to_crs(CRS_STEPENI)
    redovi = []

    for indeks, naselje in uzorak.iterrows():
        poligon = u_stepenima.loc[indeks, "geometry"]
        putanja = os.path.join(config.OVERTURE_RURAL, f"{naselje.naselje_maticni_broj}.parquet")
        try:
            preuzmi_otiske(poligon.bounds, putanja)
            zgrade = gpd.read_parquet(putanja)
            if len(zgrade):
                if zgrade.crs is None:
                    zgrade = zgrade.set_crs(CRS_STEPENI)
                zgrade = zgrade[zgrade.intersects(poligon)]

            broj = len(zgrade)
            povrsina = round(float(zgrade.to_crs(CRS_METRI).area.sum())) if broj else 0
            izvori = prebroj_izvore(zgrade)
            glavni = max(izvori, key=izvori.get) if izvori else "-"
            redovi.append((naselje.naselje_ime, naselje.opstina_ime, int(naselje["pop"]),
                           broj, povrsina, round(broj / max(int(naselje["pop"]), 1), 2), glavni))
        except Exception as greska:
            print(f"{naselje.naselje_ime}: PAO ({type(greska).__name__}: {greska})")
            redovi.append((naselje.naselje_ime, naselje.opstina_ime, int(naselje["pop"]),
                           -1, 0, 0, type(greska).__name__))

    return pd.DataFrame(redovi, columns=["naselje", "opstina", "pop", "buildings",
                                         "roof_m2", "bldg_per_cap", "top_source"])


def popunjenost_atributa() -> pd.DataFrame:
    """Udeo popunjenih opisnih Overture kolona, po okrugu i ukupno.

    Razlog zasto per_naselje racuna atribute iskljucivo iz geometrije: opisni
    atributi su retki i, sto je gore, neravnomerno popunjeni. Gusce mapirani
    okruzi su i urbaniji, pa bi popunjenost bila proksi za urbanost a ne za
    izgradjenost. Vraca prazan DataFrame ako nema kesiranih okruga.
    """
    redovi = []
    for putanja in sorted(glob.glob(os.path.join(config.OVERTURE_OKRUG, "okrug_*.parquet"))):
        try:
            zgrade = pd.read_parquet(putanja, columns=OPISNI_ATRIBUTI)
        except Exception as greska:
            print(f"{os.path.basename(putanja)}: preskocen ({type(greska).__name__})")
            continue
        sifra = os.path.basename(putanja).replace("okrug_", "").replace(".parquet", "")
        red = {"okrug": sifra, "zgrada": len(zgrade)}
        for atribut in OPISNI_ATRIBUTI:
            red[atribut] = round(100 * float(zgrade[atribut].notna().mean()), 2)
        redovi.append(red)

    if not redovi:
        return pd.DataFrame()

    tabela = pd.DataFrame(redovi)
    ukupno_zgrada = tabela["zgrada"].sum()
    zbirni = {"okrug": "UKUPNO", "zgrada": int(ukupno_zgrada)}
    for atribut in OPISNI_ATRIBUTI:
        udeo = (tabela[atribut] / 100 * tabela["zgrada"]).sum() / ukupno_zgrada
        zbirni[atribut] = round(float(udeo * 100), 2)
    return pd.concat([tabela, pd.DataFrame([zbirni])], ignore_index=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # oba izlaza su terminalna (nista ih dalje ne cita kao ulaz) -> results/
    config.obezbedi(config.RESULTS, config.OVERTURE_RURAL)

    sela = pokrivenost_sela()
    print(sela.to_string(index=False))
    sela.to_csv(config.RURAL_FOOTPRINTS, index=False, encoding="utf-8-sig")

    pokrivena = sela[sela.buildings > 0]
    print("\nzero-coverage villages:", int((sela.buildings == 0).sum()), "/", len(sela))
    print("median bldg/cap (covered):",
          round(float(pokrivena.bldg_per_cap.median()), 2) if len(pokrivena) else "n/a")
    print("source tally top:", sela.top_source.value_counts().to_dict())

    atributi = popunjenost_atributa()
    if atributi.empty:
        print("\n(nema kesiranih okruga u", config.OVERTURE_OKRUG, "- pokreni footprint.per_naselje)")
        return
    atributi.to_csv(config.OVERTURE_POPUNJENOST, index=False, encoding="utf-8-sig")
    print(f"\n=== popunjenost Overture atributa (% zgrada), {len(atributi) - 1} okruga ===")
    print(atributi.tail(6).to_string(index=False))
    print(f"WROTE {config.OVERTURE_POPUNJENOST}")


if __name__ == "__main__":
    main()
