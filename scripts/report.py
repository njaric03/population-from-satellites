import base64
import io
import json
import re
import sys

import pandas as pd

from scripts import config

# Prilozi za izvestaj se vade iz izvrsenih notebooka: trening je isao na
# Databricksu, pa su grafovi i tabele metrika ugradjeni u .ipynb izlaze, a ne na
# disku. Ova skripta ih raspakuje u figures/ i results/, da se vide bez
# pristupa MLflow eksperimentu.

# u 05 se za svaki pristup redom upisuju sirovo pa obe kalibracije (petlja u run())
VARIJANTE = ("sirovo", "kalibrisano: ridge", "kalibrisano: isotonic")
KOLONE = ["oof_r2_log", "oof_medape", "oof_wmape", "oof_bias", "oof_kalib_nagib",
          "oof_opstina_r2_log_bez_top2"]


def ucitaj(ime: str) -> dict:
    with open(config.NOTEBOOKS / f"{ime}.ipynb", encoding="utf-8") as f:
        return json.load(f)


def izlazi(nb: dict, tip: str):
    # (indeks cella, podatak) za svaki izlaz trazenog mime tipa
    for i, cell in enumerate(nb.get("cells", [])):
        for izlaz in cell.get("outputs", []):
            podatak = izlaz.get("data", {}).get(tip)
            if podatak:
                # nbformat cuva izlaz kao string ili kao listu linija
                yield i, podatak if isinstance(podatak, str) else "".join(podatak)


def tekst(nb: dict) -> str:
    return "\n".join("".join(izlaz.get("text", []))
                     for cell in nb.get("cells", []) for izlaz in cell.get("outputs", []))


def izvuci_slike(imena: list[str]) -> int:
    # ugradjeni PNG-ovi (cv_summary_figure i ostali grafovi) -> figures/
    broj = 0
    for ime in imena:
        nb = ucitaj(ime)
        for redni, (_, podatak) in enumerate(izlazi(nb, "image/png"), start=1):
            put = config.FIGURES / f"{ime}_{redni}.png"
            put.write_bytes(base64.b64decode(podatak))
            print("WROTE", put.name)
            broj += 1
    return broj


def tabele(nb: dict) -> list[pd.DataFrame]:
    return [pd.read_html(io.StringIO(html))[0] for _, html in izlazi(nb, "text/html")
            if "<table" in html]


def broj_naselja(nb: dict, uzorak: str) -> int:
    nadjeno = re.search(uzorak, tekst(nb))
    return int(nadjeno.group(1)) if nadjeno else -1


def presek(nb: dict) -> int:
    # notebooci stampaju "presek: N naselja" pre evaluacije na preseku
    return broj_naselja(nb, r"presek: (\d+) naselja")


def izvor(nb: dict, pocetak: str) -> str:
    # izvor prve celije koja sadrzi dati tekst; sluzi za imena redova koja
    # Databricks display izbaci iz tabele
    for cell in nb.get("cells", []):
        kod = "".join(cell.get("source", []))
        if pocetak in kod:
            return kod
    return ""


def metrike_fuzija() -> pd.DataFrame:
    # 05 prikazuje poredjenje bez indeksa (Databricks display), pa se imena redova
    # rekonstruisu iz redosleda: po pristupu tri varijante, pa fuzija na kraju.
    nb = ucitaj("05_fusion_train")
    tabela = tabele(nb)[0]
    pristupi = re.search(r"pristupi: \[(.*?)\]", tekst(nb))
    pristupi = re.findall(r"'(.*?)'", pristupi.group(1)) if pristupi else []

    imena = [f"{p} ({v})" for p in pristupi for v in VARIJANTE] + ["fuzija (stacking)"]
    if len(imena) != len(tabela):
        raise RuntimeError(f"05: {len(tabela)} redova, rekonstruisano {len(imena)} imena")

    tabela.insert(0, "pristup", imena)
    tabela = tabela[tabela.pristup.str.contains(r"\(sirovo\)|fuzija")].copy()
    tabela["pristup"] = tabela.pristup.str.replace(" (sirovo)", "", regex=False)
    tabela["n_naselja"] = presek(nb)
    tabela["izvor"] = "05_fusion_train"
    return tabela


def metrike_otisci() -> list[pd.DataFrame]:
    """03 ima do dve tabele; uzima se svaka koja u svesci ima izvrsen izlaz.

    Prva (tabelarni modeli na celom skupu) ide kroz Databricks display, koji brise
    indeks, pa se imena redova rekonstruisu iz izvora celije -- isto kao u
    metrike_fuzija. Druga (raster na preseku) ide kroz IPython display, koji indeks
    cuva, ali su joj kolone viseslojne.
    """
    nb = ucitaj("03_footprint_train")
    izlaz = []
    for tabela in tabele(nb):
        bez_indeksa = not isinstance(tabela.columns[0], tuple)
        if bez_indeksa:
            imena = re.findall(r'\(\s*"(.*?)",\s*oof_\w+\s*\)', izvor(nb, "TABELARNI = ["))
            if len(imena) != len(tabela):
                raise RuntimeError(
                    f"03: {len(tabela)} redova, rekonstruisano {len(imena)} imena")
            tabela.insert(0, "pristup", imena)
            tabela["n_naselja"] = broj_naselja(nb, r"tabelarni skup: (\d+) naselja")
        else:
            tabela.columns = ["pristup"] + [k[0] for k in tabela.columns[1:]]
            tabela["n_naselja"] = presek(nb)
        tabela["izvor"] = "03_footprint_train"
        izlaz.append(tabela)
    if not izlaz:
        raise RuntimeError("03: nijedna tabela nema izvrsen izlaz")
    return izlaz


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.ensure_dirs(config.RESULTS, config.FIGURES)

    broj = izvuci_slike(["01_eda", "02_tiles_train", "03_footprint_train",
                         "04_multimodal_train", "05_fusion_train"])
    print(f"\n{broj} slika -> {config.FIGURES}\n")

    metrike = pd.concat([metrike_fuzija(), *metrike_otisci()], ignore_index=True)
    metrike = metrike[["pristup", "n_naselja", *KOLONE, "izvor"]].round(4)
    metrike.to_csv(config.METRIKE, index=False, encoding="utf-8-sig")
    print(metrike.to_string(index=False))
    print("\nWROTE", config.METRIKE)


if __name__ == "__main__":
    main()
