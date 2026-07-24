import re
import sys

import pandas as pd, openpyxl, pyogrio

from scripts import config

# RPJ i RZS ponegde pisu isto naselje razlicito, pa ih automatsko spajanje ne
# uhvati. Mapira se (opstina, RPJ ime) -> RZS ime; populacija se i dalje cita
# iz popisa, ne upisuje rucno, pa ne moze da zastari ako se izvor promeni.
RUCNI_CROSSWALK = {
    ("КАЊИЖА", "ЗИМОНИЋ"):                "ВОЈВОДА ЗИМОНИЋ",
    ("СРЕМСКА МИТРОВИЦА", "ЗАСАВИЦА I"):  "ЗАСАВИЦА 1",
    ("СРЕМСКА МИТРОВИЦА", "ЗАСАВИЦА II"): "ЗАСАВИЦА 2",
    ("ПРОКУПЉЕ", "БУКОЛОРАМ"):            "БУКУЛОРАМ",      # О umesto У
}

# RPJ jedinica koja nema popisni par (artefakt registra) - ostaje nespojena
BEZ_POPISA = {("ЦРНА ТРАВА", "ГРАДСКА")}


def n_strip(s) -> str:
    """Kljuc opstine: velika slova, bez zagrada i bez prefiksa ГРАД."""
    s = re.sub(r"\s*\(.*?\)\s*", " ", str(s))
    s = re.sub(r"\s+", " ", s.upper().replace("\n", " ")).strip()
    return re.sub(r"^ГРАД\s+", "", s)


def n_plain(s) -> str:
    """Kljuc naselja: velika slova, jedan razmak; zagrade ostaju."""
    return re.sub(r"\s+", " ", str(s).upper().replace("\n", " ")).strip()


def ucitaj_rzs() -> pd.DataFrame:
    """Popis 2022 (.xlsx) -> DataFrame (opstina, naselje, pop, k_op, k_na).

    Tabela nema sifre, hijerarhija je samo u uvlacenju celije: indent 2 je
    opstina, indent 4 naselje. Redovi "Градска"/"Остала" su medjuzbirovi.
    """
    ws = openpyxl.load_workbook(config.RZS_XLSX, data_only=True)["Sheet1"]
    rows, cur = [], None
    for r in range(1, ws.max_row + 1):
        a = ws.cell(r, 1)
        ime = a.value
        if not ime or not str(ime).strip():
            continue
        nm = str(ime).split("\n")[0].strip()
        if nm in ("Градска", "Остала"):
            continue
        ind = a.alignment.indent or 0
        if ind == 2:
            cur = nm
        elif ind == 4:
            rows.append((cur, nm, ws.cell(r, 3).value))

    rzs = pd.DataFrame(rows, columns=["opstina", "naselje", "pop"])
    rzs["pop"] = pd.to_numeric(rzs["pop"].mask(rzs["pop"] == "-", 0), errors="coerce")
    rzs["k_op"] = rzs.opstina.map(n_strip)
    rzs["k_na"] = rzs.naselje.map(n_plain)
    return rzs


def ucitaj_rpj() -> pd.DataFrame:
    """GeoSrbija RPJ -> maticni broj i imena naselja (bez geometrije)."""
    g = pyogrio.read_dataframe(config.NASELJA_GPKG, read_geometry=False)[
        ["naselje_maticni_broj", "naselje_ime", "opstina_ime"]].copy()
    g["k_op"] = g.opstina_ime.map(n_strip)
    g["k_na"] = g.naselje_ime.map(n_plain)
    return g


def spoji_labele(rzs: pd.DataFrame = None, rpj: pd.DataFrame = None) -> pd.DataFrame:
    """RPJ tabela sa dodatim kolonama ``pop`` i ``stage``, u tri koraka.

    * stage 1 - par (opstina, naselje)
    * stage 2 - samo ime naselja, i to samo ako je jedinstveno na obe strane
    * stage 3 - ``RUCNI_CROSSWALK`` za pravopisne varijante

    ``stage`` 0 znaci nespojeno; ``pop`` je tada NaN. Ne pise nista na disk
    i ne stampa - dijagnostiku radi pozivalac (``main`` ili notebook).
    """
    rzs = ucitaj_rzs() if rzs is None else rzs
    g = (ucitaj_rpj() if rpj is None else rpj).copy()

    # stage1: (opstina, naselje)
    par = rzs.drop_duplicates(["k_op", "k_na"])[["k_op", "k_na", "pop"]]
    g = g.merge(par, on=["k_op", "k_na"], how="left")
    g["stage"] = g["pop"].notna().map({True: 1, False: 0})

    # stage2: po imenu naselja, bez opstine. Samo ako je ime jedinstveno na obe
    # strane, inace bi se spojila dva razlicita naselja istog imena.
    spojeni = set(zip(g.loc[g["pop"].notna(), "k_op"], g.loc[g["pop"].notna(), "k_na"]))
    preostali = rzs[~rzs.apply(lambda x: (x.k_op, x.k_na) in spojeni, axis=1)]
    br_rzs = preostali.k_na.value_counts()
    br_rpj = g.loc[g["pop"].isna(), "k_na"].value_counts()
    jedinstveni = set(br_rzs[br_rzs == 1].index) & set(br_rpj[br_rpj == 1].index)
    po_imenu = preostali[preostali.k_na.isin(jedinstveni)].drop_duplicates("k_na") \
                        .set_index("k_na")["pop"]
    sel = g["pop"].isna() & g.k_na.isin(jedinstveni)
    g.loc[sel, "pop"] = g.loc[sel, "k_na"].map(po_imenu)
    g.loc[sel, "stage"] = 2

    # stage3: pravopisne varijante; populacija se cita iz RZS, ne prepisuje
    rzs_pop = rzs.drop_duplicates(["k_op", "k_na"]).set_index(["k_op", "k_na"])["pop"]
    for (opstina, rpj_ime), rzs_ime in RUCNI_CROSSWALK.items():
        kljuc = (n_strip(opstina), n_plain(rzs_ime))
        if kljuc not in rzs_pop.index:
            raise KeyError(f"crosswalk: {kljuc} nema u RZS tabeli - proveri RUCNI_CROSSWALK")
        sel = (g.k_op == n_strip(opstina)) & (g.k_na == n_plain(rpj_ime)) & g["pop"].isna()
        g.loc[sel, "pop"] = rzs_pop.loc[kljuc]
        g.loc[sel, "stage"] = 3

    return g


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rzs, rpj = ucitaj_rzs(), ucitaj_rpj()
    print("duplikati (opstina, naselje) - RPJ:", int(rpj.duplicated(["k_op", "k_na"]).sum()),
          "| RZS:", int(rzs.duplicated(["k_op", "k_na"]).sum()))

    g = spoji_labele(rzs, rpj)
    for s, opis in ((1, "par (opstina, naselje)"), (2, "jedinstveno ime"), (3, "rucni crosswalk")):
        print(f"stage{s} {opis:24s}: {int((g.stage == s).sum())}")

    spojeno = int(g["pop"].notna().sum())
    print(f"ukupno spojeno: {spojeno}/{len(g)} = {spojeno / len(g) * 100:.2f}%")

    nespojeni = g[g["pop"].isna()]
    if len(nespojeni):
        neocekivani = [t for t in zip(nespojeni.k_op, nespojeni.k_na) if t not in BEZ_POPISA]
        print("nespojeno:", len(nespojeni), "| od toga neocekivano:", len(neocekivani))
        print(nespojeni[["opstina_ime", "naselje_ime"]].to_string(index=False))

    out = g[g["pop"].notna()][
        ["naselje_maticni_broj", "naselje_ime", "opstina_ime", "pop", "stage"]].copy()
    out["pop"] = out["pop"].astype(int)
    out.to_csv(config.NASELJE_POP, index=False, encoding="utf-8-sig")
    print("WROTE naselje_pop_final.csv | pop sum:", int(out["pop"].sum()),
          "| zeros:", int((out["pop"] == 0).sum()))


if __name__ == "__main__":
    main()
