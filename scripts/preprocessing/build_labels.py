"""Finalna label tabela: naselje_maticni_broj -> populacija (RZS 2022).
2-stepni join:
  stage1: (opstina, naselje) normalizovano (strip paren + 'ГРАД ' prefiks)
  stage2: za nematchovane -> po jedinstvenom imenu naselja (sa zadrzanom zagradom),
          samo ako je ime jedinstveno na obe strane (bez laznih spojeva).
Cilj ~99.9%. Izlaz: naselje_pop_final.csv
"""
import sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, openpyxl, pyogrio

from scripts import config

XLSX = config.RZS_XLSX
NAS  = config.NASELJA_GPKG

def n_strip(s):
    s = re.sub(r"\s*\(.*?\)\s*", " ", str(s))
    s = re.sub(r"\s+", " ", s.upper().replace("\n", " ")).strip()
    return re.sub(r"^ГРАД\s+", "", s)

def n_plain(s):
    return re.sub(r"\s+", " ", str(s).upper().replace("\n", " ")).strip()

# ---- parse RZS ----
ws = openpyxl.load_workbook(XLSX, data_only=True)["Sheet1"]
rows, cur = [], None
for r in range(1, ws.max_row + 1):
    a = ws.cell(r, 1); name = a.value
    if not name or not str(name).strip():
        continue
    nm = str(name).split("\n")[0].strip()
    if nm in ("Градска", "Остала"):
        continue
    ind = a.alignment.indent or 0
    if ind == 2:
        cur = nm
    elif ind == 4:
        rows.append((cur, nm, ws.cell(r, 3).value))
rzs = pd.DataFrame(rows, columns=["opstina", "naselje", "pop"])
rzs["pop"] = pd.to_numeric(rzs["pop"].replace("-", 0), errors="coerce")
rzs["k_op"] = rzs.opstina.map(n_strip); rzs["k_na"] = rzs.naselje.map(n_plain)
rzs["k_pl"] = rzs.naselje.map(n_plain)

g = pyogrio.read_dataframe(NAS, read_geometry=False)[
    ["naselje_maticni_broj", "naselje_ime", "opstina_ime"]].copy()
g["k_op"] = g.opstina_ime.map(n_strip); g["k_na"] = g.naselje_ime.map(n_plain)
print("RPJ dup (k_op,k_na):", int(g.duplicated(["k_op", "k_na"]).sum()),
      "| RZS dup (k_op,k_na):", int(rzs.duplicated(["k_op", "k_na"]).sum()))
g["k_pl"] = g.naselje_ime.map(n_plain)

# ---- stage1: (opstina, naselje) ----
r1 = rzs.drop_duplicates(["k_op", "k_na"])
g = g.merge(r1[["k_op", "k_na", "pop"]], on=["k_op", "k_na"], how="left")
g["stage"] = g["pop"].notna().map({True: 1, False: 0})
print("stage1 matched:", int(g["pop"].notna().sum()), "/", len(g))

# ---- stage2: unique plain settlement name ----
matched_keys = set(zip(g.loc[g["pop"].notna(), "k_op"], g.loc[g["pop"].notna(), "k_na"]))
rzs_un = rzs[~rzs.apply(lambda x: (x.k_op, x.k_na) in matched_keys, axis=1)]
rc = rzs_un.k_pl.value_counts(); uniq_rzs = set(rc[rc == 1].index)
gc = g.loc[g["pop"].isna(), "k_pl"].value_counts(); uniq_g = set(gc[gc == 1].index)
key2 = rzs_un[rzs_un.k_pl.isin(uniq_rzs)].drop_duplicates("k_pl").set_index("k_pl")["pop"]
mask = g["pop"].isna() & g.k_pl.isin(uniq_rzs) & g.k_pl.isin(uniq_g)
g.loc[mask, "pop"] = g.loc[mask, "k_pl"].map(key2)
g.loc[mask, "stage"] = 2
print("stage2 recovered:", int(mask.sum()))

# stage3: rucni crosswalk za pravopisne varijante (RZS <-> RPJ)
MANUAL = {
    ("КАЊИЖА", "ЗИМОНИЋ"): 175,                 # RZS 'Војвода Зимонић'
    ("СРЕМСКА МИТРОВИЦА", "ЗАСАВИЦА I"): 652,    # RZS 'Засавица 1'
    ("СРЕМСКА МИТРОВИЦА", "ЗАСАВИЦА II"): 532,   # RZS 'Засавица 2'
    ("ПРОКУПЉЕ", "БУКОЛОРАМ"): 2,               # RZS 'Букулорам' (O/У)
}
for (op, na), val in MANUAL.items():
    sel = (g["k_op"] == n_strip(op)) & (g["naselje_ime"].str.upper() == na) & (g["pop"].isna())
    g.loc[sel, "pop"] = val; g.loc[sel, "stage"] = 3
print("stage3 manual:", int((g["stage"] == 3).sum()),
      "| 'ГРАДСКА' (Црна Трава) = RPJ artefakt bez popisa -> iskljucen")

tot = int(g["pop"].notna().sum())
print(f"FINAL matched: {tot}/{len(g)} = {tot/len(g)*100:.2f}%")
resid = g[g["pop"].isna()]
print("still unmatched:", len(resid))
if len(resid):
    print(resid[["opstina_ime", "naselje_ime"]].to_string(index=False))

out = g[g["pop"].notna()][["naselje_maticni_broj", "naselje_ime", "opstina_ime", "pop", "stage"]].copy()
out["pop"] = out["pop"].astype(int)
out.to_csv(config.NASELJE_POP, index=False, encoding="utf-8-sig")
print("WROTE naselje_pop_final.csv | pop sum:", int(out["pop"].sum()),
      "| zeros:", int((out["pop"] == 0).sum()))
