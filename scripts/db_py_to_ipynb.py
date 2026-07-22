"""Konverzija Databricks source formata (.py) u Jupyter notebook (.ipynb).

Databricks .py notebook: celije razdvojene sa `# COMMAND ----------`, a
markdown i magic celije su zakomentarisane prefiksom `# MAGIC`. Ovaj skript
vraca ih u pravi .ipynb tako da se notebooci mogu otvoriti u Jupyter/Colab-u
i predati kao .ipynb.

Upotreba:
    python scripts/db_py_to_ipynb.py                 # svi notebooks/*.py
    python scripts/db_py_to_ipynb.py notebooks/EDA.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAZDELNIK = "# COMMAND ----------"
MAGIC = "# MAGIC"
ZAGLAVLJE = "# Databricks notebook source"


def _skini_magic(linije: list[str]) -> list[str]:
    """Ukloni `# MAGIC ` prefiks sa linija magic celije."""
    out = []
    for l in linije:
        if l.startswith(MAGIC + " "):
            out.append(l[len(MAGIC) + 1:])
        elif l.strip() == MAGIC:
            out.append("")
        else:
            out.append(l)
    return out


def _skrati(linije: list[str]) -> list[str]:
    """Skini prazne linije sa pocetka i kraja."""
    while linije and not linije[0].strip():
        linije.pop(0)
    while linije and not linije[-1].strip():
        linije.pop()
    return linije


def _izvor(linije: list[str]) -> list[str]:
    """Lista linija -> nbformat `source` (svaka linija sem poslednje ima \\n)."""
    return [l + "\n" for l in linije[:-1]] + [linije[-1]] if linije else []


def celija_iz_bloka(blok: str) -> dict | None:
    linije = _skrati(blok.split("\n"))
    if not linije:
        return None

    korisne = [l for l in linije if l.strip()]
    je_magic = all(l.startswith(MAGIC) for l in korisne)

    if je_magic:
        telo = _skrati(_skini_magic(linije))
        if telo and telo[0].strip().startswith("%md"):
            telo[0] = telo[0].strip()[len("%md"):].lstrip()
            telo = _skrati(telo)
            return {"cell_type": "markdown", "metadata": {}, "source": _izvor(telo)}
        # %pip / %sh / %sql / ostalo -> code celija sa zadrzanim magicom
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": _izvor(telo)}

    # obicna code celija; `# DBTITLE 1,Naslov` -> obican komentar
    telo = []
    for l in linije:
        if l.startswith("# DBTITLE"):
            naslov = l.split(",", 1)[1].strip() if "," in l else ""
            if naslov:
                telo.append("# " + naslov)
            continue
        telo.append(l)
    telo = _skrati(telo)
    if not telo:
        return None
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _izvor(telo)}


def konvertuj(put_py: Path) -> Path:
    tekst = put_py.read_text(encoding="utf-8").replace("\r\n", "\n")
    if tekst.startswith(ZAGLAVLJE):
        tekst = tekst[len(ZAGLAVLJE):]

    celije = [c for c in (celija_iz_bloka(b) for b in tekst.split(RAZDELNIK)) if c]
    for i, c in enumerate(celije):
        c["id"] = f"c{i:03d}"          # nbformat >= 4.5 trazi id po celiji
    nb = {
        "cells": celije,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    put_ipynb = put_py.with_suffix(".ipynb")
    put_ipynb.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    return put_ipynb


if __name__ == "__main__":
    koren = Path(__file__).resolve().parent.parent
    mete = ([Path(a) for a in sys.argv[1:]] or
            sorted((koren / "notebooks").glob("*.py")))
    for m in mete:
        izlaz = konvertuj(m)
        broj = len(json.loads(izlaz.read_text(encoding="utf-8"))["cells"])
        print(f"{m.name:24s} -> {izlaz.name:26s} ({broj} celija)")
