# Procena broja stanovnika iz satelitskih snimaka

Projekat iz Dubokog učenja (DMI, UNSPMF). Duboki modeli procenjuju broj stanovnika
naselja u Srbiji (bez Kosova i Metohije) iz javno dostupnih podataka daljinskog
osmatranja; procene se porede sa zvaničnim popisom 2022. Tim: Nikola Jarić i
Aleksandar Zdravković.

## Pristupi

Dva komplementarna osnovna pristupa i dva načina fuzije:

| # | Notebook | Ulaz | Ideja |
|---|---|---|---|
| 1 | `sentinel_train_v1.ipynb` | Sentinel-2, 6 opsega, 1 isečak po naselju | ResNet-18 regresija na `log1p(pop)` ili `log1p(gustina)` |
| 1b | `tiles_train.ipynb` | Sentinel-2 pločice 2.24 km | broj stanovnika po pločici (softplus), suma pločica = naselje, loss na sumi — rešava problem velikih naselja (MAUP) |
| 2 | `footprint_train.ipynb` | rasterizovani otisci zgrada (2 kanala: pokrivenost, zapreminska gustina) | ResNet-18 regresija na `log1p(pop)` |
| F1 | `fusion_train.ipynb` | OOF predikcije pristupa 1/1b/2 i F2 | stacking (Ridge u log prostoru) + post-hoc kalibracija po pristupu; bez GPU-a |
| F2 | `multimodal_train.ipynb` | Sentinel isečak + footprint raster istog naselja | zajednički model: dva ResNet-18 trupa, konkatenacija embeddinga, jedna regresiona glava, end-to-end |

Svi pristupi dele isti evaluacioni protokol (`src/procena`): 5-struka GroupKFold
podela po opštinama (bez prostornog curenja između trening i validacionog skupa),
out-of-fold (OOF) predikcija za svako naselje tačno jednom, isti skup metrika i
rezime grafik — rezultati su direktno uporedivi između pristupa.

## Metrike

Raspodela populacije je ekstremno iskošena (medijana naselja ~265, Beograd ~1.4M),
pa prosečne apsolutne greške i linearni R² na sumama dominiraju najveći gradovi.
Glavne metrike su zato: `oof_r2_log` (R² u log1p prostoru), `oof_medape`
(medijalna procentualna greška), `oof_wmape` (relativna greška ponderisana
populacijom), `oof_bias` (Σpred/Σstvarno) i `oof_kalib_nagib` (nagib log-log
regresije; > 1 = kompresija predikcija ka sredini). Dodatno: greške po
veličinskim stratumima naselja i opštinska agregacija sa i bez dve najveće
opštine. Detalji u docstringu `procena.cv.oof_metrics`.

## Struktura

```
src/procena/           zajednicki modul (uvoze ga svi notebooki)
  data.py              normalizacija po opsegu, Dataset, DataLoader-i
  train.py             seeding, trening/eval prolaz, dvofazni trening (glava -> fine-tuning)
  cv.py                GroupKFold foldovi, OOF metrike, CV rezime grafik
  okruzenje.py         detekcija Databricks/Colab, MLflow tracking, izlazni dir, cuvanje OOF-a
notebooks/             Databricks source format (.py) + generisani .ipynb
  EDA.ipynb            analiza podataka: jedinice, populacija, footprinti, snimci
  sentinel_train_v1    pristup 1 (Sentinel CNN, pop i density cilj)
  tiles_train          pristup 1b (plocice + agregaciona loss, log vs linear)
  footprint_train      pristup 2 (CNN nad otiscima zgrada)
  fusion_train         fuzija F1 (stacking + kalibracija nad OOF predikcijama)
  multimodal_train     fuzija F2 (zajednicki dvogranski model)
scripts/               priprema podataka + dijagnostika (lokalno — videti dole)
results/               mali artefakti: EDA grafici, sazeci, tabela labela
data/                  nije u repozitorijumu (preveliko; deli se kao zip preko Drive-a)
```

## Izvori podataka

| Uloga | Izvor |
|---|---|
| Snimci (ulaz) | Sentinel-2 L2A, Copernicus Data Space / openEO (https://dataspace.copernicus.eu) |
| Geometrija naselja | Registar prostornih jedinica, GeoSrbija (https://download.geosrbija.rs) |
| Broj stanovnika (labela) | Popis 2022, RZS (https://popis2022.stat.gov.rs/sr-latn/popisni-podaci-eksel-tabele/) |
| Otisci zgrada | Overture Maps (https://docs.overturemaps.org/guides/buildings/) |

Polazna referenca: Yeh et al., *Using publicly available satellite imagery and deep
learning to understand economic well-being in Africa*, Nature Communications (2020).

## Priprema podataka (lokalno, redom)

```
python scripts/build_labels.py            # RZS popis + geometrija naselja
python scripts/make_dataset_table.py      # master tabela naselja
python scripts/footprints_per_naselje.py  # otisci zgrada po naselju (Overture)
python scripts/cutout_sentinel_batch.py subset   # Sentinel iseci preko openEO
python scripts/footprint_rasters.py       # rasterizacija otisaka (pristup 2)
python scripts/tile_cutouts.py            # plocice (pristup 1b)
python scripts/package_for_colab.py       # data_upload.zip + footprint_upload.zip
python scripts/package_tiles.py           # tiles_upload.zip
```

Van ovog redosleda, kao dijagnostika:

```
python scripts/rural_footprints.py        # pokrivenost Overture otiscima u selima
```

Uzorkuje 6 sela (pop 50–800) i 2 depopulaciona naselja (pop 1–20) i izveštava broj
zgrada, krovnu površinu i izvore. Provera rizika za pristup 2: ako su ML-generisani
otisci retki u selima, izgrađenost je nepouzdan signal baš tamo gde je populacija
najmanja. Rezultat ide u `data/eda/`.

## Notebooci u .ipynb formatu

Izvor je Databricks `.py` format (`notebooks/*.py`) — to je ono što se menja.
`.ipynb` verzije se iz njega generišu (markdown i `%pip` ćelije se vraćaju u
prave notebook ćelije), pa ih regenerisati posle svake izmene `.py` fajla:

```
python scripts/db_py_to_ipynb.py            # svi notebooks/*.py
python scripts/db_py_to_ipynb.py notebooks/fusion_train.py
```

## Trening

**Databricks** (primarno): repo je povezan kao Databricks Repo; podaci su
raspakovani na UC Volume (`.../raw_data/data`). Otvoriti notebook i `Run all` —
putanje, MLflow tracking i izlazi se podese sami. Redosled za fuziju: prvo
trenirački notebooki (1/1b/2 i F2 `multimodal_train`, koji trenira iz sirovih
ulaza — svaki snimi `oof_<pristup>.parquet`), pa `fusion_train`. `fusion_train`
uzima svaki OOF parquet koji zatekne, pa radi i sa podskupom pristupa.

**Colab** (rezerva): kloniraj repo u `/content` (zbog `src/procena`), uploaduj
odgovarajući zip u `/content`, pa pokreni ćelije redom. MLflow: ako su postavljeni
`DATABRICKS_HOST` i `DATABRICKS_TOKEN` (env varijable / Colab Secrets), metrike
idu u zajednički Databricks eksperiment; bez njih u lokalni `mlruns/`. Težine
modela i OOF predikcije idu u `/content/out`.

Svaki run se prati kroz MLflow: hiperparametri, metrike po epohi, OOF metrike,
rezime grafik, težine po foldu i OOF parquet kao artefakti.
