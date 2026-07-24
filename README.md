# Procena broja stanovnika iz satelitskih snimaka

Projekat iz Dubokog učenja (DMI, UNSPMF). Duboki modeli procenjuju broj stanovnika
naselja u Srbiji (bez Kosova i Metohije) iz javno dostupnih podataka daljinskog
osmatranja; procene se porede sa zvaničnim popisom 2022. Tim: Nikola Jarić i
Aleksandar Zdravković.

## Pristupi

Dva komplementarna osnovna pristupa i dva načina fuzije:

| # | Notebook | Ulaz | Ideja |
|---|---|---|---|
| 1 | `02_tiles_train.ipynb` | Sentinel-2 pločice 2.24 km | broj stanovnika po pločici (softplus), suma pločica = naselje, loss na sumi; naselja variraju od sela do grada, pa fiksan isečak po naselju ne radi (MAUP) |
| 2 | `03_footprint_train.ipynb` | 14 strukturiranih atributa otisaka zgrada po naselju, svi iz geometrije (količina, oblik, raspored) | MLP na `log1p(pop)`; gradient boosting kao referentna linija |
| 2b | `03_footprint_train.ipynb` | isti otisci, rasterizovani u 2 kanala (pokrivenost, gustina zgrada) | ResNet-18 regresija na `log1p(pop)`: da li prostorni raspored nosi nešto preko brojača? |
| F1 | `05_fusion_train.ipynb` | OOF predikcije pristupa 1/2/2b i F2 | stacking (Ridge u log prostoru) + post-hoc kalibracija po pristupu; bez GPU-a |
| F2 | `04_multimodal_train.ipynb` | Sentinel isečak + footprint raster + 14 strukturiranih atributa istog naselja | jedan model koji istovremeno dobija strukturirane podatke i sliku: dva ResNet-18 trupa (512-dim svaki) + MLP grana (32-dim), konkatenacija u zajedničku glavu, end-to-end |

Svi pristupi dele isti evaluacioni protokol (`core`): 5-struka GroupKFold
podela po opštinama (bez prostornog curenja između trening i validacionog skupa),
out-of-fold (OOF) predikcija za svako naselje tačno jednom, isti skup metrika i
rezime grafik; rezultati su direktno uporedivi između pristupa.

## Metrike

Raspodela populacije je ekstremno iskošena (medijana naselja ~265, Beograd ~1.4M),
pa prosečne apsolutne greške i linearni R² na sumama dominiraju najveći gradovi.
Glavne metrike su zato: `oof_r2_log` (R² u log1p prostoru), `oof_medape`
(medijalna procentualna greška), `oof_wmape` (relativna greška ponderisana
populacijom) i `oof_bias` (Σpred/Σstvarno). Dijagnostika: `oof_kalib_nagib`
(nagib log-log regresije; > 1 = kompresija predikcija ka sredini), medAPE po
veličinskim stratumima naselja i opštinska agregacija sa i bez dve najveće
opštine.

Skup je namerno sveden na 13 ključeva. Izostavljeni su `oof_rmse_log`
(determinisana funkcija `oof_r2_log`, varijansa je ista za sve pristupe),
`oof_mae_log` (log jedinice se ne tumače), `oof_rmse_stanovnici` (kvadrat
greške u stanovnicima ga svodi na Beograd), linearni `oof_opstina_r2` sa i bez
top-2 (na rasponu opština 5k-1.6M ga nose ili ruše dva grada), i `oof_n_<strat>`
sa `oof_mae_<strat>` (prvo opisuje skup a ne run, drugo samo ponavlja da velika
naselja imaju velike apsolutne greške).

Opštinska agregacija se računa sa i bez dve najveće opštine jer to razdvaja
"model ne radi" od "megagradovi se ne ekstrapoliraju u GroupKFold-u", a medAPE
po stratumima pokazuje gde model radi a gde ne. Računa ih `core.cv.oof_metrics`.

Kompresiju predikcija ka sredini ispravlja post-hoc kalibracija u
`05_fusion_train`: za **svaki** pristup, pod istim foldovima, uči se monotona
funkcija predviđeno → ispravljeno (Ridge nagib+presek u log-log, i isotonic).
Monotona je pa ne menja redosled naselja, popravlja skalu, ne rangiranje.

## Struktura

```
core/                  zajednicki modul (uvoze ga svi treniracki notebooci)
  data.py              normalizacija po opsegu, Dataset, DataLoader-i
  train.py             seeding, trening/eval prolaz, dvofazni trening (glava -> fine-tuning)
  cv.py                GroupKFold foldovi, OOF metrike, CV rezime grafik
  environment.py       MLflow tracking, izlazni dir, cuvanje OOF-a (koren okruzenja iz scripts/config.py)
notebooks/             Jupyter notebooci; prefiks = redosled pokretanja
  01_eda.ipynb              analiza podataka: jedinice, populacija, footprinti, snimci
  02_tiles_train.ipynb      pristup 1 (plocice + agregacioni loss, log vs linear)
  03_footprint_train.ipynb  pristup 2 (tabelarni MLP + GBM referenca) i 2b (CNN nad rasterom)
  04_multimodal_train.ipynb fuzija F2 (zajednicki trogranski model: 2 slike + atributi)
  05_fusion_train.ipynb     fuzija F1 (stacking nad OOF predikcijama 02-04)
scripts/               priprema podataka (lokalno; paket, pokrece se sa -m)
  config.py            sve putanje i detekcija okruzenja; uvoze ga skripte, notebooci i core
  preprocessing/       labele i master tabela naselja, zajednicko svim pristupima
  sentinel/            satelitski ulazi: kompoziti, isecci (za F2), plocice (pristup 1)
  footprint/           otisci zgrada: strukturirani atributi (pristup 2),
                       rasterizacija (2b), coverage provera
results/               terminalne tabele i sazeci (.csv, .json)
figures/               terminalne slike (.png); popunjava se pokretanjem 01_eda
data/                  nije u repozitorijumu (preveliko; prenosi se na Databricks UC Volume)
requirements.txt       zavisnosti za lokalni rad
```

`05_fusion_train` je numerisan poslednji jer jedini cita tudje izlaze
(`oof_<pristup>.parquet`); 02-04 su medjusobno nezavisni.

Izlazi runa (`out/`, `mlruns/`, `.pt` tezine, `oof_*.parquet`) se ne cuvaju u
git-u, regenerisu se pokretanjem i ostaju uz MLflow run kao artefakti.

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
pip install -r requirements.txt
```

Skripte su paket, pa se pokreću iz korena repoa sa `-m` (tako `from scripts
import config` radi bez petljanja po `sys.path`). Redosled nije proizvoljan:
svaki korak čita izlaz nekog ranijeg:

```
python -m scripts.preprocessing.build_labels        # RZS popis .xlsx + geometrija -> naselje_pop_final.csv
python -m scripts.preprocessing.make_dataset_table  # master tabela (centroid, okrug, area) -> naselje_table.parquet
python -m scripts.footprint.per_naselje             # Overture otisci -> 14 atributa po naselju iz geometrije (pristup 2)
python -m scripts.sentinel.cutouts subset           # openEO: kompozit po okrugu -> 1 isecak po naselju
python -m scripts.footprint.rasters                 # otisci -> 2-kanalni rasteri: pokrivenost + gustina (pristup 2b)
python -m scripts.sentinel.tiles                    # kompoziti -> plocice 2.24 km (pristup 1)
```

Gotov `data/` se onda prenosi na Databricks UC Volume, odakle ga notebooci čitaju.

Zavisnosti (zašto baš taj redosled):

```
preprocessing.build_labels
  └─> preprocessing.make_dataset_table
        ├─> footprint.per_naselje ─┐
        │                          ├─> footprint.rasters
        └─> sentinel.cutouts ──────┤
                                   └─> sentinel.tiles
```

`footprint.per_naselje` i `sentinel.cutouts` mogu paralelno, oba traže samo
`naselje_table.parquet`. `footprint.rasters` i `sentinel.tiles` oba traže i
otiske iz Overture i Sentinel izlaz: rasterizacija zato što crta zgrade preko
isečka, a pločice zato što centroidima zgrada odbacuju prazne.

Van pipeline-a, dijagnostika (traži samo `build_labels`, ne ulazi ni u šta):

```
python -m scripts.footprint.coverage                # pokrivenost Overture otiscima u selima
```

Uzorkuje 6 sela (pop 50-800) i 2 depopulaciona naselja (pop 1-20) i izveštava broj
zgrada, krovnu površinu i izvore. Provera rizika za pristup 2: ako su ML-generisani
otisci retki u selima, izgrađenost je nepouzdan signal baš tamo gde je populacija
najmanja. Rezultat ide u `results/rural_footprints.csv`, odakle ga `01_eda` čita.

**Gde šta ide** (putanje su u [`scripts/config.py`](scripts/config.py), skripte
ih uvoze odatle umesto da svaka sklapa svoje):

| | sadržaj | u gitu |
|---|---|---|
| `data/` | ulazi i međukoraci: sve što neki sledeći korak čita kao ulaz: geometrije, isečci, pločice, `naselje_table.parquet`, `naselje_pop_final.csv` | ne (preveliko) |
| `results/` | terminalne tabele i sažeci (`.csv`, `.json`) | da |
| `figures/` | terminalne slike (`.png`) | da |
| `out/`, MLflow | težine po foldu, OOF predikcije, rezime grafici runa | ne (regeneriše se) |

Terminalno = niko to dalje ne konzumira kao ulaz, nego se gleda ili predaje.

## Trening

**Databricks** (primarno): repo je povezan kao Databricks Repo; podaci su
preneti na UC Volume (`.../raw_data/data`). Otvoriti notebook i `Run all`:
putanje, MLflow tracking i izlazi se podese sami. Redosled za fuziju: prvo
trenirački notebooki (1/2 i F2 `04_multimodal_train`, koji trenira iz sirovih
ulaza, svaki snimi `oof_<pristup>.parquet`), pa `05_fusion_train`. `05_fusion_train`
uzima svaki OOF parquet koji zatekne, pa radi i sa podskupom pristupa.

**Lokalno** (za razvoj i EDA): `pip install -r requirements.txt`, pa
`jupyter lab`. Bez `/databricks` na disku `core.environment` sam prelazi na
`out/` za težine i OOF, i na lokalni `mlruns/` za metrike. Ako su postavljeni
`DATABRICKS_HOST` i `DATABRICKS_TOKEN`, metrike umesto toga idu u zajednički
Databricks eksperiment. Trening celog skupa lokalno nema smisla bez GPU-a:
lokalno se pokreće `01_eda` i `05_fusion_train` (fuzija ne traži GPU).

Svaki run se prati kroz MLflow: hiperparametri, metrike po epohi, OOF metrike,
rezime grafik, težine po foldu i OOF parquet kao artefakti.
