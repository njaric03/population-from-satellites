# Procena broja stanovnika iz satelitskih snimaka

Projekat iz Dubokog učenja. Konvoluciona neuronska mreža (ResNet-18, transfer learning)
procenjuje broj stanovnika naselja na osnovu Sentinel-2 snimaka. Obuhvat je Srbija bez
Kosova i Metohije.

Tim: Nikola Jarić i Aleksandar Zdravković. Dva komplementarna pristupa:
1. CNN nad satelitskim snimkom (Sentinel-2, 6 opsega),
2. CNN nad rasterizovanim otiscima zgrada (building footprints),

uz kasnije spajanje oba u zajednički model.

## Struktura

```
notebooks/
  EDA.ipynb            analiza podataka: jedinice, populacija, footprinti, snimci
  colab_train.ipynb    trening na Google Colab GPU (pristup 1)
scripts/
  build_labels.py            spajanje RZS populacije sa geometrijom naselja
  make_dataset_table.py      master tabela naselja (centroid, labela, grupisanje)
  footprints_per_naselje.py  footprinti po naselju iz Overture baze
  cutout_sentinel_batch.py   Sentinel iseci preko openEO (batch po okrugu)
  rural_footprints.py         provera pokrivenosti footprintima u selima
  build_eda_notebook.py       generise notebooks/EDA.ipynb
  build_colab_notebook.py     generise notebooks/colab_train.ipynb
  package_for_colab.py        pakuje cutoute u data_upload.zip za Colab
results/               mali artefakti: EDA grafici, sazeci, tabela labela
data/                  nije u repozitorijumu (preveliko)
```

## Izvori podataka

| Uloga | Izvor |
|---|---|
| Snimci (ulaz) | Sentinel-2 L2A, Copernicus Data Space / openEO (https://dataspace.copernicus.eu) |
| Geometrija naselja | Registar prostornih jedinica, GeoSrbija (https://download.geosrbija.rs) |
| Broj stanovnika | Popis 2022, RZS (https://popis2022.stat.gov.rs/sr-latn/popisni-podaci-eksel-tabele/) |
| Footprinti | Overture Maps (https://docs.overturemaps.org/guides/buildings/) |

## Pokretanje

Priprema podataka (lokalno, redom):
```
python scripts/build_labels.py
python scripts/make_dataset_table.py
python scripts/footprints_per_naselje.py
python scripts/cutout_sentinel_batch.py subset
python scripts/package_for_colab.py
```

Trening: otvori `notebooks/colab_train.ipynb` na Google Colab, ukljuci GPU,
uploaduj `data_upload.zip` u `/content`, pa Run all.

## Podaci

Folder `data/` (geometrije, iseci, kompoziti, ~GB) se ne cuva u git-u. Izvedeni mali
artefakti su u `results/`. Za rad u paru, `data_upload.zip` se deli preko Google Drive-a.
