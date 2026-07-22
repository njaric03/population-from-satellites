"""scripts – priprema podataka za projekat procene broja stanovnika.

Podpaketi prate grane pipeline-a:

* ``preprocessing`` – labele i master tabela naselja; zajednicko svim pristupima.
* ``sentinel``      – satelitski ulazi: isecak po naselju (pristup 1) i
                      plocice 2.24 km (pristup 1b).
* ``footprint``     – otisci zgrada iz Overture i njihova rasterizacija (pristup 2).
* ``diagnostics``   – provere kvaliteta podataka; ne ulaze u pipeline.

Izlaz pipeline-a je ``data/`` na disku; na Databricks se prenosi na UC Volume.

Sve putanje su u :mod:`scripts.config`. Skripte se pokrecu iz korena repoa
kao moduli, npr.::

    python -m scripts.preprocessing.build_labels
"""
