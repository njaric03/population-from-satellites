"""scripts - priprema podataka za projekat procene broja stanovnika.

Podpaketi prate grane pipeline-a:

* ``preprocessing`` - labele i master tabela naselja; zajednicko svim pristupima.
* ``sentinel``      - satelitski ulazi: kompoziti po okrugu, isecak po naselju
                      (za multimodalnu fuziju) i plocice 2.24 km (pristup 1).
* ``footprint``     - otisci zgrada iz Overture, rasterizacija (pristup 2) i
                      provera pokrivenosti (``coverage``, van pipeline-a).

Izlaz pipeline-a je ``data/`` na disku; na Databricks se prenosi na UC Volume.

Sve putanje su u :mod:`scripts.config`. Skripte se pokrecu iz korena repoa
kao moduli, npr.::

    python -m scripts.preprocessing.build_labels
"""
