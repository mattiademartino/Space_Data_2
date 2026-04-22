# Space_Data_2

Repo per l'analisi atmosferica multi-payload della missione CanSat.

## Dove Guardare

- `main.ipynb`: analisi ufficiale end-to-end
- `utils.py`: loader, path canonici, stile plot e helper condivisi
- `data/`: tutti i dati sorgente
- `data/external_dataset/`: dataset esterni come UWYO ed ERA5
- `figures/`: tutti i plot prodotti dal notebook principale
- `tentativi/`: prove, script temporanei ed esperimenti
- `AGENT.md`: regole operative per agenti e collaboratori

## Dataset Usati

- `data/Probe_3_VAMOS/`
- `data/Probe_4_GRASP/`
- `data/Probe_5_OBAMA/`
- `data/Probe_6_ScamSat/`
- `data/external_dataset/uwyo_06610_2026-02-05_12Z.csv`
- `data/external_dataset/era5_dubendorf_20260206.nc` quando scaricato

## Organizzazione Delle Figure

Tutte le figure ufficiali devono andare in `figures/` e sono classificate cosi':

- `figures/plot_raw/`
- `figures/data_quality/`
- `figures/flight_dynamics/`
- `figures/cross_dataset/`
- `figures/signal_processing/`
- `figures/external_dataset/`
- `figures/scamsat/`
- `figures/exports/`

Il file `figures/atmospheric_summary.json` riassume i dataset caricati e i percorsi canonici della repo.

## Esecuzione

Per rieseguire il notebook in locale:

```bash
jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Il notebook e' stato organizzato per restare eseguibile anche in Colab, usando `utils.py` come punto centrale per path e helper.

## ERA5

Il download di ERA5 passa da `cursed.ipynb`.
Quel notebook e' stato configurato per salvare:

- il NetCDF in `data/external_dataset/`
- le figure ERA5 in `figures/external_dataset/`

Finche' il file ERA5 non e' presente, `main.ipynb` continua a funzionare e segnala chiaramente che il dataset manca.
