# AGENT.md

Se stai lavorando su questa repo come agente o collaboratore, segui queste regole senza eccezioni.

## Obiettivo Del Progetto

Questa repo raccoglie e confronta dati atmosferici e di missione provenienti da piu' payload CanSat:

- GRASP
- VAMOS
- OBAMA
- UWYO radiosonde
- ERA5 reanalysis
- ScamSat

L'analisi ufficiale vive in `main.ipynb`.
Le utility condivise vivono in `utils.py`.
Gli esperimenti, script provvisori e tentativi veloci vivono in `tentativi/`.

## Struttura Canonica Della Repo

Usa solo questi percorsi come sorgente di verita':

- `data/`
- `data/Probe_3_VAMOS/`
- `data/Probe_4_GRASP/`
- `data/Probe_5_OBAMA/`
- `data/Probe_6_ScamSat/`
- `data/external_dataset/`
- `figures/`
- `main.ipynb`
- `utils.py`

Non usare piu' cartelle legacy come sorgente principale se non per retrocompatibilita' o recupero storico.
Esempi di cartelle legacy da non usare come default:

- `CanSat data utili/`
- `dati/`

## Regole Sui Dati

- I dati sorgente vanno letti esattamente da dove si trovano ora in `data/`.
- Non spostare, rinominare o duplicare i file originali dei dataset senza una richiesta esplicita.
- I dataset esterni devono stare in `data/external_dataset/`.
- Il radiosondaggio UWYO sta in `data/external_dataset/uwyo_06610_2026-02-05_12Z.csv`.
- I file ERA5 scaricati da `cursed.ipynb` devono finire in `data/external_dataset/`, in particolare il NetCDF canonico e' `data/external_dataset/era5_dubendorf_20260206.nc`.
- ScamSat va letto da `data/Probe_6_ScamSat/scamsat_data_upload/002/`.

## Regole Sui Plot

Tutti i plot prodotti dal notebook principale devono essere salvati in `figures/`.
Non salvare nuove figure nel root della repo, in `Report/Img`, o in cartelle temporanee sparse.

Usa sempre `utils.save_figure(...)` per i salvataggi.
Le sottocartelle canoniche sono:

- `figures/plot_raw/`
- `figures/data_quality/`
- `figures/flight_dynamics/`
- `figures/cross_dataset/`
- `figures/signal_processing/`
- `figures/external_dataset/`
- `figures/scamsat/`
- `figures/exports/`

Classificazione minima attesa:

- plot grezzi e panoramiche iniziali -> `plot_raw/`
- controllo artefatti e anomalie -> `data_quality/`
- dinamica di discesa, vento, finestre di volo -> `flight_dynamics/`
- confronti tra payload o sensori -> `cross_dataset/`
- FFT, PSD, filtri, PM, analisi di segnale -> `signal_processing/`
- UWYO, ERA5 e altri riferimenti esterni -> `external_dataset/`
- prodotti dedicati ScamSat -> `scamsat/`
- JSON e metadati esportati -> `exports/`

## Notebook E Colab

`main.ipynb` deve restare eseguibile in locale e su Google Colab.

Quando modifichi `main.ipynb`:

- non usare path assoluti locali
- importa path e helper da `utils.py`
- mantieni il notebook eseguibile top-to-bottom
- se possibile verifica l'esecuzione con `jupyter nbconvert --to notebook --execute --inplace main.ipynb`

Se lavori in Colab e `utils.py` non esiste localmente, e' accettabile scaricarlo dalla repo.
Non introdurre pero' dipendenze che rendano il notebook eseguibile solo in Colab.

## Regole Su `utils.py`

`utils.py` e' il punto centrale per:

- path canonici della repo
- loader dei dataset
- stile matplotlib condiviso
- helper di salvataggio figure
- funzioni di elaborazione e confronto

Se devi introdurre:

- un nuovo dataset
- un nuovo path
- una nuova categoria di figure
- un nuovo loader riusabile

fallo prima in `utils.py` e poi usalo in `main.ipynb`.
Evita di duplicare nel notebook logica di parsing, path hardcoded o stili grafici globali.

## Regole Operative

- Prima di modificare qualcosa, leggi `README.md`, `AGENT.md` e il codice rilevante.
- Se la richiesta e' esplorativa o sperimentale, lavora in `tentativi/`.
- Se la modifica e' ufficiale e fa parte del flusso finale, aggiornala in `main.ipynb` o `utils.py`.
- Non cancellare materiale storico in `figures/previous_stuff/` salvo richiesta esplicita.
- Non rompere file o notebook gia' presenti per "ripulire" la repo.
- Non introdurre nuove convenzioni in un solo file: aggiorna anche la documentazione.

## Stile Del Codice

- Preferisci codice leggibile e lineare.
- Usa nomi coerenti con i dataset reali: `grasp`, `vamos`, `obama`, `uwyo`, `era5`, `scamsat`.
- Nei notebook, ogni cella deve avere uno scopo chiaro.
- I commenti devono spiegare una scelta o un contesto, non l'ovvio.
- Evita path hardcoded ripetuti.
- Evita salvataggi manuali con `plt.savefig(...)` sparsi: usa sempre `save_figure`.

## Checklist Finale Per Ogni Modifica

- I dati vengono ancora letti da `data/`?
- I dataset esterni stanno in `data/external_dataset/`?
- Le nuove figure finiscono in `figures/` e nella sottocartella giusta?
- `main.ipynb` gira ancora dall'inizio alla fine?
- `README.md` e `AGENT.md` sono ancora coerenti con la repo?

Se una di queste risposte e' no, la modifica non e' finita.
