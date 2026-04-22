# Bozza Presentazione

## Titolo / apertura

Il mio progetto analizza i dati raccolti durante la missione CanSat GRASP-VAMOS per capire come variano temperatura, pressione, particolato, CO2 e vento con quota e tempo, e per verificare se tecniche di signal processing possono aiutarci a distinguere il comportamento atmosferico reale dagli artefatti strumentali.

## Research question

La domanda centrale è: come cambiano temperatura, pressione, PM2.5, PM10, CO2 e vento durante la discesa, e cosa ci dice l'analisi spettrale sulla struttura della bassa troposfera e sul comportamento dei sensori?

Questa domanda ha senso perché non voglio solo descrivere i dati, ma anche interpretarli fisicamente: stabilita atmosferica, possibile inversione termica, accumulo di aerosol vicino al suolo e presenza o assenza di oscillazioni del payload durante la discesa.

## Dataset che usero

I dataset principali sono:

- `science_GRASP.csv`: temperatura, pressione, quota GPS, PM2.5 e PM10 durante la discesa. E il dataset piu utile per il profilo verticale ad alta frequenza.
- `science_VAMOS.csv`: CO2, temperatura e pressione su una finestra temporale molto piu lunga della missione.
- `wind_VAMOS.csv`: velocita e direzione del vento, componenti e accelerazioni, utile per l'analisi dinamica e spettrale.
- `OBAMA_data_decoded.xlsx`: dataset esterno di un altro CanSat usato come confronto indipendente su temperatura, pressione e umidita.
- `uwyo_06610_2026-02-05_12Z.csv`: sounding dell'University of Wyoming, usato come contesto atmosferico esterno su profilo termo-dinamico e wind shear.
- Atmosfera standard ISA: riferimento teorico per confrontare temperatura e pressione attese con quelle osservate.

## Perche questi dataset

GRASP e il cuore dell'analisi perché contiene la discesa con campionamento alto, circa 38 Hz, quindi permette di studiare bene il profilo verticale e la parte di signal processing.

VAMOS completa GRASP perché aggiunge CO2 e vento, e soprattutto da il contesto temporale dell'intera missione.

OBAMA e UWYO non sono dati "migliori" dei nostri, ma sono ottimi riferimenti esterni: OBAMA come misura indipendente in-situ, UWYO come contesto sinottico regionale, ISA come baseline fisica semplice e interpretabile.

## Metodo di analisi

Il flusso sara questo:

1. Caricamento e pulizia dei dati.
2. Allineamento temporale e conversione delle unita.
3. Ricostruzione della quota barometrica dalla pressione.
4. Analisi esplorativa con serie temporali e profili rispetto alla quota.
5. Confronto con ISA, OBAMA e UWYO.
6. Analisi del segnale su pressione, temperatura e vento con tecniche spettrali.

Le tecniche principali saranno:

- visualizzazione time series e scatter quota-variabile;
- quota barometrica da pressione, preferita alla quota GPS per il profilo verticale;
- resampling uniforme, per rendere i segnali adatti all'analisi spettrale;
- filtro low-pass Butterworth, per separare trend fisico e rumore;
- detrending lineare, per togliere la discesa monotona prima della FFT;
- FFT con finestra di Hanning;
- Welch PSD, perché e piu robusta della FFT singola per stimare contenuto energetico in frequenza;
- STFT/spectrogram, per vedere come cambia il contenuto in frequenza nel tempo;
- segmentazione per fase di volo nel caso VAMOS, cosi confronto separatamente aircraft, parachute e ground, invece di mescolare tutto in uno spettro unico.

## Scelte importanti e dataset esclusi in parte

Qui e importante dire esplicitamente che non usero tutto in modo cieco.

- In GRASP escludo la prima riga perché contiene un picco di pressione fisicamente impossibile: e un artefatto di inizializzazione del sensore.
- In VAMOS escludo i primi valori di CO2 nulli, perché rappresentano il warm-up del sensore e non una misura atmosferica valida.
- Nel dataset del vento VAMOS rimuovo l'eventuale parte prima del reset temporale, perché romperebbe l'analisi nel dominio del tempo.
- Non usero il flag `tumbling` come indicatore principale, perché nel dataset risulta praticamente sempre nullo e quindi non e informativo.
- Non usero la quota GPS come riferimento principale per il profilo verticale: la uso solo come cross-check, ma per l'analisi scelgo la quota barometrica perché e piu stabile e coerente.
- Non usero UWYO come ground truth del sito di lancio: e un riferimento esterno regionale, utile per interpretazione, non per validazione puntuale.
- Non usero OBAMA per un confronto temporale stretto campione per campione, perché la simultaneita non e garantita; lo usero invece come confronto di ordine di grandezza e coerenza fisica.

## Cosa mi aspetto di mostrare

- Un profilo termico piu caldo dell'ISA, quindi una possibile atmosfera stabile o una inversione termica, perché se la temperatura osservata decresce meno rapidamente di quella standard, o addirittura aumenta con la quota nel range misurato, questo suggerisce una struttura atmosferica non convettiva e piu stabile.
- Un aumento di PM vicino al suolo, perché gli aerosol tendono ad accumularsi nel boundary layer, dove sono presenti piu sorgenti antropiche e minore rimescolamento verticale rispetto agli strati superiori.
- Valori di CO2 sopra il background globale, perché vicino al suolo il segnale puo risentire di emissioni locali come traffico, combustione o attivita antropiche, mentre il background atmosferico globale da solo non spiegherebbe picchi cosi elevati.
- Un'analisi spettrale del vento utile per distinguere le fasi di missione, perché separando aircraft, parachute e ground posso capire se il contenuto in frequenza dipende davvero dalla dinamica del volo e non da una media poco interpretabile sull'intera missione.
- Nessuna evidenza forte di una risonanza pendolare del paracadute, perché se nello spettro o nello spectrogram non compare un picco netto e persistente a una frequenza caratteristica, allora non c'e un'oscillazione dominante chiaramente identificabile.

## Chiusura

In sintesi, il mio approccio combina analisi atmosferica e signal processing: non mi limito a plottare i dati, ma costruisco una pipeline che seleziona i dataset affidabili, scarta gli artefatti, confronta i risultati con riferimenti esterni e usa strumenti spettrali per estrarre informazione fisica reale dal rumore strumentale.
