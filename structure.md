# Intro
## Mission overview
Parlare della missione dei cansat, la quale si trova ben spiegata nella cartella /contesto.
Dire che le misurazioni dei sensori sono molto sensibili a disturbi esterni, visto che sono montate su cansat che scendono dall'alto su dei paracaduti e possono oscillare, ruotare su se stessi
## Research question


## Methods overview
### Spectral: come e perchè
Vogliamo usare metodi spettrali per identificare possibili disturbi esterni alle misurazioni.
Per esempio l'oscillazione del paracadute e la rotazione di questo attorno al proprio asse, che alterano la misura dei dati del vento o altri effetti.
### Data analysis: come e perchè 
Analisi dei dati atmosferici post lavorazione con spectral methods.

## Expected results
### A livello di time frequency
### A livello atmosferico

# Spectral analysis
## Spiegazione generale metodi 
### Metodi di identificazioni e criteri di esclusioni


## Applicazioni dei metodi ai vari dati 
Facciamo la seguente analisi per tutti i dati
### Passa basso
Applichiamo un filtro passa-basso per eliminare il rumore dei sensori
### Eliminazione trend
In modo da farlo diventare una time series, togliamo il trend principale, da lì lavoriamo
### Spectral
Applichiamo FFT e welch in modo da vedere l'energie delle varie frequenze
### Analisi e rimozioni roba esterna
Capiamo se certe frequenze sono riconducibili a fenomeni esterni e rimuoviamo
## Plot finali dei dati
Facciamo un plot dei nuovi dati
# Analisi atmosferica
 
## Vertical profile analysis
### Commento dei risultati a livello atmosferico del capitolo precedente

## External dataset
### 

# Conclusioni