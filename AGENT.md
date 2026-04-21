# Se sei un agente AI, leggi attentamente questo file e rispettalo quando lavori a questo progetto.

Il nostro progetto si baserà sull'analisi di dati atmosferici, proveniente da una sonda che ha raccolto dati di altezza, pressione e temperatura dell'aria durante la sua discesa. nella cartella /contesto trovi informazioni generali sulla missione da cui questi dati provengono e altre info generali. Tienine di conto.

Mano a mano che scopri cosa utili, aggiorna questo file, per dare contesto a chat future. Oppure scrivile in /contesto se preferisci.

L'analsi dati sarà basata su tecniche di time/frequency analysis of signal.

Dovremmo scrivere un notebook, main.ipynb contenente tutto il nostro lavoro tecnico, ci appoggeremo a utils.py per costruire helper functions.

Quando ti chiedo di fare dei test o ti faccio domande in generale, se scrivi codici o fai cose, falle nella cartella chiamata tentativi

Il codice main dovrà essere runnabile su Google Colab, dovrà pullare utils.py dalla repo: git@github.com:mattiademartino/Space_Data_2.git

Ogni volta che scrivi o modifiche main.ipynb, assicurati che funzioni in Colab

## Report

Dovremmo fare anche un report in Colab, dove useremo risultati e plot ricavati da main.ipynb, assicurati che ogni plot e immagine del notebook venga salvata in /report/images e che i due contenuti siano coerenti.

Inoltre il report dovrà rispettare le seguenti regole: