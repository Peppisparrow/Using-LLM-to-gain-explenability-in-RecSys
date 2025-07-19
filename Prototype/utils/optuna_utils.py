
import os
from pathlib import Path
import pandas as pd
import csv

class SaveResults(object):
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.results_df = pd.DataFrame(columns = ["result"])
    
    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]
           
        self.results_df = pd.concat([self.results_df, pd.DataFrame([hyperparam_dict])], ignore_index=True)
        self.results_df.to_csv(self.csv_path, index = False)

class SaveResultsWithUserAttrs(object):
    """
    Callback di Optuna per salvare i risultati di ogni trial su un file CSV.
    Salva il numero del trial, il valore della metrica, i parametri,
    e tutti gli attributi utente (user_attrs) definiti nel trial.
    """
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        # Assicurati che la directory esista
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._is_header_written = False

    def __call__(self, optuna_study, optuna_trial):
        # Definisci i dati di base da salvare
        base_data = {
            'number': optuna_trial.number,
            'value': optuna_trial.value, # Il valore ritornato dalla objective function
        }
        
        # Unisci i dati di base, i parametri e gli user_attrs in un unico dizionario
        # trial.params contiene i parametri (es. 'iterations')
        # trial.user_attrs contiene le metriche extra (es. 'PRECISION', 'RECALL')
        all_data = {**base_data, **optuna_trial.params, **optuna_trial.user_attrs}

        # Scrivi l'header solo la prima volta
        if not self._is_header_written:
            with self.csv_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_data.keys())
                writer.writeheader()
                writer.writerow(all_data)
            self._is_header_written = True
        else:
            # Aggiungi i risultati in append
            with self.csv_path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_data.keys())
                writer.writerow(all_data)