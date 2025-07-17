import pandas as pd
import json
from tqdm import tqdm
import os

# --- FILE PATHS ---
# Assicurati che questi percorsi siano corretti per la tua struttura di file
data_dir = 'Dataset/ml/ml-latest-small'
INPUT_CSV_PATH = os.path.join(data_dir, 'app_plots.csv')
OUTPUT_JSON_PATH = os.path.join(data_dir, 'item_prompts.json')

# --- Helper Function ---

def _create_item_prompt(title: str, description: str, genres: str) -> str:
    """
    Crea un prompt formattato per un singolo item (film/gioco).

    Args:
        title (str): Il titolo dell'item (dalla colonna 'new_title').
        description (str): La descrizione o trama dell'item (dalla colonna 'plot').
        genres (str): I generi associati all'item.

    Returns:
        str: Una stringa di prompt formattata.
    """
    # Pulisce gli input rimuovendo spazi extra e ritorni a capo
    clean_title = " ".join(str(title).split())
    clean_description = " ".join(str(description).split())
    clean_genres = " ".join(str(genres).split())
    
    # Assembla il prompt finale nel formato richiesto
    prompt = (
        f"Title: {clean_title}\n"
        f"Description: {clean_description}\n"
        f"Genres: {clean_genres}"
    )
    return prompt

# --- Main Function ---

def generate_prompts_from_csv(csv_path: str):
    """
    Legge il file app_plots.csv e genera un prompt per ogni riga.
    
    Tutte le informazioni necessarie (titolo, trama, generi) sono contenute
    in questo unico file, semplificando il processo di caricamento.

    Args:
        csv_path (str): Il percorso del file app_plots.csv.

    Returns:
        dict: Un dizionario che mappa app_id al suo prompt generato, o None se si verifica un errore.
    """
    # 1. Carica il file CSV principale
    try:
        print(f"Caricamento dati da '{csv_path}'...")
        # Specifichiamo le colonne necessarie per ottimizzare l'uso della memoria
        required_cols = ['app_id', 'new_title', 'plot', 'genres']
        df = pd.read_csv(csv_path, usecols=required_cols)
        
        # Gestione dei valori mancanti (NaN) per evitare errori
        # Sostituiamo i valori nulli con una stringa di default
        df['new_title'].fillna('No Title Available', inplace=True)
        df['plot'].fillna('No Plot Available', inplace=True)
        df['genres'].fillna('No Genres Available', inplace=True)

    except FileNotFoundError:
        print(f"ERRORE CRITICO: File non trovato al percorso '{csv_path}'")
        return None
    except ValueError as e:
        print(f"ERRORE CRITICO: Il file CSV non contiene le colonne necessarie. Dettagli: {e}")
        return None

    # 2. Itera su ogni riga del DataFrame per generare i prompt
    all_prompts = {}
    print("\nGenerazione dei prompt per ogni item...")
    
    # tqdm fornisce una barra di avanzamento per monitorare il processo
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creazione Prompts"):
        app_id = row['app_id']
        title = row['new_title']
        plot = row['plot']
        genres = row['genres']
        
        # Crea il prompt usando la funzione helper
        prompt = _create_item_prompt(title, plot, genres)
        
        # Salva il prompt nel dizionario. La chiave app_id è convertita in stringa
        # per garantire la compatibilità con il formato JSON.
        all_prompts[str(app_id)] = prompt

    return all_prompts

# --- Blocco di Esecuzione Principale ---
if __name__ == "__main__":
    print("--- Avvio del processo: Generazione Prompt da CSV ---")
    
    # Assicurati che la directory di output esista
    os.makedirs(data_dir, exist_ok=True)
    
    # Esegui la funzione principale per generare i prompt
    generated_prompts = generate_prompts_from_csv(csv_path=INPUT_CSV_PATH)
    
    # 3. Se i prompt sono stati generati, salvali su file
    if generated_prompts:
        try:
            count = len(generated_prompts)
            print(f"\nSalvataggio di {count} prompt nel file JSON...")
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                # 'indent=4' formatta il JSON in modo che sia leggibile
                json.dump(generated_prompts, f, indent=4, ensure_ascii=False)
            print(f"--- PROCESSO COMPLETATO ---")
            print(f"I prompt sono stati salvati con successo in: '{OUTPUT_JSON_PATH}'")
        except IOError as e:
            print(f"ERRORE: Impossibile scrivere sul file di output '{OUTPUT_JSON_PATH}': {e}")
    else:
        print("\nNessun prompt generato. Controlla i log per eventuali errori.")