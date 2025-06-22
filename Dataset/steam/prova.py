import pandas as pd
import json
import os

# --- CONFIGURAZIONE ---
# Assicurati che questi percorsi puntino alla posizione corretta dei tuoi file.
# Se lo script è nella stessa cartella dei file, non serve cambiarli.
METADATA_PATH = 'games_metadata.json'
RECOMMENDATIONS_PATH = 'recommendations.csv'


def load_game_metadata(path):
    """
    Carica i metadati dei giochi da un file JSON Lines in un dizionario.
    Il dizionario userà app_id come chiave e la descrizione come valore.
    """
    print(f"Caricamento dei metadati da '{path}'...")
    metadata_dict = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Assicuriamoci che l'app_id e la descrizione esistano
                    if 'app_id' in data and 'description' in data:
                        metadata_dict[data['app_id']] = data['description']
                except json.JSONDecodeError:
                    # Ignora le righe malformate nel file JSONL
                    print(f"Attenzione: saltata una riga JSON non valida.")
    except FileNotFoundError:
        print(f"ERRORE: File non trovato: '{path}'")
        return None
    
    print(f"Caricati metadati per {len(metadata_dict)} giochi.")
    return metadata_dict


def get_user_reviewed_game_descriptions(user_id, recommendations_path, metadata_dict):
    """
    Trova tutte le descrizioni dei giochi recensiti da un utente specifico.
    """
    if metadata_dict is None:
        return []

    print(f"\nRicerca delle recensioni per l'utente con ID: {user_id}...")
    user_app_ids = set() # Usiamo un set per evitare duplicati se un utente recensisce più volte
    
    try:
        # Leggiamo il file CSV a pezzi (chunks) per non caricare tutto in memoria.
        # Utile se recommendations.csv è molto grande.
        chunk_iter = pd.read_csv(
            recommendations_path,
            chunksize=100000, # Legge 100,000 righe alla volta
            usecols=['user_id', 'app_id'] # Leggiamo solo le colonne che ci servono
        )

        for chunk in chunk_iter:
            # Filtriamo il chunk per trovare le righe dell'utente desiderato
            user_reviews = chunk[chunk['user_id'] == user_id]
            if not user_reviews.empty:
                # Aggiungiamo gli app_id trovati al nostro set
                user_app_ids.update(user_reviews['app_id'].tolist())

    except FileNotFoundError:
        print(f"ERRORE: File non trovato: '{recommendations_path}'")
        return []

    if not user_app_ids:
        print(f"Nessuna recensione trovata per l'utente {user_id}.")
        return []

    print(f"Trovati {len(user_app_ids)} giochi unici recensiti dall'utente {user_id}.")
    
    # Ora recuperiamo le descrizioni per ogni app_id
    print("Recupero delle descrizioni dei giochi...")
    user_descriptions = []
    for app_id in user_app_ids:
        # Usiamo .get() per evitare errori se un app_id non ha metadati
        description = metadata_dict.get(app_id)
        # Aggiungiamo la descrizione solo se esiste e non è vuota
        if description:
            user_descriptions.append(description)
            
    return user_descriptions


# --- ESECUZIONE DELLO SCRIPT ---
if __name__ == "__main__":
    # Sostituisci questo ID con quello dell'utente che vuoi analizzare
    TARGET_USER_ID = 23869
    
    # 1. Carica i metadati una sola volta
    game_descriptions_db = load_game_metadata(METADATA_PATH)
    
    # 2. Ottieni le descrizioni per l'utente target
    # Controlla che il caricamento dei metadati sia andato a buon fine
    if game_descriptions_db:
        descriptions = get_user_reviewed_game_descriptions(
            user_id=TARGET_USER_ID,
            recommendations_path=RECOMMENDATIONS_PATH,
            metadata_dict=game_descriptions_db
        )
        
        # 3. Stampa i risultati
        print("\n--- RISULTATI ---")
        if descriptions:
            print(f"Trovate {len(descriptions)} descrizioni per i giochi recensiti dall'utente {TARGET_USER_ID}:\n")
            for i, desc in enumerate(descriptions, 1):
                print(f"DESCRIZIONE {i}:\n{desc}\n")
                print("-" * 20)
        else:
            print(f"Nessuna descrizione trovata per i giochi recensiti dall'utente {TARGET_USER_ID}.")