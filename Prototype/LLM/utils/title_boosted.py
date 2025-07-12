import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm # Per le barre di progresso

# --- PERCORSI DEI FILE ---
# Mantieni i percorsi dei tuoi dataset qui
data_dir = 'Dataset/steam/filtering_no_desc_giappo_corean_k10/small'  # Cartella base per i dataset
METADATA_PATH = 'Dataset/steam/games_metadata.json'
RECOMMENDATIONS_PATH = f'{data_dir}/train_recommendations.csv'
GAMES_PATH = f'{data_dir}/games.csv'
USERS_PATH = f'{data_dir}/users.csv' # File con la lista di user_id da processare
OUTPUT_PATH = f'{data_dir}/user_prompts.json' # File dove salveremo i risultati

# --- Funzioni Helper (alcune modificate, altre nuove) ---

def _load_game_titles(path):
    """Carica i titoli dei giochi da games.csv in un dizionario."""
    try:
        print("Caricamento titoli dei giochi...")
        df = pd.read_csv(path, usecols=['app_id', 'title'])
        return df.set_index('app_id')['title'].to_dict()
    except (FileNotFoundError, KeyError) as e:
        print(f"ERRORE durante il caricamento dei titoli: {e}")
        return None

def _load_game_metadata(path):
    """Carica le descrizioni dei giochi da games_metadata.json in un dizionario."""
    metadata_dict = {}
    print("Caricamento metadata dei giochi...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Lettura metadata"):
                try:
                    data = json.loads(line)
                    if 'app_id' in data and 'description' in data:
                        metadata_dict[data['app_id']] = data['description']
                except json.JSONDecodeError:
                    continue # Salta le linee malformate
        return metadata_dict
    except FileNotFoundError:
        print(f"ERRORE: File non trovato: '{path}'")
        return None

def _get_all_user_reviews_in_batch(path, target_user_ids):
    """
    Legge recommendations.csv in un unico blocco e raggruppa gli app_id per utente.
    ATTENZIONE: Richiede molta RAM.
    """
    print(f"Processando le recensioni per {len(target_user_ids)} utenti...")
    
    try:
        print("Caricamento dell'intero file delle recensioni in memoria (può richiedere molto tempo e RAM)...")
        # 1. Carica l'intero file CSV in un unico DataFrame.
        df_reviews = pd.read_csv(path, usecols=['user_id', 'app_id'])
        
        print("Filtro delle recensioni per gli utenti di interesse...")
        # 2. Filtra il DataFrame per tenere solo gli utenti che ci servono.
        relevant_reviews = df_reviews[df_reviews['user_id'].isin(target_user_ids)]
        
        print("Raggruppamento delle recensioni per utente (metodo ottimizzato)...")
        # 3. Usa groupby, che è estremamente efficiente per raggruppare i dati.
        #    Converte direttamente i gruppi di app_id in liste.
        user_reviews_map = relevant_reviews.groupby('user_id')['app_id'].apply(list).to_dict()
        
        # Converte il dizionario risultante in un defaultdict per coerenza con il resto del codice,
        # anche se in questo caso non sarebbe strettamente necessario.
        return defaultdict(list, user_reviews_map)

    except FileNotFoundError:
        print(f"ERRORE: File non trovato: '{path}'")
        return None
    except MemoryError:
        print("\n--- ERRORE DI MEMORIA ---")
        print("Il sistema non ha abbastanza RAM per caricare l'intero file delle recensioni.")
        print("Esecuzione interrotta. Si prega di utilizzare la versione precedente del codice che elabora il file in 'chunk'.")
        return None
        
def _create_llm_prompt(games_data):
    """Crea il prompt finale per l'LLM (invariato)."""
    instruction = (
        "You are an expert video game analyst. Your task is to create a concise "
        "summary of a user's gaming preferences based on a list of games they have played. "
        "Focus on identifying preferred genres, gameplay mechanics, themes, and settings."
    )
    formatted_games_list = []
    for i, game in enumerate(games_data, 1):
        clean_description = " ".join(game['description'].split())
        game_entry = (
            f"Game {i}:\n"
            f"Title: {game['title']}\n"
            f"Description: {clean_description}"
        )
        formatted_games_list.append(game_entry)
    
    data_section = "\n\n---\n\n".join(formatted_games_list)
    final_prompt = (
        f"{instruction}\n\n"
        f"Please analyze the following games:\n\n"
        f"{data_section}\n\n"
    )
    return final_prompt

# --- FUNZIONE PRINCIPALE OTTIMIZZATA ---

def generate_all_prompts_in_batch(users_path, games_path, metadata_path, recommendations_path):
    """
    Orchestra l'intero processo di generazione dei prompt per TUTTI gli utenti
    specificati nel file users_path.
    """
    # 1. Carica la lista di utenti target dal file users.csv
    try:
        users_df = pd.read_csv(users_path)
        # Usare un set per ricerche O(1), molto più veloce di una lista
        target_user_ids = set(users_df['user_id'].unique())
        if not target_user_ids:
            print("Il file degli utenti è vuoto o la colonna 'user_id' non è presente.")
            return None
    except (FileNotFoundError, KeyError) as e:
        print(f"ERRORE nel caricare il file utenti '{users_path}': {e}")
        return None

    # 2. Carica i dati di supporto UNA SOLA VOLTA
    game_titles_db = _load_game_titles(games_path)
    game_metadata_db = _load_game_metadata(metadata_path)
    
    if not game_titles_db or not game_metadata_db:
        print("Impossibile caricare i file di dati necessari. Interruzione.")
        return None

    # 3. Processa tutte le recensioni in un colpo solo
    user_reviews_map = _get_all_user_reviews_in_batch(recommendations_path, target_user_ids)
    
    if user_reviews_map is None:
        print("Interruzione a causa di un errore nel processare le recensioni.")
        return None

    # 4. Genera i prompt per ogni utente
    all_user_prompts = {}
    print("\nGenerazione dei prompt per gli utenti trovati...")
    
    # Itera sugli utenti per cui abbiamo trovato recensioni
    for user_id in tqdm(user_reviews_map.keys(), desc="Creazione Profili"):
        app_ids = user_reviews_map[user_id]
        
        user_games_data = []
        for app_id in app_ids:
            title = game_titles_db.get(app_id)
            description = game_metadata_db.get(app_id)
            
            if title or description:
                user_games_data.append({
                    'title': title or 'No title available',
                    'description': description or 'No description available'
                })
        
        if user_games_data:
            prompt = _create_llm_prompt(user_games_data)
            # Converte user_id a stringa per la chiave JSON
            all_user_prompts[str(user_id)] = prompt

    return all_user_prompts


# --- Blocco di Esecuzione Principale ---
if __name__ == "__main__":
    print("--- Inizio Processo di Profilazione Massiva degli Utenti ---")
    
    # Esegui la funzione principale che fa tutto il lavoro
    generated_prompts = generate_all_prompts_in_batch(
        users_path=USERS_PATH,
        games_path=GAMES_PATH,
        metadata_path=METADATA_PATH,
        recommendations_path=RECOMMENDATIONS_PATH
    )
    
    if generated_prompts:
        # 5. Salva i risultati in un unico file JSON
        try:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(generated_prompts, f, indent=4)
            print(f"\n--- PROCESSO COMPLETATO ---")
            print(f"Trovati e processati {len(generated_prompts)} utenti.")
            print(f"I prompt sono stati salvati in: '{OUTPUT_PATH}'")
        except IOError as e:
            print(f"ERRORE durante il salvataggio del file di output: {e}")
    else:
        print("\nNessun prompt è stato generato. Controlla i log per errori.")