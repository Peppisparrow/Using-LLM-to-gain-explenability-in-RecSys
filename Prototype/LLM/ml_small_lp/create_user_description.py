import re
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

data_dir = Path('/Users/lucapagano/Downloads/DATA/ML_small')
RECOMMENDATIONS_PATH = data_dir / 'tuning' / 'train_recommendations.csv'
#PLOTS_PATH = data_dir / 'app_plots.csv'
#OUTPUT_PATH = data_dir / 'user_prompts.parquet'
PLOTS_PATH = '/Users/lucapagano/Developer/RecSys/Using-LLM-to-gain-explenability-in-RecSys/Dataset/movielens-latest-small/movies_with_plots.csv'
OUTPUT_PATH = data_dir / 'user_prompts_DESC.parquet'

PROMPT = '''
    You're a film analyst. Based on a user's list of favorite movies, write a detailed and thoughtful description of their movie taste. Avoid vague or generic summaries and be sufficiently concise.  Start by identifying the dominant genres in the list to establish a foundation. Then, explore deeper patterns—common themes, character types, tones, or storytelling styles that appear across the films.  Also, point out any contrasting elements (e.g., quiet dramas vs. action-heavy blockbusters) that may indicate different clusters of interest or a wide range of appeal.  Your final output should be a comprehensive profile that captures both the user's core preferences and the full scope of what they find engaging in film. 
    '''

def _load_app_plots(path):
    """
    Loads app plots from the app_plots.csv file into a dictionary.
    
    Args:
        path (str): The path to the app_plots.csv file.

    Returns:
        dict: A dictionary mapping app_id to its plot. Returns None on error.
    """
    try:
        print("Loading app plots...")
        df = pd.read_csv(path, usecols=['app_id', 'new_title', 'plot', 'genres'])
        df = df.set_index('app_id')
        # Now convert the DataFrame to a dictionary that has as keys
        return df.T.to_dict()
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR loading app plots: {e}")
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
    """
    Crea il prompt finale per l'LLM data una history "games_data".
    """
    
    instruction = PROMPT
    formatted_movie_list = []
    for movie in games_data:
        clean_description = " ".join(str(movie['plot']).split())
        movie_entry = (
            f"Title: {movie['title']}\n"
            f"Description: {clean_description}\n"
            f"Genre: {movie['genres']}\n"
        )
        formatted_movie_list.append(movie_entry)
    
    data_section = "\n\n---\n\n".join(formatted_movie_list)
    final_prompt = (
        f"{instruction}\n\n"
        f"Please analyze the following movies:\n\n"
        f"{data_section}\n\n"
    )
    return final_prompt

# --- FUNZIONE PRINCIPALE OTTIMIZZATA ---

def generate_all_prompts_in_batch(recommendations_path, games_path):
    """
    Orchestra l'intero processo di generazione dei prompt per TUTTI gli utenti
    specificati nel file users_path.
    """
    # 1. Carica la lista di utenti target dal file users.csv
    try:
        users_df = pd.read_csv(recommendations_path, usecols=['user_id'])
        # Usare un set per ricerche O(1), molto più veloce di una lista
        target_user_ids = set(users_df['user_id'].unique())
        if not target_user_ids:
            print("Il file degli utenti è vuoto o la colonna 'user_id' non è presente.")
            return None
    except (FileNotFoundError, KeyError) as e:
        print(f"ERRORE nel caricare il file utenti '{recommendations_path}': {e}")
        return None

    movies = _load_app_plots(games_path)

    if not movies:
        raise ValueError("Movies data is empty or not loaded correctly.")

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
            title = movies[app_id]['new_title']
            description = movies[app_id]['plot']
            genres = movies[app_id]['genres']


            if title or description:
                user_games_data.append({
                    'title': title or 'No title available',
                    'plot': description or 'No description available',
                    'genres': genres or 'No genres available'
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
        recommendations_path=RECOMMENDATIONS_PATH,
        games_path=PLOTS_PATH
    )
    
    if generated_prompts:
       # Let's save as a parquet file for better performance
        print(f"\nSalvataggio di {len(generated_prompts)} prompt in '{OUTPUT_PATH}'...")
        try:
            # Converti il dizionario in un DataFrame e salva come Parquet
            df_prompts = pd.DataFrame.from_dict(generated_prompts, orient='index', columns=['prompt'])
            df_prompts.index.name = 'user_id'  # Imposta l'indice come user_id
            df_prompts.to_parquet(OUTPUT_PATH, engine='pyarrow')
            print("✅ Prompts salvati con successo!")
        except Exception as e:
            print(f"ERRORE nel salvare i prompt: {e}")
    else:
        print("\nNessun prompt è stato generato. Controlla i log per errori.")