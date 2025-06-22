import pandas as pd
import json

# I percorsi possono essere definiti qui come default,
# ma la funzione principale li accetterà come argomenti per maggiore flessibilità.
METADATA_PATH = 'games_metadata.json'
RECOMMENDATIONS_PATH = 'recommendations.csv'
GAMES_PATH = 'games.csv'

# --- Funzioni Helper "Private" ---
# Queste funzioni sono usate internamente dalla nostra funzione principale.

def _load_game_titles(path):
    # (codice invariato)
    try:
        df = pd.read_csv(path, usecols=['app_id', 'title'])
        return df.set_index('app_id')['title'].to_dict()
    except (FileNotFoundError, KeyError) as e:
        print(f"ERRORE durante il caricamento dei titoli: {e}")
        return None

def _load_game_metadata(path):
    # (codice invariato)
    metadata_dict = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'app_id' in data and 'description' in data:
                        metadata_dict[data['app_id']] = data['description']
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print(f"ERRORE: File non trovato: '{path}'")
        return None
    return metadata_dict

def _get_user_reviewed_game_data(user_id, recommendations_path, titles_dict, metadata_dict):
    # (codice aggiornato dall'utente, che gestisce i dati mancanti)
    user_app_ids = set()
    try:
        chunk_iter = pd.read_csv(
            recommendations_path, chunksize=100000, usecols=['user_id', 'app_id'])
        for chunk in chunk_iter:
            user_reviews = chunk[chunk['user_id'] == user_id]
            if not user_reviews.empty:
                user_app_ids.update(user_reviews['app_id'].tolist())
    except FileNotFoundError:
        print(f"ERRORE: File non trovato: '{recommendations_path}'")
        return []
    if not user_app_ids: return []
    
    user_games_data = []
    for app_id in user_app_ids:
        title = titles_dict.get(app_id)
        description = metadata_dict.get(app_id)
        # Aggiunge il gioco se ha almeno un titolo o una descrizione
        if title or description:
             user_games_data.append({
                'title': title or 'No title available',
                'description': description or 'No description available'
            })
    return user_games_data


def _create_llm_prompt(games_data):
    # (codice invariato)
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
        f"---END OF DATA---\n\n"
        f"Based on the data above, provide a summary of this user's tastes:"
    )
    return final_prompt


# --- FUNZIONE PRINCIPALE "PUBBLICA" ---
# Questa è la funzione che importerai e chiamerai da altri script.

def generate_user_taste_prompt(user_id, 
                               games_path=GAMES_PATH, 
                               metadata_path=METADATA_PATH, 
                               recommendations_path=RECOMMENDATIONS_PATH):
    """
    Orchestra l'intero processo: carica i dati, trova i giochi di un utente,
    e genera il prompt finale per un LLM.

    Args:
        user_id (int): L'ID dell'utente da analizzare.
        games_path (str): Percorso del file games.csv.
        metadata_path (str): Percorso del file games_metadata.jsonl.
        recommendations_path (str): Percorso del file recommendations.csv.

    Returns:
        str: La stringa del prompt pronta per l'LLM, o None se non si trovano dati.
    """
    print(f"--- Inizio profilazione per l'utente: {user_id} ---")
    
    # 1. Carica i dati di supporto
    game_titles_db = _load_game_titles(games_path)
    game_descriptions_db = _load_game_metadata(metadata_path)
    
    if not game_titles_db or not game_descriptions_db:
        print("Impossibile caricare i file di dati necessari. Interruzione.")
        return None
        
    # 2. Ottieni i dati per l'utente specifico
    games_data = _get_user_reviewed_game_data(
        user_id=user_id,
        recommendations_path=recommendations_path,
        titles_dict=game_titles_db,
        metadata_dict=game_descriptions_db
    )
    
    # 3. Se i dati esistono, crea e restituisci il prompt
    if games_data:
        print(f"Trovati {len(games_data)} giochi per l'utente. Creazione del prompt...")
        llm_prompt = _create_llm_prompt(games_data)
        print("Prompt generato con successo.")
        return llm_prompt
    else:
        print(f"Nessun dato trovato per l'utente {user_id}. Impossibile generare il prompt.")
        return None


# --- Blocco di Esecuzione di Esempio ---
# Questo codice viene eseguito solo quando lanci `python profiler.py` direttamente.
# Serve per testare la funzione `generate_user_taste_prompt`.
if __name__ == "__main__":
    TARGET_USER_ID = 10127955
    
    # Chiamiamo la nostra nuova funzione principale
    final_prompt = generate_user_taste_prompt(TARGET_USER_ID)
    
    if final_prompt:
        print("\n\n--- PROMPT GENERATO PER L'LLM (in inglese) ---")
        print(final_prompt)