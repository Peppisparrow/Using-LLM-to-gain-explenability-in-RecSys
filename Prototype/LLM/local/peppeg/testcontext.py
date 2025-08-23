import json
from transformers import AutoTokenizer
from tqdm import tqdm
import random
# --- 1. CONFIGURAZIONE ---

# Modello e la sua context window massima
MODEL_ID = "google/gemma-3-4b-it"
CONTEXT_WINDOW = 8192  # Gemma 3 ha una context window di 8192 token

# Percorsi ai tuoi file di dati
PROMPT_PATH = "Dataset/ml/ml-latest-small/tuning/histories_gemma_recommender.json"

# --- 2. CARICAMENTO TOKENIZER E DATI ---

print(f"🚀 Caricamento tokenizer per il modello: {MODEL_ID}")
# Usiamo AutoTokenizer, è sufficiente per contare i token
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"🔄 Caricamento dati da: {PROMPT_PATH}")
try:
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        histories = json.load(f)
except FileNotFoundError:
    print(f"🚨 ERRORE: File non trovato in '{PROMPT_PATH}'.")
    exit()

# --- 3. CONTROLLO DEI PROMPT ---

max_prompt_length = 0
long_prompts_count = 0
problematic_users = []

print("\n🔍 Analisi della lunghezza dei prompt...")
for history_item in tqdm(histories, desc="Checking Prompts"):
    user_id = history_item['user_id']
    full_history = history_item['history']
    MAX_HISTORY_ITEMS=40
    if len(full_history) > MAX_HISTORY_ITEMS:
        sampled_history = random.sample(full_history, MAX_HISTORY_ITEMS)

    # Ricostruisci il prompt esattamente come nello script di training
    interactions = '\n\n'.join([
        f"Title: {item['title']}\nDescription: {item['description']}\nGenre: {item['genres']}\nRating: {item['rating']}"
        for item in sampled_history
    ])
    
    prompt_text = (
        f"The user recently watched the following movies:\n"
        f"{interactions}\n\n"
        "Please recommend a list of ten movies that the user might like based on their watch history.\n"
        "⚠️ IMPORTANT: Do NOT recommend any movie from the list above. The recommendation must be a different movie.\n"
        f"Before making the recommendations, please provide a brief summary of the user's preferences based on their watch history.\n\n"
        f"Use the following format for your response:\n"
        f"**User's summary**:[Brief summary of the user's preferences]\n"
        f"**Recommended movies**:\n1. [First movie title]\n2. [Second movie title]\n3. [Third movie title]\n4. [Fourth movie title]\n5. [Fifth movie title]\n6. [Sixth movie title]\n7. [Seventh movie title]\n8. [Eighth movie title]\n9. [Ninth movie title]\n10. [Tenth movie title]\n"
    )

    # Tokenizza il prompt e calcola la lunghezza
    # .encode() è un modo semplice e veloce per ottenere gli ID dei token
    token_count = len(tokenizer.encode(prompt_text))

    # Aggiorna la lunghezza massima trovata
    if token_count > max_prompt_length:
        max_prompt_length = token_count

    # Controlla se supera la context window
    if token_count > CONTEXT_WINDOW:
        long_prompts_count += 1
        problematic_users.append({'user_id': user_id, 'length': token_count})
        # Stampa subito un avviso per i prompt problematici
        print(f"\n⚠️ ATTENZIONE: Il prompt per l'utente {user_id} supera la context window! Lunghezza: {token_count} token.")


# --- 4. RIEPILOGO FINALE ---

print("\n" + "="*50)
print("📊 ANALISI COMPLETATA")
print("="*50)
print(f"Lunghezza massima del prompt trovata: {max_prompt_length} token")
print(f"Context window del modello: {CONTEXT_WINDOW} token")

if long_prompts_count > 0:
    print(f"🚨 Trovati {long_prompts_count} prompt che superano la context window.")
    print("Utenti con prompt troppo lunghi:")
    for user in problematic_users:
        print(f"  - ID Utente: {user['user_id']}, Lunghezza: {user['length']}")
else:
    print("✅ Ottimo! Tutti i prompt rientrano nella context window del modello.")