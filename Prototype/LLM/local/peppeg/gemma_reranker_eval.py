import torch
import json
import os
import re
import numpy as np
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
import pandas as pd
# --- 1. CONFIGURAZIONE ---
# Percorso al tuo modello fine-tuned e salvato

# FINETUNED_MODEL_PATH = "Dataset/ml/ml-latest-small/tuning/gemma/gemma_30_candidates_grpo/checkpoint-151"ù
# FINETUNED_MODEL_PATH ="Dataset/ml/ml-latest-small/tuning/gemma4b/first_experiments_LORA/checkpoint-3223"
# TEST_SET_PATH = "Dataset/ml_small/tuning/histories_gemma_recommender_train+target.json"
# TARGET_MOVIES_PATH = "Dataset/ml_small/tuning/histories_gemma_recommender_test.json"
# candidate_items_path = "Dataset/ml_small/final/candidate_items_30_eval_with_titles.csv"
# output_report_path = "Dataset/ml_small/final/gemma_sft_results_30_candidates.json"

FINETUNED_MODEL_PATH = "Dataset/ml/ml-latest-small/tuning/gemma/gemma_50_candidates_grpo_fixed/checkpoint-605"
TEST_SET_PATH = "Dataset/ml_small/tuning/histories_gemma_recommender_train.json"
TARGET_MOVIES_PATH = "Dataset/ml_small/tuning/histories_gemma_recommender_target.json"
candidate_items_path = "Dataset/ml_small/tuning/candidate_items_50_train_hits.csv"
output_report_path = "Dataset/ml_small/final/gemma_50_candidates_grpo_train_50_hcheckpoint-605.json"

# --- 2. FUNZIONI HELPER (copiale dal tuo script di training) ---
import re
import string

# Definiamo un set di articoli in varie lingue da ignorare.
# Usare un set è molto più veloce per le ricerche.
ARTICLES_TO_REMOVE = {
    'the', 'a', 'an', 'le', 'la', 'les', 'l\'', 'il', 'lo', 'i', 'gli', 'un', 'una',
    'un\'', 'el', 'los', 'las', 'ein', 'eine', 'der', 'die', 'das', 'aka'
}

# Funzione per pulire e normalizzare una stringa di testo
def normalize_text(text: str) -> str:
    """
    Converte il testo in minuscolo, rimuove la punteggiatura, gli articoli comuni
    e normalizza gli spazi.
    """
    # 1. Minuscolo
    text = text.lower()
    text = text.replace("’", "'")
    # 2. Rimuovi la punteggiatura
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Rimuovi gli articoli
    words = text.split()
    words = [word for word in words if word not in ARTICLES_TO_REMOVE]
    text = ' '.join(words)
    # 4. Normalizza gli spazi (rimuove spazi multipli)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Funzione per analizzare i titoli complessi e restituire tutte le varianti
def parse_complex_title(title_string: str) -> list[str]:
    """
    Analizza un titolo complesso per estrarre le sue varianti.
    Gestisce:
    - Formato "Nome, Il" -> "Il Nome"
    - Titoli alternativi tra parentesi.
    - Anno di uscita.
    Restituisce una lista di titoli normalizzati.
    """
    # Rimuovi l'anno di uscita, es. (1995)
    title_string = re.sub(r'\s*\(\d{4}\)$', '', title_string).strip()

    # Separa i titoli principali e quelli tra parentesi
    # es. "City of Lost Children, The (Cité des enfants perdus, La)" ->
    # ["City of Lost Children, The", "Cité des enfants perdus, La"]
    potential_titles = re.split(r'\(|\)', title_string)
    potential_titles = [t.strip() for t in potential_titles if t.strip()]

    cleaned_titles = set() # Usiamo un set per evitare duplicati
    for title in potential_titles:
        # Gestisci il formato "Title, The" -> "The Title"
        match = re.match(r'^(.*?), (the|a|an|le|la|il|lo|el|der|die|das)$', title, re.IGNORECASE)
        if match:
            # Riordina: es. "City of Lost Children, The" -> "The City of Lost Children"
            reordered_title = f"{match.group(2)} {match.group(1)}"
            cleaned_titles.add(normalize_text(reordered_title))
        else:
            cleaned_titles.add(normalize_text(title))
            
    return list(cleaned_titles)

def dcg_score(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(len(scores), dtype=np.float64) + 2)))

def ndcg_at_10(ranked_relevance_scores, num_relevant_items):
    rank_dcg = dcg_score(ranked_relevance_scores)
    if rank_dcg == 0.0: return 0.0
    ideal_scores = np.ones(min(num_relevant_items, 10))
    ideal_dcg = dcg_score(ideal_scores)
    if ideal_dcg == 0.0: return 0.0
    return rank_dcg / ideal_dcg

# --- 3. CARICAMENTO MODELLO E DATI ---
print("🚀 Caricamento modello fine-tuned...")
print("Tentativo di caricamento con transformers standard...")
model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    FINETUNED_MODEL_PATH,
    trust_remote_code=True
)

print("🔄 Caricamento dati di test e target...")
with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
    test_histories = json.load(f)
with open(TARGET_MOVIES_PATH, 'r', encoding='utf-8') as f:
    target_data = json.load(f)
target_map = {item['user_id']: item['history'] for item in target_data}


candidate_items = pd.read_csv(candidate_items_path)
candidates_map = {}
# Raggruppa il DataFrame per user_id per elaborare i candidati di ogni utente
for user_id, group in tqdm(candidate_items.groupby('user_id'), desc="Processing candidates"):
    # Converte il gruppo di righe in una lista di dizionari per un accesso più facile
    candidates_map[user_id] = group.to_dict('records')

# --- 4. FASE DI INFERENZA E VALUTAZIONE ---
all_ndcg_scores = []
all_hits = []
results = []


for i, history_item in enumerate(tqdm(test_histories, desc="Evaluating on Test Set")):
    MAX_HISTORY_ITEMS=10
    if history_item['user_id'] not in target_map:
        continue
    target_history = target_map[history_item['user_id']]
    titles = []
    target_titles = []

    for item in history_item['history']:
        titles.append(item['title'])
    
    for item in target_history:
        target_titles.append(item['title'])

    if len(titles) < 5:
        continue

    prompt_titles = titles
    target_titles = target_titles

    if not target_titles and prompt_titles:
        target_titles.append(prompt_titles.pop())
    
    interactions = []
    full_history = history_item['history']
    if len(full_history) > MAX_HISTORY_ITEMS:
        sampled_history = random.sample(full_history, MAX_HISTORY_ITEMS)
    else:
        sampled_history = full_history
    for item in sampled_history:
        interactions.append(
            f"Title: {item['title']} \n"
            f"Description: {item['description']} \n"
            f"Genre: {item['genres']} \n"
            f"Rating: {item['rating']}"
            )

    user_id = history_item['user_id']
    user_candidates = candidates_map.get(user_id, [])

    candidate_strings = []
    for candidate in user_candidates:
        title = candidate.get('title', 'N/A')
        genres = candidate.get('genres', 'N/A')
        description = candidate.get('description', 'No description available.')
        candidate_str = (
            f"Title: {title}"
        )
        candidate_strings.append(candidate_str)
    formatted_candidates = "\n\n".join(candidate_strings)
    formatted_interactions = '\n\n'.join(interactions)
    prompt_text = (
        f"The user recently watched the following movies:\n"
        f"{formatted_interactions}\n\n"
        "Your task is to act as a filter and ranker. From the following list of candidate movies, you must select and rank the ten best movies for the user.\n"
        f"**Candidate Movies**:\n\n{formatted_candidates}\n\n"
        f"Use the following format for your response:\n"
        f"**User's summary**:[Brief summary of the user's preferences]\n"
        f"**Recommended movies**:\n1. [First movie title]\n...\n10. [Tenth movie title]\n\n"
        "Please provide only the titles of the ten recommended movies.\n"
        "⚠️ **CRITICAL INSTRUCTION**: You MUST choose exclusively from the 'Candidate Movies' list. Do NOT recommend any movie that is not in that list." # Istruzione critica ripetuta alla fine
    )
    messages = [
            {"role": "system", "content": "You are a helpful movie recommendation assistant."},
            {"role": "user", "content": prompt_text},
        ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,      # Lunghezza massima della raccomandazione
            use_cache=True,
            do_sample=True,          # NECESSARIO per usare la temperatura
            temperature=2.0,         # Come richiesto
            top_p=0.95,               # Come richiesto
            pad_token_id=tokenizer.pad_token_id
        )

    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    # --- INIZIO LOGICA DI VALUTAZIONE (simile alla reward_function) ---

    # 1. Parsing dell'output
    try:
        recs_section = generated_text.split("**Recommended movies**:")[1]
        ranked_list_raw = re.findall(r"^\s*\d+\.\s*[\"\'\*\[]?\s*(.*?)\s*[\"\'\*\]]?\s*(?:\(|$)", recs_section, re.MULTILINE)[:10]
    except IndexError:
        print(f"⚠️ Nessuna raccomandazione trovata per l'utente {generated_text}.")
        ranked_list_raw = []

    if not ranked_list_raw:
        all_ndcg_scores.append(0.0)
        continue

    normalized_target_set = set()
    for title_string in target_titles:
        variants = parse_complex_title(title_string)
        for variant in variants:
            if variant: normalized_target_set.add(variant)

    relevance_scores = []
    for rec_line in ranked_list_raw:
        clean_title = rec_line.replace('**', '').strip()
        possible_variants = parse_complex_title(clean_title)
        is_match = any(variant in normalized_target_set for variant in possible_variants)
        relevance_scores.append(1.0 if is_match else 0.0)
    # print what are the titles that matched
    # for idx, score in enumerate(relevance_scores):
    #     if score == 1.0:
    #         print(f"Matched title: {ranked_list_raw[idx]}")
    # print(f"Candidate titles: {[item['title'] for item in user_candidates]}")
    # print(f"Recomendation titles: {ranked_list_raw}")
    # print(f"Target titles: {target_titles}")
    score = ndcg_at_10(relevance_scores, len(normalized_target_set))
    hit = sum(relevance_scores)
    all_ndcg_scores.append(score)
    all_hits.append(hit)

    # Salva i risultati per un'analisi qualitativa
    results.append({
        "user_id": user_id,
        "recommendations": ranked_list_raw,
        "ground_truth": target_titles,
        "ndcg_score": score,
        "hits": hit
    })
    # <-- NUOVO BLOCCO DEBUG: si attiva solo se DEBUG_MODE è True
    if (i+1) % 10 == 0:
        print(f"Avg NDCG after {i+1} users: {np.mean(all_ndcg_scores):.4f}")
    

# --- 5. VISUALIZZAZIONE RISULTATI FINALI ---
average_ndcg = np.mean(all_ndcg_scores)
print("\n--- 📈 RISULTATI DELLA VALUTAZIONE ---")
print(f"Valutati {len(all_ndcg_scores)} utenti dal set di test.")
print(f"NDCG@10 Medio: {average_ndcg:.4f}")
print("----------------------------------------")

# Stampa qualche esempio per un'analisi qualitativa
print("\n--- 🔍 Esempi di Raccomandazioni ---")
for res in results:
    print(f"User ID: {res['user_id']}")
    print(f"  Punteggio NDCG: {res['ndcg_score']:.4f}, Hits: {res['hits']}")

# Salva i risultati completi in un file JSON per ulteriori analisi

with open(output_report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
