import torch
import json
import os
import re
import numpy as np
from unsloth import FastLanguageModel
from tqdm import tqdm
import random

# --- 1. CONFIGURAZIONE ---
# Percorso al tuo modello fine-tuned e salvato
FINETUNED_MODEL_PATH = "Dataset/ml/ml-latest-small/tuning/gemma/gemma_recommender_300/final_model"
TEST_SET_PATH = "Dataset/ml/ml-latest-small/tuning/histories_gemma_recommender_train+target.json"
TARGET_MOVIES_PATH = "Dataset/ml/ml-latest-small/tuning/histories_gemma_recommender_test.json"

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
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=FINETUNED_MODEL_PATH,
    dtype=None,
)

print("🔄 Caricamento dati di test e target...")
with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
    test_histories = json.load(f)
with open(TARGET_MOVIES_PATH, 'r', encoding='utf-8') as f:
    target_data = json.load(f)
target_map = {item['user_id']: [t['title'] for t in item['history']] for item in target_data}


# --- 4. FASE DI INFERENZA E VALUTAZIONE ---
all_ndcg_scores = []
results = []

for i, history_item in enumerate(tqdm(test_histories, desc="Evaluating on Test Set")):
    user_id = history_item['user_id']
    
    # Salta se l'utente non ha un target nel nostro file
    if user_id not in target_map:
        continue
        
    ground_truth_titles = target_map[user_id]

    MAX_HISTORY_ITEMS = 300
    full_history = history_item['history']
    if len(full_history) > MAX_HISTORY_ITEMS:
        sampled_history = random.sample(full_history, MAX_HISTORY_ITEMS)
    else:
        sampled_history = full_history
    
    # Crea il prompt (esattamente come nel training)
    interactions = '\n\n'.join([f"Title: {item['title']}\nDescription: {item['description']}\nGenres: {item['genres']}\nRating: {item['rating']}" for item in sampled_history])
    prompt_text = (
        f"The user recently watched the following movies:\n"
        f"{interactions}\n\n"
        "Please recommend a list of ten movies that the user might like based on their watch history.\n"
        "⚠️ IMPORTANT: Do NOT recommend any movie from the list above. The recommendation must be a different movie."
        f"Before making the recommendations, please provide a brief summary of the user's preferences based on their watch history.\n\n"
        f"Use the following format for your response:\n"
        f"**User's summary**:[Brief summary of the user's preferences]\n"
        f"**Recommended movies**:\n1. [First movie title]\n2. [Second movie title]\n3. [Third movie title]\n4. [Fourth movie title]\n5. [Fifth movie title]\n6. [Sixth movie title]\n7. [Seventh movie title]\n8. [Eighth movie title]\n9. [Ninth movie title]\n10. [Tenth movie title]\n"
        "Please provide only the titles of the recommended movies.\n"
    )
    messages = [
            {"role": "system", "content": "You are a helpful movie recommendation assistant."},
            {"role": "user", "content": prompt_text},
        ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[-1]

    # with torch.inference_mode():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=400,      # Lunghezza massima della raccomandazione
    #         use_cache=True,
    #         do_sample=True,          # NECESSARIO per usare la temperatura
    #         temperature=1.0,         # Come richiesto
    #         pad_token_id=tokenizer.pad_token_id
    #     )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,      # Lunghezza massima della raccomandazione
            use_cache=True,
            do_sample=False,          # NECESSARIO per usare la temperatura
            num_beams=3,            # Usa beam search per una generazione più coerente
        ) 

    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    # --- INIZIO LOGICA DI VALUTAZIONE (simile alla reward_function) ---

    # 1. Parsing dell'output
    try:
        recs_section = generated_text.split("**Recommended movies**:")[1]
        ranked_list_raw = re.findall(r"^\s*\d+\.\s*(.*)", recs_section, re.MULTILINE)[:10]
    except IndexError:
        print(f"⚠️ Nessuna raccomandazione trovata per l'utente {generated_text}.")
        ranked_list_raw = []

    if not ranked_list_raw:
        all_ndcg_scores.append(0.0)
        continue

    # 2. Preparazione target
    normalized_target_set = set()
    for title_string in ground_truth_titles:
        variants = parse_complex_title(title_string)
        for variant in variants:
            if variant: normalized_target_set.add(variant)

    relevance_scores = []
    for rec_line in ranked_list_raw:
        # 1. Isola la parte di testo che contiene il titolo (prima di ':' o '-')
        title_part = rec_line.split(':')[0].split('–')[0]
        # 2. Pulisci da caratteri di formattazione come '**'
        clean_title = title_part.replace('**', '').strip()
        # 3. USA LA STESSA FUNZIONE DEI TARGET per gestire anno, titoli alternativi, etc.
        possible_variants = parse_complex_title(clean_title)
        # 4. Controlla se una QUALSIASI delle varianti generate matcha con il set dei target
        is_match = any(variant in normalized_target_set for variant in possible_variants)
        relevance_scores.append(1.0 if is_match else 0.0)
    
    score = ndcg_at_10(relevance_scores, len(normalized_target_set))
    all_ndcg_scores.append(score)
    results.append({"user_id": user_id, "recommendations": ranked_list_raw, "ground_truth": ground_truth_titles, "ndcg_score": score})

    # Salva i risultati per un'analisi qualitativa
    results.append({
        "user_id": user_id,
        "recommendations": ranked_list_raw,
        "ground_truth": ground_truth_titles,
        "ndcg_score": score
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
for res in results[:5]:
    print(f"User ID: {res['user_id']}")
    print(f"  Raccomandazioni Generate: {res['recommendations']}")
    print(f"  Verità (Ground Truth): {res['ground_truth']}")
    print(f"  Punteggio NDCG: {res['ndcg_score']:.4f}\n")