import torch
import json
import os
import random
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

# --- 1. Definisci Percorsi, Costanti e Parametri di Training ---
PRE_FLIGHT_CHECK = False  # Imposta a False per saltare il pre-flight check
model_id = "unsloth/gemma-3-4b-it"
prompt_path = "Dataset/ml_small/tuning/histories_gemma_recommender_train.json"
candidate_items_path = "Dataset/ml_small/tuning/candidate_items_50_train_hits.csv"
target_movies_path = "Dataset/ml_small/tuning/histories_gemma_recommender_target.json"
output_dir = "Dataset/ml/ml-latest-small/tuning/gemma/gemma_50_candidates_grpo_fixed"
output_model_path = f"{output_dir}/final_model"

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
from typing import Dict, List, Any, Tuple
import torch
    
# --- 2. Preparazione dei Dati (con split 90/10) ---
print(f"🔄 Loading and preparing user movie data from '{prompt_path}'...")
try:
    with open(prompt_path, 'r', encoding='utf-8') as f:
        histories = json.load(f)
except FileNotFoundError:
    print(f"🚨 ERROR: Input file not found at '{prompt_path}'.")
    exit()

try:
    with open(target_movies_path, 'r', encoding='utf-8') as f:
        target = json.load(f)
except FileNotFoundError:
    print(f"🚨 ERROR: Input file not found at '{prompt_path}'.")
    exit()

candidate_items = pd.read_csv(candidate_items_path)
candidates_map = {}
# Raggruppa il DataFrame per user_id per elaborare i candidati di ogni utente
for user_id, group in tqdm(candidate_items.groupby('user_id'), desc="Processing candidates"):
    # Converte il gruppo di righe in una lista di dizionari per un accesso più facile
    candidates_map[user_id] = group.to_dict('records')

print("✅ Candidate map created.")

# Struttura dati per il training: lista di dizionari
training_data = []
# usa solo i primi 5 utenti per il test
# Itera su ogni utente per creare i dati di training
print("Creazione della mappa dei target per una ricerca efficiente...")
target_map = {item['user_id']: item['history'] for item in target}

for i, history in tqdm(enumerate(histories), desc="Preparing Data"):
    MAX_HISTORY_ITEMS=10
    if history['user_id'] not in target_map:
        continue # Salta questo utente e passa al prossimo
    target_history = target_map[history['user_id']]
    titles = []
    target_titles = []

    for item in history['history']:
        titles.append(item['title'])
    for item in target_history:
        target_titles.append(item['title'])

    if len(titles) < 5:
        continue

    prompt_titles = titles
    target_titles = target_titles

    # Se la lista target è vuota (per arrotondamento), sposta un film
    if not target_titles and prompt_titles:
        target_titles.append(prompt_titles.pop())
    
    # Formatta il prompt
    interactions = []
    full_history = history['history']
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
    #CREA I formatted_candidates USANDO candidate_items
    user_id = history['user_id']
    user_candidates = candidates_map.get(user_id, [])

    candidate_strings = []
    # Itera sui film candidati per l'utente e formattali
    for candidate in user_candidates:
        # Gestisce il caso in cui alcuni campi potrebbero essere mancanti (NaN in pandas)
        title = candidate.get('title', 'N/A')
        genres = candidate.get('genres', 'N/A')
        description = candidate.get('description', 'No description available.')
        
        # Crea una stringa formattata per ogni candidato
        candidate_str = (
            f"Title: {title}"
        )
        candidate_strings.append(candidate_str)
    
    # Unisci le stringhe di tutti i candidati, separandole chiaramente
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

    # Crea il messaggio nel formato chat
    # Usiamo il formato `apply_chat_template` senza tokenizzazione qui
    # perché il trainer lo gestirà internamente.
    messages = [
        {"role": "system", "content": "You are a helpful movie recommendation assistant."},
        {"role": "user", "content": prompt_text},
    ]

    # Scegliamo un titolo target a caso come riferimento per la ricompensa
    # La funzione di ricompensa controllerà la presenza in *tutti* i target_titles
    # Usiamo un solo titolo come "completion di riferimento" per il logging
    reference_completion = random.choice(target_titles)

    training_data.append({
        "user_id": history['user_id'],
        "prompt": messages,
        "reference_completion": f"Based on your watch history, a great recommendation would be **{reference_completion}**.",
        "target_movies": target_titles # Lista di film target per la ricompensa
    })

# Converti in un oggetto Dataset di Hugging Face
train_dataset = Dataset.from_list(training_data)
print(f"✅ Prepared dataset with {len(train_dataset)} examples.")

# --- 3. Carica Modello e Tokenizer con Unsloth ---
print(f"🚀 Loading model '{model_id}' with Unsloth for 4-bit training...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    dtype=torch.bfloat16,
    load_in_4bit = False,
)

# Aggiungi LoRA al modello per un training efficiente (PEFT)
rank = 8
model = FastLanguageModel.get_peft_model(
    model,
    r=rank, # Rank LoRA, valori comuni sono 8, 16, 32, 64
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=rank * 2,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# Imposta il template della chat per Gemma
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma",
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "model"},
)
# --- AGGIUNTA: PRE-FLIGHT CHECK - Valutazione Zero-Shot del Modello Base ---
if PRE_FLIGHT_CHECK:
    print("\n" + "#"*20 + " 🧪 INIZIO PRE-FLIGHT CHECK 🧪 " + "#"*20)
    print("Valutiamo la capacità del modello base di seguire le istruzioni prima del fine-tuning.")

    # Numero di esempi da testare
    NUM_TEST_SAMPLES = 1
    total_generated_recs = 0
    total_valid_recs = 0

    # Assicuriamoci che il modello sia in modalità valutazione
    model.eval()

    with torch.no_grad(): # Disabilita il calcolo dei gradienti per l'inferenza
        for i in range(min(NUM_TEST_SAMPLES, len(train_dataset))):
            print("\n" + "-"*20 + f" Test Case #{i+1} " + "-"*20)
            
            # 1. Recupera i dati per un utente
            test_case = train_dataset[i]
            user_id = histories[i]['user_id'] # Recupera l'user_id originale
            
            # 2. Prepara il prompt per l'inferenza
            prompt_for_model = tokenizer.apply_chat_template(
                test_case["prompt"], 
                tokenize=False, 
                add_generation_prompt=True
            )

            # 3. Recupera la lista dei titoli candidati originali per questo utente
            candidate_list_for_user = candidates_map.get(user_id, [])
            # Normalizza i titoli dei candidati per un confronto corretto e mettili in un set per efficienza
            normalized_candidate_set = set()
            for cand in candidate_list_for_user:
                title = cand.get('title')
                if title:
                    variants = parse_complex_title(title)
                    for v in variants:
                        normalized_candidate_set.add(v)

            # 4. Genera una risposta dal modello base
            inputs = tokenizer([prompt_for_model], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=300, use_cache=True, pad_token_id=tokenizer.pad_token_id)
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            input_length = inputs.input_ids.shape[1]
            new_tokens = outputs[0, input_length:]
            # 4. Decodifica SOLO i nuovi token per ottenere la risposta pulita
            assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            print(f"👤 USER ID: {user_id}")
            print("\n--- RISPOSTA DEL MODELLO BASE ---")
            print(assistant_response)
            
            # 5. Analizza la risposta: estrai le raccomandazioni e confrontale con i candidati
            try:
                recs_section = assistant_response.split("**Recommended movies**:")[1]
                ranked_list_raw = re.findall(r"^\s*\d+\.\s*[\"\'\*\[]?\s*(.*?)\s*[\"\'\*\]]?\s*(?:\(|$)", recs_section, re.MULTILINE)
                ranked_list_raw = ranked_list_raw[:10]
            except IndexError:
                ranked_list_raw = []

            print("\n--- ANALISI VALIDITÀ ---")
            if not ranked_list_raw:
                print("🔴 Il modello non ha generato una lista di raccomandazioni nel formato atteso.")
            else:
                num_valid = 0
                for rec_title in ranked_list_raw:
                    # Normalizza il titolo raccomandato nello stesso modo dei candidati
                    possible_variants = parse_complex_title(rec_title)
                    # Controlla se ALMENO UNA delle varianti normalizzate è nel set dei candidati
                    is_valid = any(variant in normalized_candidate_set for variant in possible_variants)
                    
                    if is_valid:
                        print(f"  ✅ Valido: '{rec_title.strip()}'")
                        num_valid += 1
                    else:
                        print(f"  ❌ Invalido (allucinazione): '{rec_title.strip()}'")

                total_generated_recs += len(ranked_list_raw)
                total_valid_recs += num_valid
                print(f"\n📊 Risultato per questo utente: {num_valid}/{len(ranked_list_raw)} raccomandazioni valide.")

    print("\n" + "#"*20 + " 🏁 FINE PRE-FLIGHT CHECK 🏁 " + "#"*20)
    if total_generated_recs > 0:
        validity_percentage = (total_valid_recs / total_generated_recs) * 100
        print(f"\nRIEPILOGO GENERALE:")
        print(f"  - Raccomandazioni totali generate: {total_generated_recs}")
        print(f"  - Raccomandazioni valide (dai candidati): {total_valid_recs}")
        print(f"  - Percentuale di validità: {validity_percentage:.2f}%")
    else:
        print("\nNessuna raccomandazione è stata generata nel formato corretto per la valutazione.")
    # --- TEST DI DIVERSITÀ DELLE RISPOSTE ---
    print("\n🔁 Test di generazione multipla per valutare la varietà delle risposte...")

    num_samples = 12  # Numero di volte che vuoi generare
    generated_responses = []

    model.eval()  # Modalità valutazione
    prompt = tokenizer.apply_chat_template(test_case["prompt"], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)
    inputs_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        for i in range(num_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,           # ⚠️ Abilita campionamento per output variabili
                temperature=2.0,          # Maggiore temperatura = più diversità
                top_p=0.95,               # Top-p sampling
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            generated_responses.append(response.strip())

    # --- MOSTRA I RISULTATI ---
    for idx, resp in enumerate(generated_responses, 1):
        print(f"\n📥 Risposta #{idx}:\n{resp}")

    print("#"*60)
# --- 4. Definisci la Funzione di Ricompensa e il Trainer GRPO ---
def dcg_score(scores):
    """Calcola il Discounted Cumulative Gain basato sulla tua formula."""
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(len(scores), dtype=np.float64) + 2)),
        dtype=np.float64
    )

def ndcg_at_10(ranked_relevance_scores, num_relevant_items):
    """Calcola NDCG@10."""
    # Il DCG è calcolato sui punteggi di rilevanza della lista raccomandata
    rank_dcg = dcg_score(ranked_relevance_scores)

    if rank_dcg == 0.0:
        return 0.0

    # L'IDCG è il DCG ideale, con tutti gli item rilevanti nelle prime posizioni
    ideal_scores = np.ones(min(num_relevant_items, 10))
    ideal_dcg = dcg_score(ideal_scores)

    if ideal_dcg == 0.0:
        return 0.0

    return rank_dcg / ideal_dcg
def reward_function(prompts, completions, **kwargs):
    """
    Funzione di ricompensa basata su NDCG@10.
    """
    all_target_movies = kwargs["target_movies"]
    rewards = []
    
    for i in range(len(completions)):
        try:
            generated_text = completions[i][0]['content']
        except (IndexError, KeyError, TypeError):
            rewards.append(0.0)
            continue

        # 1. PARSING: Estrai la lista numerata di film dall'output del modello
        # Questa regex cerca righe che iniziano con un numero, un punto e uno spazio.
        try:
            recs_section = generated_text.split("**Recommended movies**:")[1]
            ranked_list_raw = re.findall(r"^\s*\d+\.\s*[\"\'\*\[]?\s*(.*?)\s*[\"\'\*\]]?\s*(?:\(|$)", recs_section, re.MULTILINE)
            # Prendi al massimo i primi 10
            ranked_list_raw = ranked_list_raw[:10]
        except IndexError:
            ranked_list_raw = [] # Nessuna sezione di raccomandazioni trovata

        if not ranked_list_raw:
            rewards.append(0.0)
            continue
            
        # 2. PREPARAZIONE TARGET: Crea il "super-set" di titoli target normalizzati
        target_movie_list_for_user = all_target_movies[i]
        normalized_target_set = set()
        for title_string in target_movie_list_for_user:
            variants = parse_complex_title(title_string)
            for variant in variants:
                if variant: normalized_target_set.add(variant)

        relevance_scores = []
        for rec_line in ranked_list_raw:
            # 1. Isola la parte di testo che contiene il titolo (prima di ':' o '-')
            clean_title = rec_line.replace('**', '').strip()
            possible_variants = parse_complex_title(clean_title)
            # 4. Controlla se una QUALSIASI delle varianti generate matcha con il set dei target
            is_match = any(variant in normalized_target_set for variant in possible_variants)
            relevance_scores.append(1.0 if is_match else 0.0)
        
        # 4. CALCOLO NDCG@10: Usa le funzioni helper
        score = ndcg_at_10(relevance_scores, len(normalized_target_set))
        hits = sum(relevance_scores)
        rewards.append(score)
        
        if score > 0:
            print(f"✅ Hit! NDCG@10: {score:.4f} - Hits: {hits}")
        else:
            print(f"❌ Miss! NDCG@10: {score:.4f}")
    return rewards


# Configurazione per GRPO
# grpo_args = GRPOConfig(
#     output_dir=output_dir,
#     beta=0.1,  # Parametro chiave di GRPO/DPO. Controlla quanto ci si allontana dal modello di riferimento.
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     gradient_checkpointing=True,
#     learning_rate=5e-5,
#     num_train_epochs=1, # Per un test, 1 epoca è sufficiente. Aumenta a 2-3 per risultati migliori.
#     save_strategy="epoch",
#     logging_steps=10,
#     lr_scheduler_type="cosine",
#     warmup_steps=10,
#     optim="adamw_8bit",
#     bf16=not torch.cuda.is_bf16_supported(), # Usa bf16 se supportato
#     fp16=torch.cuda.is_bf16_supported(),
#     remove_unused_columns=False,
#     pad_token_id=tokenizer.pad_token_id,
#     max_length=1024,      # Lunghezza massima input + output
#     max_prompt_length=200, # Lunghezza massima del prompt
# )
grpo_args = GRPOConfig(
    num_train_epochs=3,          # MODIFICATO
    temperature = 2.0,
    top_p = 0.95,
    learning_rate = 5e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 16, # Decrease if out of memory
    max_completion_length = 500,
    save_strategy="epoch",       # AGGIUNTO
    report_to = "none", # Can use Weights & Biases
    output_dir = output_dir,
)

# Istanzia il Trainer
trainer = GRPOTrainer(
    model=model,
    args=grpo_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function,
)

# --- 5. Esegui il Training ---
print("\n🧠 Starting GRPO fine-tuning...")
trainer.train()
print("✨ Training complete!")

# --- 6. Salva il Modello Fine-Tuned ---
print(f"💾 Saving fine-tuned model to '{output_model_path}'...")
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print("✅ Model saved successfully.")

# --- Esempio di Inferenza (opzionale) ---
print("\n🧪 Running a test inference with the fine-tuned model...")
# Fai il merge degli adapter LoRA nel modello base per un'inferenza più veloce
if hasattr(model, "merge_and_unload"):
    model.merge_and_unload()

# Prendi un esempio dal nostro dataset
test_case = train_dataset[0]
prompt = tokenizer.apply_chat_template(test_case["prompt"], tokenize=False, add_generation_prompt=True)

inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True, pad_token_id=tokenizer.pad_token_id)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- TEST INFERENCE ---")
print(response)
print("\n--- Target movies for this user were: ---")
print(test_case["target_movies"])