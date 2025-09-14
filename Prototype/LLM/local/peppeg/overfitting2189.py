import torch
import json
import random
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import string

# --- CONFIG ---
model_id = "google/gemma-3-4b-it"
prompt_path = "Dataset/ml/ml-latest-small/tuning/histories_gemma_recommender_train.json"
candidate_items_path = "Dataset/ml/ml-latest-small/tuning/candidate_items_30_train_hits.csv"
target_movies_path = "Dataset/ml/ml-latest-small/tuning/histories_gemma_recommender_target.json"
output_dir = "Dataset/ml/ml-latest-small/tuning/gemma/grpo_overfit_test"
output_model_path = f"{output_dir}/final_model"
MAX_USERS = 50  # Overfitting su un piccolo sottoinsieme

# --- FUNZIONI UTILI ---
ARTICLES_TO_REMOVE = {
    'the', 'a', 'an', 'le', 'la', 'les', 'l\'', 'il', 'lo', 'i', 'gli', 'un', 'una',
    'un\'', 'el', 'los', 'las', 'ein', 'eine', 'der', 'die', 'das', 'aka'
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in ARTICLES_TO_REMOVE]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_complex_title(title_string: str) -> list[str]:
    title_string = re.sub(r'\s*\(\d{4}\)$', '', title_string).strip()
    potential_titles = re.split(r'\(|\)', title_string)
    potential_titles = [t.strip() for t in potential_titles if t.strip()]

    cleaned_titles = set()
    for title in potential_titles:
        match = re.match(r'^(.*?), (the|a|an|le|la|il|lo|el|der|die|das)$', title, re.IGNORECASE)
        if match:
            reordered_title = f"{match.group(2)} {match.group(1)}"
            cleaned_titles.add(normalize_text(reordered_title))
        else:
            cleaned_titles.add(normalize_text(title))
    return list(cleaned_titles)

# --- 1. CARICA DATI ---
print("🔄 Caricamento dati di test e target...")
with open(prompt_path, 'r', encoding='utf-8') as f:
    histories = json.load(f)
with open(target_movies_path, 'r', encoding='utf-8') as f:
    target_data = json.load(f)
target_map = {item['user_id']: item['history'] for item in target_data}
candidate_items = pd.read_csv(candidate_items_path)
candidates_map = {}
for user_id, group in tqdm(candidate_items.groupby('user_id'), desc="Processing candidates"):
    candidates_map[user_id] = group.to_dict('records')

training_data = []

for i, history in tqdm(enumerate(histories), desc="Preparing data"):
    if i >= MAX_USERS:
        break
    MAX_HISTORY_ITEMS=10
    if history['user_id'] not in target_map:
        continue # Salta se non ci sono target
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
    if not target_titles and prompt_titles:
        target_titles.append(prompt_titles.pop())
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
    user_id = history['user_id']
    user_candidates = candidates_map.get(user_id, [])
    candidate_strings = []
    for candidate in user_candidates:
        title = candidate.get('title', 'N/A')
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
    print("A"*50)
    print(prompt_text)
    messages = [
        {"role": "system", "content": "You are a helpful movie recommendation assistant."},
        {"role": "user", "content": prompt_text},
    ]

    ref = random.choice(target_titles)
    training_data.append({
        "user_id": history['user_id'],
        "prompt": messages,
        "reference_completion": f"One good pick is {ref}.",
        "target_movies": target_titles
    })

train_dataset = Dataset.from_list(training_data)
print(f"✅ Dataset creato con {len(train_dataset)} esempi.")

# --- 3. CARICA MODELLO ---
model, tokenizer = FastModel.from_pretrained(
    model_name=model_id,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

model = FastModel.get_peft_model(model, r=8,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

tokenizer = get_chat_template(tokenizer, chat_template="gemma",
    mapping={"role":"role","content":"content","user":"user","assistant":"model"})

# --- 4. FUNZIONE RICOMPENSA ---
def dcg_score(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(len(scores), dtype=np.float64) + 2)))
def ndcg_at_10(ranked_relevance_scores, num_relevant_items):
    rank_dcg = dcg_score(ranked_relevance_scores)
    if rank_dcg == 0.0: return 0.0
    ideal_scores = np.ones(min(num_relevant_items, 10))
    ideal_dcg = dcg_score(ideal_scores)
    if ideal_dcg == 0.0: return 0.0
    return rank_dcg / ideal_dcg

def reward_function(prompts, completions, **kwargs):
    all_target_movies = kwargs["target_movies"]
    rewards = []
    for i in range(len(completions)):
        try:
            generated_text = completions[i][0]['content']
        except (IndexError, KeyError, TypeError):
            rewards.append(0.0)
            continue
        try:
            recs_section = generated_text.split("**Recommended movies**:")[1]
            ranked_list_raw = re.findall(r"^\s*\d+\.\s*[\"\'\*\[]?\s*(.*?)\s*[\"\'\*\]]?\s*(?:\(|$)", recs_section, re.MULTILINE)
            ranked_list_raw = ranked_list_raw[:10]
        except IndexError:
            ranked_list_raw = []
        if not ranked_list_raw:
            rewards.append(0.0)
            continue
        target_movie_list_for_user = all_target_movies[i]
        normalized_target_set = set()
        for title_string in target_movie_list_for_user:
            variants = parse_complex_title(title_string)
            for variant in variants:
                if variant: normalized_target_set.add(variant)
        relevance_scores = []
        for rec_line in ranked_list_raw:
            clean_title = rec_line.replace('**', '').strip()
            possible_variants = parse_complex_title(clean_title)
            is_match = any(variant in normalized_target_set for variant in possible_variants)
            relevance_scores.append(1.0 if is_match else 0.0)
        score = ndcg_at_10(relevance_scores, len(normalized_target_set))
        hits = sum(relevance_scores)
        rewards.append(score)
        if score > 0:
            print(f"✅ Hit! NDCG@10: {score:.4f} - Hit count: {hits}")
        # else:
        #     print(f"❌ Miss! NDCG@10: {score:.4f} - Generated: {ranked_list_raw}, Target: {normalized_target_set}")
    return rewards

grpo_args = GRPOConfig(
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    logging_steps=1,
    num_generations=5,
    max_completion_length=300,
    output_dir=output_dir,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function,
)

# --- 6. FUNZIONE DI VALUTAZIONE ---
def evaluate_model(model, tokenizer, dataset):
    model.eval()
    ndcg_scores = []
    hits = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            test_case = dataset[i]
            prompt = tokenizer.apply_chat_template(test_case["prompt"], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs,max_new_tokens=300,use_cache=True,do_sample=True,temperature=0.1,pad_token_id=tokenizer.pad_token_id)
            input_len = inputs["input_ids"].shape[-1]
            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            target_titles = test_case["target_movies"]
            try:
                recs_section = response.split("**Recommended movies**:")[1]
                ranked_list_raw = re.findall(r"^\s*\d+\.\s*[\"\'\*\[]?\s*(.*?)\s*[\"\'\*\]]?\s*(?:\(|$)", recs_section, re.MULTILINE)[:10]
            except IndexError:
                print(f"⚠️ Nessuna raccomandazione trovata per l'utente {response}.")
                ranked_list_raw = []
            if not ranked_list_raw:
                ndcg_scores.append(0.0)
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
            score = ndcg_at_10(relevance_scores, len(normalized_target_set))
            hit = sum(relevance_scores)
            ndcg_scores.append(score)
            hits.append(hit)
            # <-- NUOVO BLOCCO DEBUG: si attiva solo se DEBUG_MODE è True
            if (i + 1) % 10 == 0:
                current_progress = len(ndcg_scores)
                print(f"Avg NDCG after processing {current_progress} new users: {np.mean(ndcg_scores):.4f}")
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    hit_rate = np.mean(hits) if hits else 0.0
    print(f"NDCG medio: {avg_ndcg:.4f}")
    print(f"Hit ratio: {hit_rate:.4f}")

# --- 7. VALUTA PRIMA DEL TRAINING ---
print("\n🔍 Performance del modello BASE:")
evaluate_model(model, tokenizer, train_dataset)

# --- 8. TRAINING ---
print("\n🚀 Inizio training (overfitting test)...")
trainer.train()
print("✨ Training terminato!")

# --- 9. VALUTA DOPO IL TRAINING ---
print("\n🔍 Performance del modello FINE-TUNED:")
evaluate_model(model, tokenizer, train_dataset)

# --- 10. SALVA ---
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print(f"✅ Modello salvato in {output_model_path}")
