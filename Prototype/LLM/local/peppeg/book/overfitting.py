import torch
import json
import random
import argparse
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

# --- FUNZIONI UTILI ---
ARTICLES_TO_REMOVE = {
    'the', 'a', 'an', 'le', 'la', 'les', 'l\'', 'il', 'lo', 'i', 'gli', 'un', 'una',
    'un\'', 'el', 'los', 'las', 'ein', 'eine', 'der', 'die', 'das', 'aka'
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("'", "'")
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in ARTICLES_TO_REMOVE]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_complex_title(title_string: str) -> list[str]:
    if ':' in title_string:
        parts = title_string.split(':')
        all_parts = [title_string] + [p.strip() for p in parts if p.strip()]
    else:
        all_parts = [title_string]
    cleaned_titles = set()
    for part in all_parts:
        potential_titles = re.split(r'\(|\)', part)
        potential_titles = [t.strip() for t in potential_titles if t.strip()]
        for title in potential_titles:
            title = re.sub(r',?\s*#\d+', '', title).strip()
            match = re.match(r'^(.*?), (the|a|an|le|la|il|lo|el|der|die|das)$', 
                           title, re.IGNORECASE)
            if match:
                reordered_title = f"{match.group(2)} {match.group(1)}"
                cleaned_titles.add(normalize_text(reordered_title))
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
            recs_section = generated_text.split("**Recommended books**:")[1]
            ranked_list_raw = re.findall(r"^\s*\d+\.\s*[\"\'\*\[]?\s*(.*?)\s*[\"\'\*\]]?\s*(?:\(|$)", recs_section, re.MULTILINE)
            ranked_list_raw = ranked_list_raw[:10]
        except IndexError:
            print(f"⚠️ Nessuna raccomandazione trovata per l'utente {generated_text}.")
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
    return rewards

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
            if (i + 1) % 10 == 0:
                current_progress = len(ndcg_scores)
                print(f"Avg NDCG after processing {current_progress} new users: {np.mean(ndcg_scores):.4f}")
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    hit_rate = np.mean(hits) if hits else 0.0
    print(f"NDCG medio: {avg_ndcg:.4f}")
    print(f"Hit ratio: {hit_rate:.4f}")

def main(args):
    # --- CONFIG ---
    model_id = "google/gemma-3-4b-it"
    prompt_path = f"{args.datapath}/histories_gemma_recommender_train.json"
    target_movies_path = f"{args.datapath}/histories_gemma_recommender_test.json"
    output_model_path = f"{args.output_dir}/final_model"
    
    print(f"📁 Datapath: {args.datapath}")
    print(f"📁 Candidate items: {args.candidate_items_path}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Max history items: {args.max_history_items}")
    print(f"👥 Max users: {args.max_users}")
    
    # --- 1. CARICA DATI ---
    print("\n🔄 Caricamento dati di test e target...")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        histories = json.load(f)
    with open(target_movies_path, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    target_map = {item['user_id']: item['history'] for item in target_data}
    candidate_items = pd.read_csv(args.candidate_items_path)
    candidates_map = {}
    for user_id, group in tqdm(candidate_items.groupby('user_id'), desc="Processing candidates"):
        candidates_map[user_id] = group.to_dict('records')

    training_data = []

    for i, history in tqdm(enumerate(histories), desc="Preparing data"):
        if i >= args.max_users:
            break
        if history['user_id'] not in target_map:
            continue
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
        if len(full_history) > args.max_history_items:
            sampled_history = random.sample(full_history, args.max_history_items)
        else:
            sampled_history = full_history
        for item in sampled_history:
            interactions.append(
                f"Title: {item['title']} \n"
                f"Description: {item['desc']} \n"
                f"Genre: {item['genres']} \n"
                f"Rating: {item['Rating']}"
                )
        user_id = history['user_id']
        user_candidates = candidates_map.get(user_id, [])
        candidate_strings = []
        for candidate in user_candidates:
            title = candidate.get('new_title', 'N/A')
            genres = candidate.get('genres', 'N/A')
            description = candidate.get('description', 'No description available.')
            candidate_str = (
                f"Title: {title}"
            )
            candidate_strings.append(candidate_str)
        formatted_candidates = "\n\n".join(candidate_strings)
        formatted_interactions = '\n\n'.join(interactions)
        prompt_text = (
            f"The user recently read the following books:\n"
            f"{formatted_interactions}\n\n"
            "Your task is to act as a filter and ranker. From the following list of candidate books, you must select and rank the ten best books for the user.\n"
            f"**Candidate Books**:\n\n{formatted_candidates}\n\n"
            f"Use the following format for your response:\n"
            f"**User's summary**:[Brief summary of the user's preferences]\n"
            f"**Recommended books**:\n1. [First book title]\n...\n10. [Tenth book title]\n\n"
            "Please provide only the titles of the ten recommended books.\n"
            "⚠️ **CRITICAL INSTRUCTION**: You MUST choose exclusively from the 'Candidate Books' list. Do NOT recommend any book that is not in that list."
        )
        if args.verbose:
            print("A"*50)
            print(prompt_text)
        messages = [
            {"role": "system", "content": "You are a helpful book recommendation assistant."},
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
    print("\n🔄 Caricamento modello...")
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

    # --- 5. CONFIG TRAINING ---
    grpo_args = GRPOConfig(
        temperature=1.0,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        num_generations=5,
        max_completion_length=300,
        output_dir=args.output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
    )

    # --- 7. VALUTA PRIMA DEL TRAINING ---
    if args.eval_before_training:
        print("\n🔍 Performance del modello BASE:")
        evaluate_model(model, tokenizer, train_dataset)

    # --- 8. TRAINING ---
    print("\n🚀 Inizio training...")
    trainer.train()
    print("✨ Training terminato!")

    # --- 10. SALVA ---
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    print(f"✅ Modello salvato in {output_model_path}")

    # --- 9. VALUTA DOPO IL TRAINING ---
    # print("\n🔍 Performance del modello FINE-TUNED:")
    # evaluate_model(model, tokenizer, train_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train book recommendation model with GRPO")
    
    # Parametri richiesti
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path alla directory dei dati (es. Dataset/goodbooks-10k-extended-master/tuning)"
    )
    parser.add_argument(
        "--candidate_items_path",
        type=str,
        required=True,
        help="Path al file CSV dei candidati (es. Dataset/.../candidate_items_30_hits.csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory di output per il modello (es. Dataset/.../model/book)"
    )
    
    # Parametri opzionali con valori di default
    parser.add_argument(
        "--max_history_items",
        type=int,
        default=10,
        help="Numero massimo di item nella history dell'utente (default: 10)"
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=7000,
        help="Numero massimo di utenti da utilizzare per il training (default: 7000)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Numero di epoche di training (default: 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per il training (default: 1)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--eval_before_training",
        action="store_true",
        help="Valuta il modello prima del training"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stampa informazioni di debug dettagliate"
    )
    
    args = parser.parse_args()
    main(args)