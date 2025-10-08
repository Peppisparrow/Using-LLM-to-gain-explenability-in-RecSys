
import torch
import json
import os
import re
import numpy as np
from unsloth import FastModel, FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
import pandas as pd
from peft import PeftModel
import argparse
import re
import string

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

    cleaned_titles = set() # Usiamo un set per evitare duplicati
    for title in potential_titles:
        match = re.match(r'^(.*?), (the|a|an|le|la|il|lo|el|der|die|das)$', title, re.IGNORECASE)
        if match:
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


def main(args):
    # --- CARICAMENTO MODELLO E DATI ---
    print("🚀 Caricamento modello fine-tuned...")
    print(f"📁 Model path: {args.model_path}")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    test_path = f"{args.datapath}/histories_gemma_recommender_train.json"
    target_path = f"{args.datapath}/histories_gemma_recommender_test.json"
    MAX_HISTORY_ITEMS = args.max_history_items

    print("🔄 Caricamento dati di test e target...")
    print(f"📁 Test set: {test_path}")
    print(f"📁 Target set: {target_path}")
    print(f"📁 Candidates: {args.candidate_items_path}")
    
    
    print("🚀 Caricamento modello fine-tuned...")
    print(f"Tentativo di caricamento con transformers standard...{args.model_path}")

    model, tokenizer = FastModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = args.max_seq_length,
        load_in_4bit = False,
        load_in_8bit = False,
    )

    print("🔄 Caricamento dati di test e target...")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_histories = json.load(f)
    with open(target_path, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    target_map = {item['user_id']: item['history'] for item in target_data}
    candidate_items = pd.read_csv(args.candidate_items_path)
    candidates_map = {}
    # Raggruppa il DataFrame per user_id per elaborare i candidati di ogni utente
    for user_id, group in tqdm(candidate_items.groupby('user_id'), desc="Processing candidates"):
        # Converte il gruppo di righe in una lista di dizionari per un accesso più facile
        candidates_map[user_id] = group.to_dict('records')

    # --- 4. FASE DI INFERENZA E VALUTAZIONE ---
    processed_user_ids = set()
    if os.path.exists( args.checkpoint_path):
        print(f"✅ Checkpoint file found at '{ args.checkpoint_path}'. Attempting to resume.")
        try:
            with open( args.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            results = checkpoint_data.get("results", [])
            all_ndcg_scores = checkpoint_data.get("all_ndcg_scores", [])
            all_hits = checkpoint_data.get("all_hits", [])
            processed_user_ids = {res['user_id'] for res in results}
            
            print(f"📈 Resuming evaluation. {len(processed_user_ids)} users already processed.")
            print(f"Current average NDCG: {np.mean(all_ndcg_scores):.4f}" if all_ndcg_scores else "No previous scores found.")

        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Could not read checkpoint file due to an error: {e}. Starting a new evaluation.")
            all_ndcg_scores = []
            all_hits = []
            results = []
    else:
        print("ℹ️ No checkpoint file found. Starting a new evaluation from scratch.")
        all_ndcg_scores = []
        all_hits = []
        results = []


    for i, history_item in enumerate(tqdm(test_histories, desc="Evaluating on Test Set")):
        user_id = history_item['user_id']
        if history_item['user_id'] not in target_map:
            continue
        if user_id in processed_user_ids:
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

        # --- 🛡️ BLOCCO TRY/EXCEPT PER GESTIRE GLI OOM ---
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    use_cache=True,
                    do_sample=True,
                    temperature=args.temperature,
                    # top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        except Exception as e:
            # Se si verifica un errore di memoria
            print(f"\n⚠️  SKIPPING User {user_id} due to Out of Memory error. Il prompt era troppo lungo.")
            
            # Pulisci la cache della memoria della GPU per evitare problemi con i prossimi utenti
            torch.cuda.empty_cache()
            
            # Salta al prossimo utente nel ciclo
            continue
        
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
        score = ndcg_at_10(relevance_scores, len(normalized_target_set))
        hit = sum(relevance_scores)
        all_ndcg_scores.append(score)
        all_hits.append(hit)

        results.append({
            "user_id": user_id,
            "recommendations": ranked_list_raw,
            "ground_truth": target_titles,
            "ndcg_score": score,
            "hits": hit
        })
        # <-- NUOVO BLOCCO DEBUG: si attiva solo se DEBUG_MODE è True
        if (i + 1) % 10 == 0:
            # Calcola l'indice corretto per il messaggio di log
            current_progress = len(all_ndcg_scores)
            total_to_process = len(test_histories) - (len(processed_user_ids) - current_progress)
            print(f"Avg NDCG after processing {current_progress} new users: {np.mean(all_ndcg_scores):.4f}")

        # Salva un checkpoint ogni CHECKPOINT_FREQUENCY utenti
        if len(results) > 0 and len(results) % args.checkpoint_frequency == 0:
            print(f"\n--- 💾 SAVING CHECKPOINT ({len(results)} total users processed) ---")
            
            # Prepara un dizionario con tutti i dati da salvare
            checkpoint_data = {
                "results": results,
                "all_ndcg_scores": all_ndcg_scores,
                "all_hits": all_hits
            }
            
            try:
                with open(args.checkpoint_path, "w", encoding="utf-8") as f_checkpoint:
                    json.dump(checkpoint_data, f_checkpoint, indent=4, ensure_ascii=False)
                print(f"✅ Checkpoint saved successfully to {args.checkpoint_path}\n")
            except Exception as e:
                print(f"❌ Error saving checkpoint: {e}\n")
        

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

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuta un modello di raccomandazione fine-tuned")
    
    # Parametri obbligatori
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path al modello fine-tuned da valutare"
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path al file JSON con le history di test e target"
    )
    parser.add_argument(
        "--candidate_items_path",
        type=str,
        required=True,
        help="Path al file CSV con i candidati"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path per salvare i risultati della valutazione (JSON)"
    )
    
    # Parametri opzionali
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path per salvare/caricare checkpoint intermedi (default: stesso di output_path)"
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="Numero massimo di utenti da valutare (default: tutti)"
    )
    parser.add_argument(
        "--max_history_items",
        type=int,
        default=0,
        help="Numero massimo di item nella history (0 = tutti, default: 0)"
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=10,
        help="Salva checkpoint ogni N utenti (default: 10)"
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=10,
        help="Stampa statistiche ogni N utenti (default: 10)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature per la generazione (default: 0.1)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p per la generazione (default: 0.95)"
    )
    parser.add_argument(
        "--use_top_p",
        action="store_true",
        help="Usa top-p sampling durante la generazione"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="Numero massimo di nuovi token da generare (default: 300)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Lunghezza massima della sequenza (default: 1024)"
    )
    parser.add_argument(
        "--content_type",
        type=str,
        choices=["books", "movies"],
        default="books",
        help="Tipo di contenuto: 'books' o 'movies' (default: books)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Riprendi da checkpoint se esiste"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stampa informazioni dettagliate durante la valutazione"
    )
    
    args = parser.parse_args()
    
    # Se checkpoint_path non è specificato, usa output_path
    if args.checkpoint_path is None:
        args.checkpoint_path = args.output_path
    
    main(args)