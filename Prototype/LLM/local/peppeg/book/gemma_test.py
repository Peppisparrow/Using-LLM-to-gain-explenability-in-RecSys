import torch
import json
import os
import re
import argparse
import numpy as np
from unsloth import FastModel
from tqdm import tqdm
import random
import pandas as pd
import string

# --- FUNZIONI HELPER ---
ARTICLES_TO_REMOVE = {
    'the', 'a', 'an', 'le', 'la', 'les', 'l\'', 'il', 'lo', 'i', 'gli', 'un', 'una',
    'un\'', 'el', 'los', 'las', 'ein', 'eine', 'der', 'die', 'das', 'aka'
}

def normalize_text(text: str) -> str:
    """
    Converte il testo in minuscolo, rimuove la punteggiatura, gli articoli comuni
    e normalizza gli spazi.
    """
    text = text.lower()
    text = text.replace("'", "'")
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in ARTICLES_TO_REMOVE]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_complex_title(title_string: str) -> list[str]:
    """
    Analizza un titolo complesso per estrarre le sue varianti.
    """
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
    if rank_dcg == 0.0: 
        return 0.0
    ideal_scores = np.ones(min(num_relevant_items, 10))
    ideal_dcg = dcg_score(ideal_scores)
    if ideal_dcg == 0.0: 
        return 0.0
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

    print("🔄 Caricamento dati di test e target...")
    print(f"📁 Test set: {test_path}")
    print(f"📁 Target set: {target_path}")
    print(f"📁 Candidates: {args.candidate_items_path}")
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_histories = json.load(f)
    with open(target_path, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    
    target_map = {item['user_id']: item['history'] for item in target_data}
    candidate_items = pd.read_csv(args.candidate_items_path)
    candidates_map = {}
    
    for user_id, group in tqdm(candidate_items.groupby('user_id'), desc="Processing candidates"):
        candidates_map[user_id] = group.to_dict('records')

    # --- GESTIONE CHECKPOINT ---
    processed_user_ids = set()
    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"✅ Checkpoint trovato: '{args.checkpoint_path}'. Ripresa valutazione.")
        try:
            with open(args.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            results = checkpoint_data.get("results", [])
            all_ndcg_scores = checkpoint_data.get("all_ndcg_scores", [])
            all_hits = checkpoint_data.get("all_hits", [])
            processed_user_ids = {res['user_id'] for res in results}
            
            print(f"📈 {len(processed_user_ids)} utenti già processati.")
            if all_ndcg_scores:
                print(f"NDCG medio corrente: {np.mean(all_ndcg_scores):.4f}")

        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Errore lettura checkpoint: {e}. Inizio nuova valutazione.")
            all_ndcg_scores = []
            all_hits = []
            results = []
    else:
        print("ℹ️ Nessun checkpoint trovato. Inizio nuova valutazione.")
        all_ndcg_scores = []
        all_hits = []
        results = []

    # --- FASE DI INFERENZA E VALUTAZIONE ---
    users_evaluated = 0
    
    for i, history_item in enumerate(tqdm(test_histories, desc="Evaluating on Test Set")):
        user_id = history_item['user_id']
        
        # Limita il numero di utenti se specificato
        if args.max_users is not None and users_evaluated >= args.max_users:
            print(f"✅ Raggiunto limite di {args.max_users} utenti. Terminazione.")
            break
        
        if user_id not in target_map:
            continue
        
        if user_id in processed_user_ids:
            continue
        
        target_history = target_map[user_id]
        titles = [item['title'] for item in history_item['history']]
        target_titles = [item['title'] for item in target_history]

        if len(titles) < 5:
            continue

        if not target_titles and titles:
            target_titles.append(titles.pop())
        
        # Costruzione delle interazioni
        full_history = history_item['history']
        if args.max_history_items > 0 and len(full_history) > args.max_history_items:
            sampled_history = random.sample(full_history, args.max_history_items)
        else:
            sampled_history = full_history
        
        interactions = []
        for item in sampled_history:
            interactions.append(
                f"Title: {item['title']} \n"
                f"Description: {item.get('description', item.get('desc', 'N/A'))} \n"
                f"Genre: {item['genres']} \n"
                f"Rating: {item.get('rating', item.get('Rating', 'N/A'))}"
            )

        # Preparazione candidati
        user_candidates = candidates_map.get(user_id, [])
        candidate_strings = []
        for candidate in user_candidates:
            title = candidate.get('new_title', 'N/A')
            candidate_str = f"Title: {title}"
            candidate_strings.append(candidate_str)
        
        formatted_candidates = "\n\n".join(candidate_strings)
        formatted_interactions = '\n\n'.join(interactions)
        
        # Determina il tipo di contenuto (books o movies)
        content_type = "books" if args.content_type == "books" else "movies"
        content_verb = "read" if args.content_type == "books" else "watched"
        
        prompt_text = (
            f"The user recently {content_verb} the following {content_type}:\n"
            f"{formatted_interactions}\n\n"
            f"Your task is to act as a filter and ranker. From the following list of candidate {content_type}, "
            f"you must select and rank the ten best {content_type} for the user.\n"
            f"**Candidate {content_type.capitalize()}**:\n\n{formatted_candidates}\n\n"
            f"Use the following format for your response:\n"
            f"**User's summary**:[Brief summary of the user's preferences]\n"
            f"**Recommended {content_type}**:\n1. [First {content_type[:-1]} title]\n...\n10. [Tenth {content_type[:-1]} title]\n\n"
            f"Please provide only the titles of the ten recommended {content_type}.\n"
            f"⚠️ **CRITICAL INSTRUCTION**: You MUST choose exclusively from the 'Candidate {content_type.capitalize()}' list. "
            f"Do NOT recommend any {content_type[:-1]} that is not in that list."
        )
        
        messages = [
            {"role": "system", "content": f"You are a helpful {content_type[:-1]} recommendation assistant."},
            {"role": "user", "content": prompt_text},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[-1]

        # Generazione con gestione errori
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p if args.use_top_p else None,
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        except Exception as e:
            print(f"\n⚠️ SKIPPING User {user_id} - Errore: {e}")
            torch.cuda.empty_cache()
            continue
        
        # Parsing dell'output
        try:
            recs_section = generated_text.split(f"**Recommended {content_type}**:")[1]
            ranked_list_raw = re.findall(
                r"^\s*\d+\.\s*[\"\'\*\[]?\s*(.*?)\s*[\"\'\*\]]?\s*(?:\(|$)", 
                recs_section, 
                re.MULTILINE
            )[:10]
        except IndexError:
            if args.verbose:
                print(f"⚠️ Nessuna raccomandazione trovata per l'utente {user_id}.")
            ranked_list_raw = []

        if not ranked_list_raw:
            all_ndcg_scores.append(0.0)
            all_hits.append(0)
            continue

        # Normalizzazione target
        normalized_target_set = set()
        for title_string in target_titles:
            if title_string is not None:
                variants = parse_complex_title(title_string)
                for variant in variants:
                    if variant: 
                        normalized_target_set.add(variant)

        # Calcolo relevance scores
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
        users_evaluated += 1

        # Salva risultati
        results.append({
            "user_id": user_id,
            "recommendations": ranked_list_raw,
            "ground_truth": target_titles,
            "ndcg_score": score,
            "hits": hit
        })
        
        # Log periodico
        if args.verbose and (users_evaluated % args.log_frequency == 0):
            print(f"Avg NDCG dopo {users_evaluated} utenti: {np.mean(all_ndcg_scores):.4f}")

        # Salvataggio checkpoint
        if users_evaluated > 0 and users_evaluated % args.checkpoint_frequency == 0:
            print(f"\n--- 💾 SALVATAGGIO CHECKPOINT ({users_evaluated} utenti processati) ---")
            
            checkpoint_data = {
                "results": results,
                "all_ndcg_scores": all_ndcg_scores,
                "all_hits": all_hits
            }
            
            try:
                with open(args.checkpoint_path, "w", encoding="utf-8") as f_checkpoint:
                    json.dump(checkpoint_data, f_checkpoint, indent=4, ensure_ascii=False)
                print(f"✅ Checkpoint salvato: {args.checkpoint_path}\n")
            except Exception as e:
                print(f"❌ Errore salvataggio checkpoint: {e}\n")

    # --- RISULTATI FINALI ---
    average_ndcg = np.mean(all_ndcg_scores) if all_ndcg_scores else 0.0
    average_hits = np.mean(all_hits) if all_hits else 0.0
    
    print("\n" + "="*50)
    print("📈 RISULTATI DELLA VALUTAZIONE")
    print("="*50)
    print(f"Utenti valutati: {len(all_ndcg_scores)}")
    print(f"NDCG@10 Medio: {average_ndcg:.4f}")
    print(f"Hit Rate Medio: {average_hits:.4f}")
    print("="*50)

    # Esempi qualitativi
    if args.verbose and results:
        print("\n--- 🔍 Esempi di Raccomandazioni ---")
        for res in results[:min(5, len(results))]:
            print(f"\nUser ID: {res['user_id']}")
            print(f"  NDCG: {res['ndcg_score']:.4f}, Hits: {res['hits']}")
            print(f"  Raccomandazioni: {res['recommendations'][:3]}...")

    # Salvataggio risultati finali
    final_output = {
        "summary": {
            "total_users": len(all_ndcg_scores),
            "average_ndcg": average_ndcg,
            "average_hits": average_hits
        },
        "results": results,
        "all_ndcg_scores": all_ndcg_scores,
        "all_hits": all_hits
    }
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Risultati salvati in: {args.output_path}")

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