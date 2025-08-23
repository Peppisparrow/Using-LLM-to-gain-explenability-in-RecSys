import torch
import json
import os
import pickle
import numpy as np
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm
from LLM.utils.convert_npz_boosted import convert_to_npz

# --- 1. Definisci Percorsi e Costanti ---

# Modelli
BASE_MODEL_ID = "google/gemma-3-4b-it"
FINETUNED_MODEL_PATH = "Dataset/ml/ml-latest-small/tuning/gemma/model-no-4bit/gemma3-grpo"

# Input (raccomandazioni generate dallo script precedente)
BASE_RECOMMENDATIONS_PATH = "Dataset/ml/ml-latest-small/tuning/gemma/model-no-4bit/item_prompts.json"
FINETUNED_RECOMMENDATIONS_PATH = "Dataset/ml/ml-latest-small/tuning/gemma/model-no-4bit/gemma3-grpo/item_prompts.json"

# Output (directory dove salvare gli embedding)
BASE_OUTPUT_DIR = "Dataset/ml/ml-latest-small/tuning/gemma/model-no-4bit/"
FINETUNED_OUTPUT_DIR = "Dataset/ml/ml-latest-small/tuning/gemma/model-no-4bit/gemma3-grpo/"

CHECKPOINT_FREQ = 50  # Salva un checkpoint ogni 50 utenti


# --- 3. Funzione Principale per Generare gli Embedding ---

def generate_embeddings(model_identifier: str, input_json_path: str, output_dir: str, model_name_for_files: str):
    """
    Carica un modello, elabora le raccomandazioni testuali e salva gli embedding.
    """
    print("="*80)
    print(f"🚀 Inizio generazione embedding per il modello: {model_name_for_files}")
    print(f"📂 Input: {input_json_path}")
    print(f"🗂️ Output in: {output_dir}")
    print("="*80)

    # --- Setup dei percorsi ---
    os.makedirs(output_dir, exist_ok=True)
    output_npz_last = os.path.join(output_dir, f"ItemPlots_GemmaLast.npz")
    output_npz_mean = os.path.join(output_dir, f"ItemPlots_GemmaMean.npz")
    checkpoint_path_last = os.path.join(output_dir, f"checkpoint_itemplotslast.pkl")
    checkpoint_path_mean = os.path.join(output_dir, f"checkpoint_itemplotsmean.pkl")

    # --- Carica Modello e Tokenizer con Unsloth ---
    print(f"🧠 Caricamento del modello '{model_identifier}'...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_identifier,
        dtype=None,
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "model"},
    )
    
    # --- Carica i Dati di Input (Raccomandazioni) ---
    print(f"🔄 Caricamento delle raccomandazioni da '{input_json_path}'...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            item_prompt_data = json.load(f)
        print(f"✅ Caricati dati per {len(item_prompt_data)} utenti.")
    except FileNotFoundError:
        print(f"🚨 ERRORE: File di input non trovato. Impossibile procedere.")
        return
    
    # --- Logica di Checkpoint ---
    final_last_results, final_mean_results = {}, {}
    if os.path.exists(checkpoint_path_last) and os.path.exists(checkpoint_path_mean):
        print("🔄 Trovato checkpoint. Caricamento progressi...")
        with open(checkpoint_path_last, 'rb') as f: final_last_results = pickle.load(f)
        with open(checkpoint_path_mean, 'rb') as f: final_mean_results = pickle.load(f)
        print(f"✅ Ripresa con {len(final_last_results)} utenti già processati.")
    else:
        print("🏁 Nessun checkpoint trovato. Si parte da zero.")

    # Determina quali utenti processare
    processed_ids = set(final_last_results.keys())
    items_to_process = [(uid, text) for uid, text in item_prompt_data.items() if uid not in processed_ids]

    if not items_to_process:
        print("✅ Tutti gli utenti sono già stati processati.")
    else:
        print(f"🤖 Generazione embedding per i {len(items_to_process)} utenti rimanenti...")
        for i, (item_id, recommendation_text) in enumerate(tqdm(items_to_process, desc=f"Processing ({model_name_for_files})")):
            # Per l'embedding, un prompt semplice è sufficiente
            messages = [
                {"role": "system", "content": "You are a helpful movie recommendation assistant."},
                {"role": "user", "content": recommendation_text},
            ]
            
            # Applica il template della chat e tokenizza
            prompt_string = tokenizer.apply_chat_template(messages, tokenize=False)

            # 2. Tokenizza la stringa formattata per ottenere i tensori.
            #    Il tokenizer si aspetta una lista di stringhe, quindi usiamo [prompt_string].
            inputs = tokenizer(
                [prompt_string],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to("cuda")
            # --- FINE MODIFICA ---

            with torch.no_grad():
                # Esegui il modello per ottenere gli hidden states
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1].cpu() # Prendi l'ultimo strato e sposta su CPU

            # Calcola i due tipi di embedding
            last_embedding = last_hidden_states[:, -1, :] # Embedding dell'ultimo token
            mean_embedding = last_hidden_states.mean(dim=1) # Media degli embedding di tutti i token

            # Salva nei dizionari
            final_last_results[item_id] = {
                "prompt": recommendation_text,
                "embedding": last_embedding.squeeze().tolist()
            }
            final_mean_results[item_id] = {
                "prompt": recommendation_text,
                "embedding": mean_embedding.squeeze().tolist()
            }

            # Salva checkpoint
            if (i + 1) % CHECKPOINT_FREQ == 0:
                print(f"\n💾 Checkpoint: Salvataggio di {len(final_last_results)} risultati...")
                with open(checkpoint_path_last, 'wb') as f: pickle.dump(final_last_results, f)
                with open(checkpoint_path_mean, 'wb') as f: pickle.dump(final_mean_results, f)

    # --- Conversione Finale e Pulizia ---
    print("\n✅ Processo completato. Conversione dei risultati in formato NPZ...")
    convert_to_npz(final_last_results, output_npz_last, id_key='item_id')
    convert_to_npz(final_mean_results, output_npz_mean, id_key='item_id')

    print("🧹 Pulizia dei file di checkpoint...")
    if os.path.exists(checkpoint_path_last): os.remove(checkpoint_path_last)
    if os.path.exists(checkpoint_path_mean): os.remove(checkpoint_path_mean)
    print("✨ Fatto!")


# --- 4. ESECUZIONE PRINCIPALE ---
if __name__ == "__main__":
    # --- Esecuzione per il MODELLO BASE ---
    generate_embeddings(
        model_identifier=BASE_MODEL_ID,
        input_json_path=BASE_RECOMMENDATIONS_PATH,
        output_dir=BASE_OUTPUT_DIR,
        model_name_for_files="Base"
    )

    # --- Esecuzione per il MODELLO FINE-TUNED ---
    generate_embeddings(
        model_identifier=FINETUNED_MODEL_PATH,
        input_json_path=FINETUNED_RECOMMENDATIONS_PATH,
        output_dir=FINETUNED_OUTPUT_DIR,
        model_name_for_files="Finetuned"
    )

    print("\n\n🎉 Tutti i processi di generazione embedding sono terminati.")