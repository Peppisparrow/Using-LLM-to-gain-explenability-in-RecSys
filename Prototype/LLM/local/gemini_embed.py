import google.generativeai as genai
API_KEY = 'AIzaSyA41qr4tgrBnRQ2xd6r0Hn9cT6UForL0vY'
genai.configure(api_key=API_KEY)
from tqdm import tqdm
import time
import pickle
import json
import numpy as np
import os
from LLM.utils.convert_npz_boosted import convert_to_npz
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
# --- 1. Define File Paths ---

data_dir = "Dataset/ml/ml-latest-small/final"
prompts_path = os.path.join(data_dir, 'item_prompts.json')

# Definiamo i percorsi per l'output finale e per il checkpoint
output_path = f"{data_dir}/item_embeddings_gemini.pkl"
checkpoint_path = f"{data_dir}/gemini_checkpoint_RETRIEVAL_DOCUMENT.pkl" # <-- NUOVO: File per il checkpoint
output_npz_file = f"{data_dir}/item_embeddings_gemini_RETRIEVAL_DOCUMENT"

# --- 2. Caricamento dei Prompt (Invariato) ---
print(f"🔄 Loading item prompts from '{prompts_path}'...")
try:
    with open(prompts_path, "r", encoding='utf-8') as f:
        item_prompts = json.load(f)
    print(f"✅ Loaded {len(item_prompts)} item prompts.")
except FileNotFoundError:
    print(f"🚨 ERROR: File not found at '{prompts_path}'. Please check the path.")
    exit()
except json.JSONDecodeError:
    print(f"🚨 ERROR: The file '{prompts_path}' is not a valid JSON file.")
    exit()

# --- 3. Logica di Checkpoint e Ripresa ---

final_results = {}
if os.path.exists(checkpoint_path):
    print(f"✅ Checkpoint found at '{checkpoint_path}'. Resuming...")
    with open(checkpoint_path, "rb") as f:
        final_results = pickle.load(f)
    print(f"📊 Loaded {len(final_results)} items from checkpoint.")
else:
    print("🚦 No checkpoint found. Starting from scratch.")

# Filtra i prompt per elaborare solo quelli mancanti
processed_ids = set(final_results.keys())
all_ids = list(item_prompts.keys())
all_prompts = list(item_prompts.values())

ids_to_process = [item_id for item_id in all_ids if item_id not in processed_ids]
prompts_to_process = [item_prompts[item_id] for item_id in ids_to_process]

# --- 4. Generazione degli Embeddings con Salvataggio Continuo ---

if not prompts_to_process:
    print("✨ All items already processed. Nothing to do.")
else:
    print(f"🚀 Calling Gemini API for the remaining {len(prompts_to_process)} prompts...")
    batch_size = 1 # Un batch size ridotto per maggiore sicurezza con i limiti API

    # Itera sui dati ancora da processare
    for i in tqdm(range(0, len(ids_to_process), batch_size), desc="🧠 Processing batches"):
        batch_ids = ids_to_process[i:i + batch_size]
        batch_prompts = prompts_to_process[i:i + batch_size]

        try:
            # Chiama l'API di Gemini per il batch corrente
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=batch_prompts,
                task_type="RETRIEVAL_DOCUMENT"
            )

            # Aggiungi i risultati del batch al dizionario principale
            for j, item_id in enumerate(batch_ids):
                final_results[item_id] = {
                    "prompt": batch_prompts[j],
                    "embedding": result['embedding'][j]
                }
            
            # Salva il checkpoint dopo ogni batch di successo
            with open(checkpoint_path, "wb") as f:
                pickle.dump(final_results, f, protocol=pickle.HIGHEST_PROTOCOL)

            time.sleep(3) # Pausa per rispettare i limiti API

        except Exception as e:
            print(f"\n🚨 An error occurred: {e}")
            print("🛑 Stopping script. Run it again to resume from the last saved checkpoint.")
            exit()


# --- 5. Salvataggio Finale e Conversione (Logica spostata e adattata) ---

print(f"\n✅ All {len(final_results)} embeddings calculated and saved in checkpoint.")

# Salva il risultato finale completo nel file di output designato
print(f"💾 Saving the final combined data to '{output_path}'...")
with open(output_path, "wb") as f:
    pickle.dump(final_results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✨ Processing complete!")

# Esegue la funzione di conversione sul risultato finale
convert_to_npz(final_results, output_npz_file, id_key='item_id')