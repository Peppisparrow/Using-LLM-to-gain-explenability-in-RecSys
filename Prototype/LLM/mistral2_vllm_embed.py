import torch
from vllm import LLM, PoolingParams
from tqdm import tqdm
import pickle
import numpy as np

# --- 1. Caricamento Dati ---
# Carica i risultati dal primo script per sapere per quali prompt calcolare gli embedding
data_dir = "Dataset/steam/filtering_no_desc_giappo_corean_k10"
results_path = f"{data_dir}/user_results_final.pkl" # Assicurati che il percorso sia corretto
print(f"🔄 Caricando i risultati da '{results_path}'...")
try:
    with open(results_path, "rb") as f:
        text_results = pickle.load(f)
    print(f"✅ Caricati {len(text_results)} record.")
except FileNotFoundError:
    print(f"🚨 Errore: File non trovato. Esegui prima lo script di generazione del testo.")
    exit()

prompts_for_embedding = [
    f"[INST] {data['input_prompt']} [/INST]" 
    for data in text_results.values()
]
user_ids_for_embedding = list(text_results.keys())

# --- 2. Inizializzazione Motore vLLM in MODALITÀ EMBEDDING ---
model_path = "models/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c"
num_gpus = torch.cuda.device_count()

print(f"🚀 Creazione motore vLLM in modalità 'embed' con {num_gpus} GPU...")
llm = LLM(
    model=model_path,
    dtype='bfloat16',
    trust_remote_code=True,
    tensor_parallel_size=num_gpus,
    task="embed",  # <-- LA MODIFICA FONDAMENTALE
    override_pooler_config={"type":"MEAN"}
)
print("✅ Motore vLLM per embedding creato con successo.")


# --- 3. Calcolo Embedding (con workaround per la vecchia versione) ---
CHUNK_SIZE = 10000
all_embeddings = []

for i in tqdm(range(0, len(prompts_for_embedding), CHUNK_SIZE), desc="Calcolando Embedding con vLLM"):
    batch_prompts = prompts_for_embedding[i:i + CHUNK_SIZE]
    
    # Chiamata a llm.encode() che ora funzionerà
    embedding_outputs = llm.embed(batch_prompts)

    # --- MODIFICA CHIAVE QUI ---
    # L'output contiene già l'embedding finale mediato. Accediamo direttamente.
    for output in embedding_outputs:
        all_embeddings.append(output.outputs.embedding)


# --- 4. Unione e Salvataggio (invariato) ---
final_results = text_results.copy()
for i, user_id in enumerate(user_ids_for_embedding):
    if user_id in final_results:
        # L'embedding è già un array/lista, lo aggiungiamo direttamente
        # (potrebbe essere necessario .tolist() se è un array numpy)
        final_results[user_id]["embedding"] = all_embeddings[i] if isinstance(all_embeddings[i], list) else all_embeddings[i].tolist()

final_output_path = f"{data_dir}/user_results_with_embeddings.pkl"
print(f"\n💾 Salvando il file finale completo in '{final_output_path}'...")
with open(final_output_path, "wb") as f:
    pickle.dump(final_results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✨ Elaborazione completata!")

import json
json_output_path = f"{data_dir}/user_results_with_embeddings_4.json"
print(f"💾 Salvando anche il JSON in '{json_output_path}'...")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(
        final_results,
        f,
        ensure_ascii=False,  # mantiene eventuali caratteri UTF-8
        indent=4            # leggibilità; toglilo se vuoi file più compatto
    )
print("✅ JSON salvato.")