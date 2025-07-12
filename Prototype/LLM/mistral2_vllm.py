import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer # <-- Importa il tokenizer
from tqdm import tqdm
import polars as pl
import json
import pickle

def create_llm_engine(model_path, num_gpus):
    print(f"🚀 Creazione/Reset del motore vLLM con {num_gpus} GPU...")
    llm = LLM(
        model=model_path,
        dtype='bfloat16',
        trust_remote_code=True,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.8,  # Usa il 90% della memoria GPU
        enforce_eager=True,
        # Aggiungi un timeout per la comunicazione tra worker
    )
    print("✅ Motore vLLM creato con successo.")
    return llm
# --- 1. Caricamento Dati (invariato) ---
data_dir = "Dataset/steam/filtering_no_desc_giappo_corean_k10/small"
user = pl.read_csv(f"{data_dir}/users.csv")
user_ids = user["user_id"].to_list()
with open(f'{data_dir}/user_prompts.json', 'r') as f:
    user_taste_prompts = json.load(f)

# --- 2. Preparazione dei Prompt ---
# vLLM non ha bisogno di un ciclo di batching manuale.
# Dagli tutti i prompt in una volta sola.

# --- 3. Caricamento Modello con vLLM ---
# vLLM gestisce la quantizzazione in modo diverso.
# 'tensor_parallel_size' divide il modello su più GPU, se disponibili.
model_path = "models/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# --- 2. Preparazione dei Prompt con Troncamento ---
valid_users_data = []
all_prompts = []
# Definiamo una lunghezza massima sicura per i prompt.
# Lasciamo spazio per la risposta (150 token) e un margine di sicurezza.
MAX_PROMPT_LEN = 1024 # Un limite ragionevole per un profilo utente

print(f"Preparing and truncating prompts to a max length of {MAX_PROMPT_LEN} tokens...")
for user_id in tqdm(user_ids, desc="Processing prompts"):
    prompt = user_taste_prompts.get(str(user_id))
    if prompt:
        end_string = (
            f"\n\n---END OF DATA---\n\n"
            f"Based on the data above, provide a summary of this user's tastes:"
        )
        end_tokens = tokenizer.encode(end_string)
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > MAX_PROMPT_LEN:
            max_prompt_tokens = MAX_PROMPT_LEN - len(end_tokens)
            truncated_ids = input_ids[:max_prompt_tokens]
            final_ids = truncated_ids + end_tokens   
        else:
            final_ids = input_ids
        final_prompt_text = tokenizer.decode(final_ids, skip_special_tokens=True)
        formatted_prompt = f"[INST] {final_prompt_text} [/INST]"
        all_prompts.append(formatted_prompt)
        valid_users_data.append({"user_id": str(user_id), "input_prompt": final_prompt_text})

preprocessed_data_path = f"{data_dir}/preprocessed_prompts.json"
print(f"💾 Salvataggio dati in '{preprocessed_data_path}' per uso futuro...")
with open(preprocessed_data_path, 'w', encoding='utf-8') as f:
    json.dump({
        "all_prompts": all_prompts,
        "valid_users_data": valid_users_data
    }, f, ensure_ascii=False, indent=4) # 'indent=4' lo rende leggibile
print("Salvataggio completato.")


preprocessed_data_path = f"{data_dir}/preprocessed_prompts.json"
print(f"💾 Uso i dati precomputati in '{preprocessed_data_path}'")
with open(preprocessed_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    all_prompts = data["all_prompts"]
    valid_users_data = data["valid_users_data"]
print("Caricamento completato.")

# --- NUOVO: Percorso del file di checkpoint e caricamento ---
CHECKPOINT_PATH = "user_results_checkpoint.pkl"

try:
    with open(CHECKPOINT_PATH, "rb") as f:
        output_dict_vllm = pickle.load(f)
    print(f"✅ Checkpoint trovato. Caricati {len(output_dict_vllm)} risultati esistenti.")
except FileNotFoundError:
    output_dict_vllm = {}
    print("ℹ️ Nessun checkpoint trovato. Inizio da zero.")

# --- Inizializzazione Motore (ORA È CORRETTO) ---
num_gpus = torch.cuda.device_count()
llm = create_llm_engine(model_path, num_gpus) # Usa solo la funzione helper
sampling_params = SamplingParams(max_tokens=150, temperature=0.7)
CHUNK_SIZE = 10000

# --- Ciclo di Elaborazione con Salto e Salvataggio Incrementale ---
print(f"Generating responses for {len(all_prompts)} users in chunks of {CHUNK_SIZE}...")

for i in tqdm(range(0, len(all_prompts), CHUNK_SIZE), desc="Processing Chunks"):
    chunk_prompts = all_prompts[i:i + CHUNK_SIZE]
    chunk_user_data = valid_users_data[i:i + CHUNK_SIZE]

    # --- NUOVO: Controlla se il chunk è già stato processato ---
    # Controlliamo solo il primo utente del chunk per efficienza
    if chunk_user_data and chunk_user_data[0]["user_id"] in output_dict_vllm:
        print(f"\n⏭️ Salto del chunk {i}-{i + CHUNK_SIZE} perché già processato.")
        continue

    # --- Logica di generazione e gestione errore (invariata) ---
    try:
        outputs = llm.generate(chunk_prompts, sampling_params)
        
        # Popola i risultati per il chunk corrente
        for j, output in enumerate(outputs):
            user_id = chunk_user_data[j]["user_id"]
            output_dict_vllm[user_id] = {
                "input_prompt": chunk_user_data[j]["input_prompt"],
                "generated_text": output.outputs[0].text.strip(),
            }
        
        # --- NUOVO: Salva il checkpoint DOPO OGNI CHUNK di successo ---
        print(f"\n💾 Salvataggio checkpoint con {len(output_dict_vllm)} risultati totali...")
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(output_dict_vllm, f, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print(f"\n🚨 ERRORE GRAVE nel lotto {i}-{i + CHUNK_SIZE}: {e}")
        print("Il motore è probabilmente instabile. Salto questo lotto e riavvio il motore...")
        del llm
        torch.cuda.empty_cache()
        llm = create_llm_engine(model_path, num_gpus)
        continue

# --- Salvataggio Finale (opzionale, ma buona pratica) ---
print(f"\nSalvataggio finale di {len(output_dict_vllm)} risultati...")
with open(f"{data_dir}/user_results_final.pkl", "wb") as f:
    pickle.dump(output_dict_vllm, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Elaborazione completata!")

input_pkl_path = f"{data_dir}/user_results_final.pkl"
output_json_path = f"{data_dir}/user_results_final.json"

# --- 1. Caricamento del file Pickle ---
print(f"🔄 Caricando il file pickle da '{input_pkl_path}'...")
try:
    with open(input_pkl_path, "rb") as f:
        data_from_pkl = pickle.load(f)
    print(f"✅ Caricati {len(data_from_pkl)} record.")
except FileNotFoundError:
    print(f"🚨 Errore: File non trovato in '{input_pkl_path}'. Assicurati che il percorso sia corretto.")
    exit()

# --- 2. Conversione nel formato JSON desiderato ---
print("⚙️ Convertendo i dati nel formato JSON...")
json_output_list = []
# Itera su chiavi (user_id) e valori (il dizionario interno) del dizionario caricato
for user_id, user_data in tqdm(data_from_pkl.items(), desc="Formatting JSON"):
    # Crea un nuovo dizionario per l'output JSON
    record = {
        "user_id": user_id,
        "input_prompt": user_data.get("input_prompt", ""), # .get() per sicurezza
        "generated_text": user_data.get("generated_text", "")
    }
    json_output_list.append(record)

# --- 3. Salvataggio del file JSON ---
print(f"💾 Salvando il file JSON in '{output_json_path}'...")
with open(output_json_path, "w", encoding="utf-8") as f:
    # Usa indent=4 per una formattazione leggibile
    json.dump(json_output_list, f, ensure_ascii=False, indent=4)

print("✨ Conversione completata con successo!")