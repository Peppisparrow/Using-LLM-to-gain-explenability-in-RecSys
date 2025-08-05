import torch
import json
import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from tqdm import tqdm

from LLM.utils.convert_npz_boosted import convert_to_npz
import time # Aggiunto per eventuali pause o logging

# --- 1. Definisci Percorsi e Costanti (Hardcoded) ---
model_id = "google/gemma-3-4b-it"
prompt_path = "Dataset/ml/ml-latest-small/final/item_prompts.json"
output_npz_file_last = "Dataset/ml/ml-latest-small/final/ItemPlotGemmaLast"
output_npz_file_mean = "Dataset/ml/ml-latest-small/final/ItemPlotGemmaMean"
CHECKPOINT_FREQ = 50  # Salva un checkpoint ogni 50 utenti

# --- 2. Carica il Modello e il Processor ---
print(f"🚀 Loading model: {model_id}...")
print("This might take a few minutes depending on your connection and hardware.")

# Assicurati che la directory per l'output esista

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# --- 3. Carica i Dati degli Utenti ---
print(f"🔄 Loading user movie lists from '{prompt_path}'...")
try:
    with open(prompt_path, 'r', encoding='utf-8') as f:
        user_movie_data = json.load(f)
    print(f"✅ Loaded data for {len(user_movie_data)} users.")
except FileNotFoundError:
    print(f"🚨 ERROR: Input file not found at '{prompt_path}'.")
    exit()
except json.JSONDecodeError:
    print(f"🚨 ERROR: The file '{prompt_path}' is not a valid JSON file.")
    exit()

# --- 4. Logica di Checkpoint e Generazione delle Raccomandazioni ---

print(f"🧠 Generating recommendations for the remaining users...")

# Itera su tutti gli utenti, usando l'indice per il checkpoint
total_users = len(user_movie_data)
# Usa un iteratore per il ciclo in modo da poterlo passare a tqdm
user_items_iterator = list(user_movie_data.items())
final_last_results = {}
final_mean_results = {}
for i, (user_id, prompt_text) in enumerate(tqdm(user_items_iterator, desc="Processing Users")):

    # Usa il template di conversazione del modello per formattare il prompt
    messages = [
        {"role": "system", "content":  [{"type": "text", "text": "You are a helpful movie recommendation assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]

    # Il processor formatta e tokenizza il prompt
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]

    last_embedding = last_hidden_states[:, -1, :]  # Prendi l'ultimo hidden state
    mean_embedding = last_hidden_states.mean(dim=1)  # Media dell'ultimo hidden state
    
    final_last_results[user_id] = {
        "prompt": prompt_text,
        # Convert the NumPy array to a list for standard serialization with pickle.
        "embedding": last_embedding.tolist() 
    }
    final_mean_results[user_id] = {
        "prompt": prompt_text,
        # Convert the NumPy array to a list for standard serialization with pickle.
        "embedding": mean_embedding.tolist() 
    }
# Save the final dictionary to a pickle file for easy reuse
    # Esegue la funzione di conversione
convert_to_npz(final_last_results, output_npz_file_last, id_key='item_id')
convert_to_npz(final_mean_results, output_npz_file_mean, id_key='item_id')
