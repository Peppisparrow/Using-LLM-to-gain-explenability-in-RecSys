import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import json
import numpy as np
import polars as pl

# --- 1. Caricamento Dati ---
data_dir = "Dataset/steam/filtering_no_desc_giappo_corean_k10/mid"
user = pl.read_csv(f"{data_dir}/users.csv")
user_ids_from_csv = user["user_id"].to_list()

print(f"📄 Caricando i prompt dal file JSON...")
with open(f'{data_dir}/user_prompts.json', 'r') as f:
    user_taste_prompts = json.load(f)
print("✅ Dati JSON caricati.")

# --- 2. Preparazione degli Input per il Modello ---
# Qui continuiamo il tuo esempio. Creiamo due liste:
# una con i prompt da processare e una con i relativi user_id.

prompts_for_embedding = []
user_ids_for_embedding = []

# Iteriamo su tutti gli utenti del file CSV
for user_id in tqdm(user_ids_from_csv, desc="🔎 Estrazione dei prompt"):
    # Cerchiamo il prompt per l'utente corrente nel file JSON
    # Usiamo str(user_id) perché le chiavi JSON sono stringhe
    prompt = user_taste_prompts.get(str(user_id))
    
    # Aggiungiamo il prompt alla lista solo se esiste
    if prompt:
        prompts_for_embedding.append(prompt)
        user_ids_for_embedding.append(user_id)

print(f"\nFound {len(prompts_for_embedding)} prompts to process.")


# --- 3. Inizializzazione del Modello SentenceTransformer ---
model_name = "google/t5-v1_1-small"
print(f"🚀 Caricando il modello SentenceTransformer '{model_name}'...")
# La libreria userà la GPU automaticamente se disponibile
model = SentenceTransformer(model_name)
print("✅ Modello caricato con successo.")


# --- 4. Calcolo degli Embedding ---
# La libreria gestisce il batching in modo efficiente.
print(f"🧠 Calcolo degli embedding per {len(prompts_for_embedding)} prompt...")
all_embeddings = model.encode(
    prompts_for_embedding,
    show_progress_bar=True,  # Mostra una barra di avanzamento tqdm
    batch_size=128           # Puoi aggiustare questo valore in base alla VRAM della tua GPU
)
print(f"✅ Embedding calcolati. Shape: {all_embeddings.shape}")


# --- 5. Unione e Salvataggio dei Risultati ---
# Costruiamo il dizionario finale da zero
final_results = {}

print("\n✍️ Unendo i risultati e gli embedding...")
for i, user_id in enumerate(user_ids_for_embedding):
    # Per ogni utente, creiamo un dizionario che contiene il prompt originale e il suo embedding
    final_results[user_id] = {
        'input_prompt': prompts_for_embedding[i],
        'embedding': all_embeddings[i].tolist() # Convertiamo l'array NumPy in una lista per la serializzazione
    }

# Salva i risultati finali in un file Pickle
final_output_path = f"{data_dir}/user_results_with_embeddings_t5.pkl"
print(f"💾 Salvataggio dei risultati finali in '{final_output_path}'...")
with open(final_output_path, "wb") as f:
    pickle.dump(final_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
print("✨ Elaborazione completata!")