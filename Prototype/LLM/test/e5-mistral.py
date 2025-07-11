import torch
from sentence_transformers import SentenceTransformer # <-- LIBRERIA CHIAVE
from tqdm import tqdm
import pickle
import json
import numpy as np

# --- 1. Caricamento Dati ---
# Questa sezione rimane invariata
data_dir = "Dataset/steam/filtering_no_desc_giappo_corean_k10"
results_path = f"{data_dir}/user_results_final.pkl"
print(f"🔄 Caricando i risultati da '{results_path}'...")
try:
    with open(results_path, "rb") as f:
        text_results = pickle.load(f)
    print(f"✅ Caricati {len(text_results)} record.")
except FileNotFoundError:
    print(f"🚨 Errore: File non trovato. Esegui prima lo script di generazione del testo.")
    exit()

# --- MODIFICA CRUCIALE: Formattazione del Prompt ---
# I modelli E5 non usano tag speciali come [INST]. Usiamo direttamente il prompt.
prompts_for_embedding = [
    data['input_prompt']
    for data in text_results.values()
]
user_ids_for_embedding = list(text_results.keys())

# --- 2. Inizializzazione del Modello SentenceTransformer ---
# Usiamo la libreria standard per i modelli di embedding come E5
model_name = "intfloat/e5-mistral-7b-instruct"
print(f"🚀 Caricamento del modello SentenceTransformer '{model_name}'...")
# La libreria userà automaticamente la GPU se disponibile
model = SentenceTransformer(model_name, trust_remote_code=True)
print("✅ Modello caricato con successo.")


# --- 3. Calcolo degli Embedding ---
# La libreria gestisce il batching e il pooling in modo ottimale
print(f"🧠 Calcolo degli embedding per {len(prompts_for_embedding)} prompt...")
all_embeddings = model.encode(
    prompts_for_embedding,
    show_progress_bar=True,  # Mostra una barra di avanzamento tqdm
    batch_size=1            # Puoi aggiustare questo valore in base alla VRAM della tua GPU
)
print(f"✅ Embedding calcolati. Shape: {all_embeddings.shape}")


# --- 4. Unione e Salvataggio ---
# Questa logica rimane invariata
final_results = text_results.copy()
for i, user_id in enumerate(user_ids_for_embedding):
    if user_id in final_results:
        # L'output di .encode() è un array NumPy, lo convertiamo in lista per salvarlo
        final_results[user_id]["embedding"] = all_embeddings[i].tolist()

# Salvataggio del file Pickle finale
final_output_path = f"{data_dir}/user_results_with_embeddings_e5.pkl"
print(f"\n💾 Salvando il file finale completo in '{final_output_path}'...")
with open(final_output_path, "wb") as f:
    pickle.dump(final_results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✨ Elaborazione completata!")