import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import json
import numpy as np
from LLM.utils.convert_npz import convert_to_npz

# --- 1. Caricamento Dati ---
# The data loading part remains the same.
data_dir = "Dataset/steam/filtering_no_desc_giappo_corean_k10/small"
results_path = f"{data_dir}/user_results_final.pkl"
print(f"🔄 Loading results from '{results_path}'...")
try:
    with open(results_path, "rb") as f:
        text_results = pickle.load(f)
    print(f"✅ Loaded {len(text_results)} records.")
except FileNotFoundError:
    print(f"🚨 Error: File not found. Please run the text generation script first.")
    exit()

# --- CRITICAL CHANGE: Prompt Formatting ---
# T5 models do not use special instruction tags. We use the raw prompt.
prompts_for_embedding = [
    data['input_prompt']
    for data in text_results.values()
]
user_ids_for_embedding = list(text_results.keys())

# --- 2. Initialize SentenceTransformer Model ---
# We use sentence-transformers, the standard for embedding with T5 models.
model_name = "google/t5-v1_1-small"
print(f"🚀 Loading SentenceTransformer model '{model_name}'...")
# The library automatically uses the GPU if available.
model = SentenceTransformer(model_name)
print("✅ Model loaded successfully.")

# --- 3. Calculate Embeddings ---
# sentence-transformers is highly efficient and handles batching automatically.
print(f"🧠 Calculating embeddings for {len(prompts_for_embedding)} prompts...")
all_embeddings = model.encode(
    prompts_for_embedding,
    show_progress_bar=True,  # This will display a tqdm progress bar
    batch_size=128            # You can adjust this based on your VRAM
)
print(f"✅ Embeddings calculated. Shape: {all_embeddings.shape}")


# --- 4. Merge and Save Results ---
# This logic remains the same.
final_results = text_results.copy()
for i, user_id in enumerate(user_ids_for_embedding):
    if user_id in final_results:
        # The output from .encode() is a NumPy array. We convert it to a list for serialization.
        final_results[user_id]["embedding"] = all_embeddings[i].tolist()

# Save the final results to a Pickle file
final_output_path = f"{data_dir}/user_results_with_embeddings_t5.pkl"
print(f"\n💾 Saving final results to '{final_output_path}'...")
with open(final_output_path, "wb") as f:
    pickle.dump(final_results, f, protocol=pickle.HIGHEST_PROTOCOL)
print("✨ Processing complete!")

results_path = f"{data_dir}/user_results_with_embeddings_t5.pkl"
with open(results_path, "rb") as f:
        text_results = pickle.load(f)

output_npz_file = f"{data_dir}/user_embeddings_compressed_t5"
    
    # Esegue la funzione di conversione
convert_to_npz(text_results, output_npz_file)

