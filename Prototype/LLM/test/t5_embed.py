import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import json
import numpy as np
from LLM.utils.convert_npz_boosted import convert_to_npz

# --- 1. Caricamento Dati ---
# The data loading part remains the same.
data_dir = "Dataset/steam/filtering_no_desc_giappo_corean_k10"
# Input file containing the prompts for each game
prompts_path = f"{data_dir}/game_prompts.json" 
# Final output file that will store prompts and their embeddings
output_path = f"{data_dir}/game_embeddings_t5.pkl" 

# --- 2. Load Game Prompts ---
print(f"🔄 Loading game prompts from '{prompts_path}'...")
try:
    with open(prompts_path, "r", encoding='utf-8') as f:
        # game_prompts will be a dictionary like {'app_id_1': 'prompt_1', ...}
        game_prompts = json.load(f)
    print(f"✅ Loaded {len(game_prompts)} game prompts.")
except FileNotFoundError:
    print(f"🚨 ERROR: File not found. Please run the game prompt generation script first.")
    exit()
except json.JSONDecodeError:
    print(f"🚨 ERROR: The file '{prompts_path}' is not a valid JSON file.")
    exit()

# --- 3. Prepare Data for Embedding ---
# Extract the prompts and their corresponding app_ids into separate lists.
# It's crucial to maintain the same order for later mapping.
prompts_for_embedding = list(game_prompts.values())
app_ids_for_embedding = list(game_prompts.keys())

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

# --- 6. Combine Prompts with Embeddings and Save ---
# Create a new dictionary to store the final, structured results.
final_results = {}
for i, app_id in enumerate(tqdm(app_ids_for_embedding, desc="💾 Structuring final results")):
    # For each app_id, create a dictionary containing its original prompt and the new embedding.
    final_results[app_id] = {
        "prompt": prompts_for_embedding[i],
        # Convert the NumPy array to a list for standard serialization with pickle.
        "embedding": all_embeddings[i].tolist() 
    }

# Save the final dictionary to a pickle file for easy reuse.
print(f"\nSaving the final combined data to '{output_path}'...")
with open(output_path, "wb") as f:
    # Using the highest protocol is more efficient for modern Python versions.
    pickle.dump(final_results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✨ Processing complete! All game prompts now have a corresponding embedding.")
with open(output_path, "rb") as f:
        text_results = pickle.load(f)

output_npz_file = f"{data_dir}/game_embeddings_t5"
    
    # Esegue la funzione di conversione
convert_to_npz(text_results, output_npz_file, id_key='app_id')