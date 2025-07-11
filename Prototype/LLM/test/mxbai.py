import torch
from sentence_transformers import SentenceTransformer # <-- Key Library
from tqdm import tqdm
import pickle
import numpy as np

# --- 1. Data Loading ---
# This section remains unchanged.
data_dir = "Dataset/steam/filtering_no_desc_giappo_corean_k10"
results_path = f"{data_dir}/user_results_final.pkl"
print(f"🔄 Loading results from '{results_path}'...")
try:
    with open(results_path, "rb") as f:
        text_results = pickle.load(f)
    print(f"✅ Loaded {len(text_results)} records.")
except FileNotFoundError:
    print(f"🚨 Error: File not found. Please run the text generation script first.")
    exit()

# --- 2. Prompt Formatting ---
# The prompt structure is kept simple for mxbai, which is effective.
# For optimal performance with mxbai-embed-large, it's recommended to prefix each input with "Represent this sentence for searching relevant passages: "
prompts_for_embedding = [
    f"Represent this sentence for searching relevant passages: {data['input_prompt']}"
    for data in text_results.values()
]
user_ids_for_embedding = list(text_results.keys())

# --- 3. Initializing the SentenceTransformer Model ---
# We replace the E5 model with the official mxbai-embed-large-v1 model.
model_name = "mixedbread-ai/mxbai-embed-large-v1"
print(f"🚀 Loading the SentenceTransformer model '{model_name}'...")
# The library will automatically use the GPU if available.
model = SentenceTransformer(model_name)
print("✅ Model loaded successfully.")


# --- 4. Calculating Embeddings ---
# The library efficiently handles batching and pooling.
print(f"🧠 Calculating embeddings for {len(prompts_for_embedding)} prompts...")
all_embeddings = model.encode(
    prompts_for_embedding,
    show_progress_bar=True,  # Displays a tqdm progress bar
    batch_size=10            # You can adjust this value based on your GPU's VRAM
)
print(f"✅ Embeddings calculated. Shape: {all_embeddings.shape}")


# --- 5. Merging and Saving ---
# This logic remains the same.
final_results = text_results.copy()
for i, user_id in enumerate(user_ids_for_embedding):
    if user_id in final_results:
        # The output of .encode() is a NumPy array, which we convert to a list for serialization.
        final_results[user_id]["embedding"] = all_embeddings[i].tolist()

# Save the final Pickle file.
final_output_path = f"{data_dir}/user_results_with_embeddings_mxbai.pkl"
print(f"\n💾 Saving the complete final file to '{final_output_path}'...")
with open(final_output_path, "wb") as f:
    pickle.dump(final_results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✨ Processing complete!")