import torch
import json
import os
import pickle  # MODIFIED: Added pickle for efficient saving/loading of Python objects
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from tqdm import tqdm

from LLM.utils.convert_npz_boosted import convert_to_npz
import time

# --- 1. Definisci Percorsi e Costanti ---
model_id = "google/gemma-3-4b-it"
prompt_path = "Dataset/ml/ml-latest-small/final/item_prompts.json"
output_npz_file_last = "Dataset/ml/ml-latest-small/final/ItemPlotGemmaLast"
output_npz_file_mean = "Dataset/ml/ml-latest-small/final/ItemPlotGemmaMean"

# MODIFIED: Define paths for checkpoint files
checkpoint_path_last = "Dataset/ml/ml-latest-small/final/ItemPlotGemmaLast_checkpoint.pkl"
checkpoint_path_mean = "Dataset/ml/ml-latest-small/final/ItemPlotGemmaMean_checkpoint.pkl"
CHECKPOINT_FREQ = 50  # Salva un checkpoint ogni 50 item

# --- 2. Carica il Modello e il Processor ---
print(f"🚀 Loading model: {model_id}...")
print("This might take a few minutes depending on your connection and hardware.")

# Assicurati che la directory per l'output esista
output_dir = os.path.dirname(output_npz_file_last)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# --- 3. Carica i Dati degli Item ---
print(f"🔄 Loading item prompts from '{prompt_path}'...")
try:
    with open(prompt_path, 'r', encoding='utf-8') as f:
        item_prompt_data = json.load(f)
    print(f"✅ Loaded data for {len(item_prompt_data)} items.")
except FileNotFoundError:
    print(f"🚨 ERROR: Input file not found at '{prompt_path}'.")
    exit()
except json.JSONDecodeError:
    print(f"🚨 ERROR: The file '{prompt_path}' is not a valid JSON file.")
    exit()

# --- 4. Logica di Checkpoint e Generazione ---

# MODIFIED: Load from checkpoint if it exists
final_last_results = {}
final_mean_results = {}

if os.path.exists(checkpoint_path_last) and os.path.exists(checkpoint_path_mean):
    print("🔄 Found existing checkpoints. Loading...")
    try:
        with open(checkpoint_path_last, 'rb') as f:
            final_last_results = pickle.load(f)
        with open(checkpoint_path_mean, 'rb') as f:
            final_mean_results = pickle.load(f)
        print(f"✅ Resuming from {len(final_last_results)} processed items.")
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"⚠️ WARNING: Checkpoint files seem corrupted: {e}. Starting from scratch.")
        final_last_results = {}
        final_mean_results = {}
else:
    print("ℹ️ No checkpoints found. Starting from scratch.")

# MODIFIED: Determine which items still need to be processed
processed_ids = set(final_last_results.keys())
items_to_process = [
    (item_id, prompt) for item_id, prompt in item_prompt_data.items()
    if item_id not in processed_ids
]

if not items_to_process:
    print("✅ All items have already been processed. Skipping to final conversion.")
else:
    print(f"🧠 Generating embeddings for the remaining {len(items_to_process)} items...")

    # Itera solo sugli item rimanenti
    for i, (item_id, prompt_text) in enumerate(tqdm(items_to_process, desc="Processing Items")):

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

        with torch.no_grad(): # MODIFIED: Use no_grad for inference to save memory
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1].cpu() # MODIFIED: Move to CPU to free up VRAM

        last_embedding = last_hidden_states[:, -1, :]
        mean_embedding = last_hidden_states.mean(dim=1)

        final_last_results[item_id] = {
            "prompt": prompt_text,
            "embedding": last_embedding.tolist()
        }
        final_mean_results[item_id] = {
            "prompt": prompt_text,
            "embedding": mean_embedding.tolist()
        }

        # MODIFIED: Save a checkpoint periodically
        # We check `i > 0` to avoid saving on the very first item
        if (i + 1) % CHECKPOINT_FREQ == 0 and i > 0:
            print(f"\n--- Saving checkpoint at item {i+1}/{len(items_to_process)} ---")
            try:
                with open(checkpoint_path_last, 'wb') as f_last:
                    pickle.dump(final_last_results, f_last)
                with open(checkpoint_path_mean, 'wb') as f_mean:
                    pickle.dump(final_mean_results, f_mean)
                print("✅ Checkpoint saved successfully.")
            except Exception as e:
                print(f"🚨 ERROR saving checkpoint: {e}")

# --- 5. Final Conversion and Cleanup ---

print("\n✅ Processing complete. Converting all results to NPZ format...")

# The conversion function is called only once with the complete dictionaries
convert_to_npz(final_last_results, output_npz_file_last, id_key='item_id')
convert_to_npz(final_mean_results, output_npz_file_mean, id_key='item_id')

print("🎉 Final NPZ files saved.")

# MODIFIED: Clean up checkpoint files after successful completion
print("🧹 Cleaning up checkpoint files...")
try:
    if os.path.exists(checkpoint_path_last):
        os.remove(checkpoint_path_last)
    if os.path.exists(checkpoint_path_mean):
        os.remove(checkpoint_path_mean)
    print("✨ Done.")
except Exception as e:
    print(f"⚠️ Could not delete checkpoint files: {e}")