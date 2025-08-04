import torch
import json
import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from tqdm import tqdm

# --- 1. Define File Paths and Model ID (Hardcoded) ---
model_id = "google/gemma-3-4b-it"  # Modello Gemma 2 ottimizzato per l'italiano, ma risponderà in inglese se richiesto
prompt_path = "Dataset/ml/ml-latest-small/final/user_titles.json"
output_path = "Dataset/ml/ml-latest-small/final/recommendations.json"

# --- 2. Load Language Model and Processor ---
print(f"🚀 Loading model: {model_id}...")
print("This might take a few minutes depending on your connection and hardware.")

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# --- 3. Load User Movie Data ---
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

# --- 4. Generate Recommendations ---
final_recommendations = {}
print(f"🧠 Generating recommendations for {len(user_movie_data)} users...")
# usa solo i primi 10 utenti per il debug
for user_id, titles_str in tqdm(user_movie_data.items(), desc="Processing Users"):
    titles = [title.strip() for title in titles_str.split(',')]
    formatted_titles = '\n'.join(f"- {title}" for title in titles)
    
    # Crea il prompt in inglese
    prompt_text = (
        f"The user recently watched the following movies:\n"
        f"{formatted_titles}\n\n"
        "Please recommend a new movie that the user will likely enjoy.\n"
        "⚠️ IMPORTANT: Do NOT recommend any movie from the list above. The recommendation must be a different movie."
    )

    # Usa il template di conversazione del modello per formattare il prompt
    messages = [
        {"role": "system", "content":  [{"type": "text", "text": "You are a helpful movie recommendation assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]

    # Il processor formatta e tokenizza il prompt nel modo corretto per Gemma
    inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    # Genera la risposta
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    # Decodifica solo la parte generata, escludendo il prompt
    input_len = inputs["input_ids"].shape[-1]
    recommendation = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    final_recommendations[user_id] = recommendation.strip()

# --- 5. Save Results ---
print(f"\n💾 Saving final recommendations to '{output_path}'...")
try:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_recommendations, f, indent=4, ensure_ascii=False)
    print("✨ Processing complete! All recommendations have been saved.")
except IOError as e:
    print(f"🚨 ERROR saving output file: {e}")