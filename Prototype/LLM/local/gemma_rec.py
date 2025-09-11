import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from tqdm import tqdm

# Carica il file JSON con le interazioni utente-film
prompt_path = "Dataset/ml/ml-latest-small/final/user_titles.json"
output_path = "Dataset/ml/ml-latest-small/final/recommendations.json"

with open(prompt_path, 'r', encoding='utf-8') as f:
    user_data = json.load(f)

# Carica tokenizer e modello
model_id = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
streamer = TextStreamer(tokenizer)

# Funzione per costruire il prompt
def build_prompt(title_string):
    titles = [t.strip() for t in title_string.split(',') if t.strip()]
    titles = titles[:5]  # Limita a 5 titoli per evitare troppa lunghezza
    formatted_titles = '\n'.join(f"- {title}" for title in titles)
    prompt = (
        f"The user recently watched the following movies:\n"
        f"{formatted_titles}\n\n"
        "Please recommend a new movie that the user will likely enjoy.\n"
        "⚠️ IMPORTANT: Do NOT recommend any movie from the list above. The recommendation must be a different movie.\n"
    )
    return prompt
# per debug prendiamo solo i primi 10 utenti
user_data = {k: user_data[k] for k in list(user_data)[:10]}
# Dizionario per salvare le raccomandazioni
recommendations = {}
# Loop sugli utenti
for user_id, title_string in tqdm(user_data.items(), desc="Generating recommendations"):
    prompt = build_prompt(title_string)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(
        input_ids,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    # Estrai solo la parte nuova generata
    generated_part = decoded_output[len(prompt):].strip()
    recommendations[user_id] = generated_part
# Salva le raccomandazioni in un nuovo JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(recommendations, f, ensure_ascii=False, indent=4)

print(f"✅ Raccomandazioni salvate in '{output_path}'")