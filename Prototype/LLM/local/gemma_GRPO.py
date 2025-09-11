import torch
import json
import os
import random
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

# --- 1. Definisci Percorsi, Costanti e Parametri di Training ---
model_id = "google/gemma-3-4b-it"
prompt_path = "Dataset/ml/ml-latest-small/tuning/user_titles.json"
output_dir = "Dataset/ml/ml-latest-small/tuning/gemma/model-no-4bit"
output_model_path = f"{output_dir}/gemma3-grpo"

import re
import string

# Definiamo un set di articoli in varie lingue da ignorare.
# Usare un set è molto più veloce per le ricerche.
ARTICLES_TO_REMOVE = {
    'the', 'a', 'an', 'le', 'la', 'les', 'l\'', 'il', 'lo', 'i', 'gli', 'un', 'una',
    'un\'', 'el', 'los', 'las', 'ein', 'eine', 'der', 'die', 'das'
}

# Funzione per pulire e normalizzare una stringa di testo
def normalize_text(text: str) -> str:
    """
    Converte il testo in minuscolo, rimuove la punteggiatura, gli articoli comuni
    e normalizza gli spazi.
    """
    # 1. Minuscolo
    text = text.lower()
    # 2. Rimuovi la punteggiatura
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Rimuovi gli articoli
    words = text.split()
    words = [word for word in words if word not in ARTICLES_TO_REMOVE]
    text = ' '.join(words)
    # 4. Normalizza gli spazi (rimuove spazi multipli)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Funzione per analizzare i titoli complessi e restituire tutte le varianti
def parse_complex_title(title_string: str) -> list[str]:
    """
    Analizza un titolo complesso per estrarre le sue varianti.
    Gestisce:
    - Formato "Nome, Il" -> "Il Nome"
    - Titoli alternativi tra parentesi.
    - Anno di uscita.
    Restituisce una lista di titoli normalizzati.
    """
    # Rimuovi l'anno di uscita, es. (1995)
    title_string = re.sub(r'\s*\(\d{4}\)$', '', title_string).strip()

    # Separa i titoli principali e quelli tra parentesi
    # es. "City of Lost Children, The (Cité des enfants perdus, La)" ->
    # ["City of Lost Children, The", "Cité des enfants perdus, La"]
    potential_titles = re.split(r'\(|\)', title_string)
    potential_titles = [t.strip() for t in potential_titles if t.strip()]

    cleaned_titles = set() # Usiamo un set per evitare duplicati
    for title in potential_titles:
        # Gestisci il formato "Title, The" -> "The Title"
        match = re.match(r'^(.*?), (the|a|an|le|la|il|lo|el|der|die|das)$', title, re.IGNORECASE)
        if match:
            # Riordina: es. "City of Lost Children, The" -> "The City of Lost Children"
            reordered_title = f"{match.group(2)} {match.group(1)}"
            cleaned_titles.add(normalize_text(reordered_title))
        else:
            cleaned_titles.add(normalize_text(title))
            
    return list(cleaned_titles)

# --- 2. Preparazione dei Dati (con split 90/10) ---
print(f"🔄 Loading and preparing user movie data from '{prompt_path}'...")
try:
    with open(prompt_path, 'r', encoding='utf-8') as f:
        user_movie_data = json.load(f)
except FileNotFoundError:
    print(f"🚨 ERROR: Input file not found at '{prompt_path}'.")
    exit()

# Struttura dati per il training: lista di dizionari
training_data = []
# usa solo i primi 5 utenti per il test
# Itera su ogni utente per creare i dati di training
for user_id, titles_str in tqdm(user_movie_data.items(), desc="Preparing Data"):
    titles = [title.strip() for title in titles_str.split('),') if title.strip()]
    # aggiungi una parentesi chiusa se a tutti i titoli
    titles = [title + ')' if not title.endswith(')') else title for title in titles]
    # Ignora utenti con meno di 5 film per uno split significativo
    if len(titles) < 5:
        continue
    
    # Mescola i titoli per garantire uno split casuale
    random.shuffle(titles)
    
    # Esegui lo split 90/10
    split_index = int(len(titles) * 0.9)
    prompt_titles = titles[:split_index]
    target_titles = titles[split_index:]
    print(f"User {user_id} - Prompt titles: {len(prompt_titles)}, Target titles: {len(target_titles)}")
    print(f"User {user_id} - Prompt titles: {target_titles}")
    # Se la lista target è vuota (per arrotondamento), sposta un film
    if not target_titles and prompt_titles:
        target_titles.append(prompt_titles.pop())
    
    # Formatta il prompt
    formatted_titles = '\n'.join(f"- {title}" for title in prompt_titles)
    prompt_text = (
        f"The user recently watched the following movies:\n"
        f"{formatted_titles}\n\n"
        "Please recommend a new movie that the user will likely enjoy.\n"
        "⚠️ IMPORTANT: Do NOT recommend any movie from the list above. The recommendation must be a different movie."
    )

    # Crea il messaggio nel formato chat
    # Usiamo il formato `apply_chat_template` senza tokenizzazione qui
    # perché il trainer lo gestirà internamente.
    messages = [
        {"role": "system", "content": "You are a helpful movie recommendation assistant."},
        {"role": "user", "content": prompt_text},
    ]

    # Scegliamo un titolo target a caso come riferimento per la ricompensa
    # La funzione di ricompensa controllerà la presenza in *tutti* i target_titles
    # Usiamo un solo titolo come "completion di riferimento" per il logging
    reference_completion = random.choice(target_titles)

    training_data.append({
        "prompt": messages,
        "reference_completion": f"Based on your watch history, a great recommendation would be **{reference_completion}**.",
        "target_movies": target_titles # Lista di film target per la ricompensa
    })

# Converti in un oggetto Dataset di Hugging Face
train_dataset = Dataset.from_list(training_data)
print(f"✅ Prepared dataset with {len(train_dataset)} examples.")

# --- 3. Carica Modello e Tokenizer con Unsloth ---
print(f"🚀 Loading model '{model_id}' with Unsloth for 4-bit training...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    dtype=None, # Usa il default
)

# Aggiungi LoRA al modello per un training efficiente (PEFT)
model = FastLanguageModel.get_peft_model(
    model,
    r=16, # Rank LoRA, valori comuni sono 8, 16, 32, 64
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# Imposta il template della chat per Gemma
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma",
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "model"},
)

# --- 4. Definisci la Funzione di Ricompensa e il Trainer GRPO ---

def reward_function(
    completions,
    **kwargs,
):
    """
    Funzione di ricompensa potenziata per GRPO.
    Utilizza una logica di matching avanzata per confrontare le generazioni
    con i film target.
    """
    list_of_target_movies = kwargs["target_movies"]
    rewards = []

    for i in range(len(completions)):
        try:
            # Estrai il testo generato e normalizzalo UNA SOLA VOLTA
            generated_text = completions[i][0]['content']
            normalized_generated_text = normalize_text(generated_text)
        except (IndexError, KeyError, TypeError):
            normalized_generated_text = ""

        match_found = False
        
        # Itera sulla lista dei film target per questo specifico esempio
        target_movie_list = list_of_target_movies[i]
        
        # Questo ciclo è necessario nel caso la lista fosse annidata
        # come abbiamo visto nel debugging precedente
        flat_target_movie_list = []
        if target_movie_list and isinstance(target_movie_list[0], list):
            flat_target_movie_list = [movie for sublist in target_movie_list for movie in sublist]
        else:
            flat_target_movie_list = target_movie_list

        for title_string in flat_target_movie_list:
            if not isinstance(title_string, str): continue
            
            # 1. Analizza il titolo complesso per ottenere tutte le varianti normalizzate
            possible_normalized_titles = parse_complex_title(title_string)
            
            # 2. Controlla se una qualsiasi delle varianti è presente nel testo generato
            if any(title in normalized_generated_text for title in possible_normalized_titles if title):
                match_found = True
                break  # Trovata corrispondenza, esci dal ciclo interno

        if match_found:
            print("✅ URCAAAAA, TROVATO")
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards

# Configurazione per GRPO
# grpo_args = GRPOConfig(
#     output_dir=output_dir,
#     beta=0.1,  # Parametro chiave di GRPO/DPO. Controlla quanto ci si allontana dal modello di riferimento.
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     gradient_checkpointing=True,
#     learning_rate=5e-5,
#     num_train_epochs=1, # Per un test, 1 epoca è sufficiente. Aumenta a 2-3 per risultati migliori.
#     save_strategy="epoch",
#     logging_steps=10,
#     lr_scheduler_type="cosine",
#     warmup_steps=10,
#     optim="adamw_8bit",
#     bf16=not torch.cuda.is_bf16_supported(), # Usa bf16 se supportato
#     fp16=torch.cuda.is_bf16_supported(),
#     remove_unused_columns=False,
#     pad_token_id=tokenizer.pad_token_id,
#     max_length=1024,      # Lunghezza massima input + output
#     max_prompt_length=200, # Lunghezza massima del prompt
# )
grpo_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_completion_length = 200,
    num_train_epochs=5,          # MODIFICATO
    save_strategy="epoch",       # AGGIUNTO
    report_to = "none", # Can use Weights & Biases
    output_dir = output_dir,

    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

# Istanzia il Trainer
trainer = GRPOTrainer(
    model=model,
    args=grpo_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function,
)

# --- 5. Esegui il Training ---
print("\n🧠 Starting GRPO fine-tuning...")
trainer.train()
print("✨ Training complete!")

# --- 6. Salva il Modello Fine-Tuned ---
print(f"💾 Saving fine-tuned model to '{output_model_path}'...")
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print("✅ Model saved successfully.")

# --- Esempio di Inferenza (opzionale) ---
print("\n🧪 Running a test inference with the fine-tuned model...")
# Fai il merge degli adapter LoRA nel modello base per un'inferenza più veloce
if hasattr(model, "merge_and_unload"):
    model.merge_and_unload()

# Prendi un esempio dal nostro dataset
test_case = train_dataset[0]
prompt = tokenizer.apply_chat_template(test_case["prompt"], tokenize=False, add_generation_prompt=True)

inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True, pad_token_id=tokenizer.pad_token_id)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- TEST INFERENCE ---")
print(response)
print("\n--- Target movies for this user were: ---")
print(test_case["target_movies"])