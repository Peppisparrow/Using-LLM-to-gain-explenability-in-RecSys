import torch
import json
import os
import random
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm

# --- 1. Definisci Percorsi e Costanti ---
# Percorso del modello base originale
BASE_MODEL_ID = "google/gemma-3-4b-it"

# Percorso del tuo modello fine-tuned
FINETUNED_MODEL_PATH = "Dataset/ml/ml-latest-small/tuning/gemma/model-no-4bit/gemma3-grpo"

# Percorso dei dati di input (lista di film per utente)
PROMPT_DATA_PATH = "Dataset/ml/ml-latest-small/tuning/user_titles.json"

# Percorsi dei file di output per le raccomandazioni
# Output per il modello base
BASE_MODEL_OUTPUT_PATH = "Dataset/ml/ml-latest-small/tuning/gemma/model-no-4bit/recommendations.json"
# Output per il modello fine-tuned
FINETUNED_MODEL_OUTPUT_PATH = "Dataset/ml/ml-latest-small/tuning/gemma/model-no-4bit/gemma3-grpo/recommendations.json"

# Frequenza di salvataggio del checkpoint (ogni N utenti)
CHECKPOINT_FREQ = 50

def generate_recommendations(model_identifier: str, output_path: str):
    """
    Carica un modello specificato, genera raccomandazioni di film per gli utenti,
    e salva i risultati in un file JSON.

    Args:
        model_identifier (str): L'ID del modello da Hugging Face o il percorso locale.
        output_path (str): Il percorso del file JSON di output.
    """
    print("="*80)
    print(f"🚀 Inizio generazione per il modello: {model_identifier}")
    print("="*80)

    # Assicurati che la directory di output esista
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 2. Carica Modello e Tokenizer con Unsloth ---
    print(f"🧠 Caricamento del modello e tokenizer con Unsloth...")
    # Usiamo FastLanguageModel anche per l'inferenza per coerenza e performance
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_identifier,
    dtype=None, # Usa il default
    )


    # Imposta il template della chat per Gemma, come nel training
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "model"},
    )

    # --- 3. Carica i Dati degli Utenti ---
    print(f"🔄 Caricamento della lista di film da '{PROMPT_DATA_PATH}'...")
    try:
        with open(PROMPT_DATA_PATH, 'r', encoding='utf-8') as f:
            user_movie_data = json.load(f)
        print(f"✅ Caricati dati per {len(user_movie_data)} utenti.")
    except FileNotFoundError:
        print(f"🚨 ERRORE: File di input non trovato in '{PROMPT_DATA_PATH}'. Salto questo modello.")
        return
    except json.JSONDecodeError:
        print(f"🚨 ERRORE: Il file '{PROMPT_DATA_PATH}' non è un JSON valido. Salto questo modello.")
        return

    # --- 4. Logica di Checkpoint e Generazione ---
    checkpoint_path = output_path.replace(".json", "_checkpoint.json")
    final_recommendations = {}

    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Trovato checkpoint. Caricamento progressi da '{checkpoint_path}'...")
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                final_recommendations = json.load(f)
            print(f"✅ Ripresa con {len(final_recommendations)} utenti già processati.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ ATTENZIONE: Impossibile caricare il checkpoint. Si riparte da zero. Errore: {e}")
            final_recommendations = {}
    else:
        print("\n🏁 Nessun checkpoint trovato. Si inizia una nuova esecuzione.")

    print(f"🤖 Generazione raccomandazioni per gli utenti rimanenti...")
    
    user_items_iterator = list(user_movie_data.items())

    for i, (user_id, titles_str) in enumerate(tqdm(user_items_iterator, desc=f"Processing Users ({os.path.basename(model_identifier)})")):
        if user_id in final_recommendations:
            continue

        titles = [title.strip() for title in titles_str.split('),')]
        titles = [title + ')' if not title.endswith(')') else title for title in titles]
        formatted_titles = '\n'.join(f"- {title}" for title in titles)
        
        # Crea il prompt ESATTAMENTE come nel training
        prompt_text = (
            f"The user recently watched the following movies:\n"
            f"{formatted_titles}\n\n"
            "Please recommend a new movie that the user will likely enjoy.\n"
            "⚠️ IMPORTANT: Do NOT recommend any movie from the list above. The recommendation must be a different movie."
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful movie recommendation assistant."},
            {"role": "user", "content": prompt_text},
        ]

        # Applica il template della chat e tokenizza
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[-1]

        # Genera la risposta con temperatura 1.0
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,      # Lunghezza massima della raccomandazione
                use_cache=True,
                do_sample=True,          # NECESSARIO per usare la temperatura
                temperature=1.0,         # Come richiesto
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decodifica solo i nuovi token generati
        recommendation = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        final_recommendations[user_id] = recommendation.strip()

        # Salva il checkpoint
        if (i + 1) % CHECKPOINT_FREQ == 0 and len(final_recommendations) > 0:
            print(f"\n💾 Checkpoint: Salvataggio di {len(final_recommendations)} raccomandazioni in '{checkpoint_path}'...")
            try:
                with open(checkpoint_path, 'w', encoding='utf-8') as f_checkpoint:
                    json.dump(final_recommendations, f_checkpoint, indent=4, ensure_ascii=False)
            except IOError as e:
                print(f"🚨 ERRORE durante il salvataggio del checkpoint: {e}")

    # --- 5. Salva i Risultati Finali e Pulisci ---
    print(f"\n💾 Salvataggio finale di {len(final_recommendations)} raccomandazioni in '{output_path}'...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_recommendations, f, indent=4, ensure_ascii=False)
        
        if os.path.exists(checkpoint_path):
            print(f"🧹 Pulizia del file di checkpoint: '{checkpoint_path}'")
            os.remove(checkpoint_path)
            
        print(f"✨ Processo completato per il modello {model_identifier}!")
    except IOError as e:
        print(f"🚨 ERRORE durante il salvataggio del file finale: {e}")

# --- ESECUZIONE PRINCIPALE ---
if __name__ == "__main__":
    # 1. Genera raccomandazioni con il MODELLO BASE
    generate_recommendations(
        model_identifier=BASE_MODEL_ID,
        output_path=BASE_MODEL_OUTPUT_PATH
    )
    
    # 2. Genera raccomandazioni con il MODELLO FINE-TUNED
    generate_recommendations(
        model_identifier=FINETUNED_MODEL_PATH,
        output_path=FINETUNED_MODEL_OUTPUT_PATH
    )

    print("\n\n🎉 Tutti i processi di generazione sono terminati.")