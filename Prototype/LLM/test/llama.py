import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import polars as pl
from tqdm import tqdm
# Importa la tua funzione per generare i prompt
from title import generate_user_taste_prompt

# --------------------------------------------------------------------------
# NOTA IMPORTANTE: ACCESSO A LLAMA 3
# --------------------------------------------------------------------------
# Per usare i modelli Llama 3, devi:
# 1. Avere un account Hugging Face.
# 2. Richiedere l'accesso al modello sulla sua pagina Hugging Face:
#    https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# 3. Autenticarti nel tuo ambiente di lavoro. Esegui nel terminale:
#    huggingface-cli login
#    E incolla il tuo token di accesso (con permessi di 'read').
# --------------------------------------------------------------------------


# 1. Caricamento del Modello e del Tokenizer (aggiornato a Llama 3)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# NOTA SULLA MEMORIA: Llama-3-8B è un modello grande (~16GB in half-precision).
# Se hai problemi di memoria, puoi caricarlo in 4-bit usando bitsandbytes:
# model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
# Altrimenti, caricalo in half-precision (bfloat16) se hai una GPU compatibile.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Usa bfloat16 per risparmiare memoria
    device_map="auto"            # Distribuisce automaticamente il modello su GPU/CPU
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Caricamento dati
user = pl.read_csv("Dataset\steam/filtering_no_desc_giappo_corean_k10/users.csv")
user_ids = user["user_id"].to_list()

# Loop principale per analizzare gli utenti
for user_id in tqdm(user_ids, desc="Analizzando i profili utente con Llama 3"):
    # Genera il prompt specifico per l'utente
    input_prompt = generate_user_taste_prompt(user_id)

    # --- MODIFICA CHIAVE: Formattazione del Prompt per Llama 3 ---
    # Llama 3 usa un formato di chat specifico. `apply_chat_template` lo crea correttamente.
    messages = [
        {"role": "user", "content": input_prompt},
    ]

    # `apply_chat_template` crea la sequenza di ID corretta per il modello
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, # Aggiunge i token per indicare al modello di rispondere
        return_tensors="pt"
    ).to(model.device)

    # --- Generazione del testo ---
    # Per Llama 3, è buona pratica specificare l'ID del token di fine sequenza (eos_token_id)
    # per garantire che la generazione si fermi correttamente.
    output_sequences = model.generate(
        input_ids=input_ids,
        max_new_tokens=10,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decodifica solo i nuovi token generati
    generated_ids = output_sequences[0, input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Prompt di Input: '{input_prompt}'")
    print(f"Testo Generato: '{generated_text.strip()}'")
    print("-" * 25)

    # --- Estrazione dell'Hidden State (logica invariata) ---
    print("\n--- Estrazione dell'Hidden State ---")
    with torch.no_grad():
        model_output = model(
            input_ids=input_ids,
            output_hidden_states=True  # Parametro FONDAMENTALE
        )
    
    # `model_output.hidden_states` è una tupla di tensori, uno per ogni layer del modello
    all_hidden_states = model_output.hidden_states
    
    # Prendiamo l'output dell'ultimo layer
    last_hidden_state = all_hidden_states[-1]
    
    print(f"URCAAA!!! Numero di layer nel modello: {len(all_hidden_states)}")
    print(f"Dimensioni dell'ultimo hidden state (output del prompt): {last_hidden_state.shape}")
    
    # Eseguiamo il pooling per ottenere un singolo vettore di embedding per l'intero prompt
    # La strategia "mean pooling" calcola la media dei vettori di tutti i token
    embedding_mean_pooling = last_hidden_state.mean(dim=1)
    
    print(f"Dimensioni dell'output dopo mean pooling: {embedding_mean_pooling.shape}")
    print("\n" + "="*50 + "\n")