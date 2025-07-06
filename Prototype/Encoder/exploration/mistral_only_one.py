import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from title import *
from polars import pl
from tqdm import tqdm
# 1. Caricamento del Modello e del Tokenizer
# Usiamo un modello della famiglia Mistral.
# "Instruct" significa che è stato ottimizzato per seguire istruzioni.
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# NOTA SULLA MEMORIA: Mistral-7B è un modello grande (~14GB in half-precision).
# Se hai problemi di memoria, puoi caricarlo in 4-bit usando bitsandbytes:
# model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
# Altrimenti, caricalo in half-precision (bfloat16) se hai una GPU compatibile.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # Usa bfloat16 per risparmiare memoria
    device_map="auto"           # Distribuisce automaticamente il modello su GPU/CPU
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Preparazione del Prompt di Input
# I modelli "Instruct" funzionano meglio con un template specifico.
# Per Mistral, il template è [INST] ... [/INST].
#input_prompt = generate_user_taste_prompt(10127955)
user = pl.read_csv("Dataset/steam/user.csv")
user_ids = user["user_id"].to_list()
for user_id in tqdm(user_ids, desc="Analizzando i profili utente"):
    input_prompt = generate_user_taste_prompt(user_id)
    formatted_prompt = f"[INST] {input_prompt} [/INST]"

    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
    input_len = input_ids.shape[1]

    output_sequences = model.generate(
        input_ids=input_ids,
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id # Imposta il pad token per evitare warning
    )

    
    generated_ids = output_sequences[0, input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Prompt di Input: '{input_prompt}'")
    print(f"Testo Generato: '{generated_text}'")
    print(f"Testo Generato: '{generated_text.strip()}'")
    print("-" * 25)


    print("\n--- Estrazione dell'Hidden State ---")
    with torch.no_grad():
        model_output = model(
            input_ids=input_ids,
            output_hidden_states=True # Parametro FONDAMENTALE
        )
    all_hidden_states = model_output.hidden_states
    print(f"URCAAA!!!Numero di hidden states estratti: {len(all_hidden_states)}")

    last_hidden_state = all_hidden_states[-1]
    print(f"Numero di layer nel modello: {len(all_hidden_states)}")
    print(f"Dimensioni dell'ultimo hidden state (output del prompt): {last_hidden_state.shape}")
    print(f"Tipo di tensore: {last_hidden_state}")
    embedding_mean_pooling = last_hidden_state.mean(dim=1)
    print(f"Dimensioni dell'output dopo mean pooling: {embedding_mean_pooling.shape}")
