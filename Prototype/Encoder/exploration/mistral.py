import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from title import *
import polars as pl
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
user = pl.read_csv("Dataset/steam/users.csv")
user_ids = user["user_id"].to_list()
output_dict = {}

for user_id in tqdm(user_ids, desc="Analizzando i profili utente"):
    #input_prompt = generate_user_taste_prompt(user_id)
    input_prompt = "Le rose sono rosse, le viole sono blu, il cielo è azzurro e io amo i videogiochi."
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

    with torch.no_grad():
        model_output = model(
            input_ids=input_ids,
            output_hidden_states=True # Parametro FONDAMENTALE
        )
    all_hidden_states = model_output.hidden_states
    last_hidden_state = all_hidden_states[-1]
    embedding_mean_pooling = last_hidden_state.mean(dim=1)
    output_dict['user_id'] = {
        "input_prompt": input_prompt,
        "generated_text": generated_text.strip(),
        "embedding_mean_pooling": embedding_mean_pooling # Converti in lista per serializzazione
    }

# Save as pickle with highest protocol
import pickle
with open("user_embeddings.pkl", "wb") as f:
    pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

