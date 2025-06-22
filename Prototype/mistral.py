import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from title import *
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
input_prompt = generate_user_taste_prompt(10127955)
formatted_prompt = f"[INST] {input_prompt} [/INST]"

# Tokenizzazione del prompt.
# Invia i tensori al dispositivo corretto (es. 'cuda' se si usa la GPU).
input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
input_len = input_ids.shape[1]
print(f"Lunghezza del prompt tokenizzato: {input_len} token")

# 3. Generazione del Testo in Output
print("--- Generazione del Testo ---")
# Usiamo .generate() come prima.
# max_new_tokens limita il numero di token generati *dopo* il prompt.
output_sequences = model.generate(
    input_ids=input_ids,
    max_new_tokens=1000,
    pad_token_id=tokenizer.eos_token_id # Imposta il pad token per evitare warning
)

# A differenza di T5, l'output di generate() include anche il prompt di input.
# Dobbiamo decodificare solo i token generati.
generated_ids = output_sequences[0, input_ids.shape[-1]:] # Seleziona solo i nuovi token
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"Prompt di Input: '{input_prompt}'")
print(f"Testo Generato: '{generated_text.strip()}'")
print("-" * 25)


# 4. Estrazione dell'Hidden State (equivalente all'output dell'encoder)
# Eseguiamo un forward pass del modello solo con il prompt per ottenere gli stati nascosti.
print("\n--- Estrazione dell'Hidden State ---")
with torch.no_grad():
    model_output = model(
        input_ids=input_ids,
        output_hidden_states=True # Parametro FONDAMENTALE
    )

# `model_output.hidden_states` è una tupla con l'output di ogni layer del modello.
all_hidden_states = model_output.hidden_states

# L'equivalente dell' "encoder_last_hidden_state" di T5 è l'ultimo stato nascosto
# della computazione sul prompt. Corrisponde all'ultimo tensore nella tupla.
last_hidden_state = all_hidden_states[-1]

# Analizziamo le dimensioni del tensore
print(f"Numero di layer nel modello: {len(all_hidden_states)}")
print(f"Dimensioni dell'ultimo hidden state (output del prompt): {last_hidden_state.shape}")

# La forma del tensore è [batch_size, sequence_length, hidden_size]
# - batch_size: 1
# - sequence_length: Il numero di token nel nostro prompt formattato.
# - hidden_size: La dimensione dello stato nascosto di Mistral-7B (è 4096).
batch_size, sequence_length, hidden_size = last_hidden_state.shape
print(f"Dettagli Dimensioni: Batch Size={batch_size}, Lunghezza Sequenza={sequence_length}, Dimensione Hidden State={hidden_size}")