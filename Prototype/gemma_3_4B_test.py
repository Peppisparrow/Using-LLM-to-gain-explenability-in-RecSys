from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

# --- QUESTA PARTE DEL TUO CODICE È PERFETTA E RIMANE INVARIATA ---

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Given an game description: A rogue-like pirate adventure game. Explore a skeleton-riddled archipelago in search of a giant monster. Find treasures, weapons and secrets. Can you defeat the terrible kraken?. What is the generes of the game?"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

# Generiamo il testo come prima
with torch.inference_mode():
    # NOTA: Qui otteniamo la sequenza COMPLETA, non solo la parte generata
    full_generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)

decoded = processor.decode(full_generation[0][input_len:], skip_special_tokens=True)
print("--- TESTO GENERATO ---")
print(decoded)
print("-" * 25)


# --- INIZIO DELLA NUOVA SEZIONE: OTTENERE GLI HIDDEN STATES ---

print("\n--- ESTRAZIONE DEGLI HIDDEN STATES ---")

# Ora facciamo un forward pass con l'output completo per ottenere gli stati interni
with torch.inference_mode():
    # Passiamo l'intero output di generazione al modello
    # e chiediamo esplicitamente gli hidden states
    outputs = model(
        input_ids=full_generation,
        output_hidden_states=True,
        return_dict=True
    )

# `outputs.hidden_states` è una tupla. Ogni elemento è un tensore che rappresenta
# gli stati di un layer per tutti i token della sequenza (prompt + risposta).
# La tupla contiene: l'embedding iniziale + l'output di ogni layer del decoder.
all_hidden_states = outputs.hidden_states

# L'ultimo hidden state è quello più ricco di contesto ed è il più usato.
# Corrisponde all'output dell'ultimo layer del decoder.
last_hidden_state = all_hidden_states[-1]

print(f"Numero totale di stati nascosti (layer + embedding): {len(all_hidden_states)}")
print(f"Dimensioni del tensore dell'ultimo hidden state: {last_hidden_state.shape}")
# La forma sarà [batch_size, lunghezza_sequenza_totale, dimensione_hidden_state]

# Ora possiamo separare gli hidden states del prompt da quelli della risposta generata
# usando la lunghezza del prompt che avevamo calcolato prima (input_len)
hidden_states_prompt = last_hidden_state[:, :input_len, :]
hidden_states_generated = last_hidden_state[:, input_len:, :]

print(f"\nDimensioni degli hidden states relativi al PROMPT: {hidden_states_prompt.shape}")
print(f"Dimensioni degli hidden states relativi alla RISPOSTA GENERATA: {hidden_states_generated.shape}")

# Infine, stampiamo un esempio di hidden state per il primo token generato
# (il vettore che rappresenta il primo token della risposta)
primo_token_generato_hidden_state = hidden_states_generated[0, 0, :]
print(f"\nHidden state del primo token generato (prime 10 componenti):\n{primo_token_generato_hidden_state[:]}")