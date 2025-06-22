# Importa le classi necessarie da transformers e torch
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# 1. Caricamento del Modello e del Tokenizer
# Scegliamo un modello FLAN-T5. "google/flan-t5-base" è un buon punto di partenza
# per un equilibrio tra prestazioni e requisiti di calcolo.
model_name = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 2. Preparazione del Prompt di Input
# Il prompt che vuoi dare al modello.
# Per FLAN-T5, è utile formulare il prompt come un'istruzione.
input_prompt = "Given an game description: A rogue-like pirate adventure game. Explore a skeleton-riddled archipelago in search of a giant monster. Find treasures, weapons and secrets. Can you defeat the terrible kraken?. What is the generes of the game?"
# Tokenizzazione del prompt.
# Il tokenizer converte il testo in una sequenza di numeri (ID dei token)
# che il modello può comprendere.
# return_tensors="pt" restituisce i tensori nel formato PyTorch.
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

# 3. Generazione del Testo in Output
# Utilizziamo il metodo .generate() per ottenere il testo di output.
# Questo è il modo standard per compiti di generazione.
print("--- Generazione del Testo ---")
output_sequences = model.generate(
    input_ids=input_ids,
    max_length=400 # Imposta una lunghezza massima per l'output
)

# Decodifica degli ID di output per ottenere il testo leggibile.
# La decodifica converte la sequenza di ID di token di nuovo in una stringa.
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print(f"Prompt di Input: '{input_prompt}'")
print(f"Testo Generato: '{generated_text}'")
print("-" * 25)


# 4. Estrazione dell'Hidden State dell'Encoder
# Per ottenere gli hidden states, dobbiamo fare una chiamata diretta al modello (forward pass)
# e specificare che vogliamo gli stati nascosti come output.
# NOTA: La chiamata a model() è diversa da model.generate().
print("\n--- Estrazione dell'Hidden State dell'Encoder (Metodo Corretto) ---")

# Si accede direttamente al componente encoder del modello
encoder = model.get_encoder()

# Eseguiamo il forward pass solo sull'encoder
with torch.no_grad():
    # Passiamo gli input_ids tokenizzati direttamente all'encoder
    encoder_outputs = encoder(
        input_ids=input_ids,
        output_hidden_states=True,  # Chiediamo gli hidden states
        return_dict=True
    )

# L'oggetto di output ora proviene direttamente dall'encoder.
# I nomi degli attributi cambiano leggermente:
# - `hidden_states` (invece di `encoder_hidden_states`)
# - `last_hidden_state` (invece di `encoder_last_hidden_state`)

# Estraiamo gli stati nascosti
# Tupla con l'output di ogni layer, a partire dagli embedding
all_hidden_states = encoder_outputs.hidden_states

# Estraiamo l'ultimo stato nascosto (il più comune da usare)
last_hidden_state = encoder_outputs.last_hidden_state

# Analizziamo le dimensioni dei tensori
print(f"Numero di layer nell'encoder (inclusi gli embedding): {len(all_hidden_states)}")
print(f"Dimensioni dell'ultimo hidden state dell'encoder: {last_hidden_state.shape}")

# La forma del tensore è [batch_size, sequence_length, hidden_size]
batch_size, sequence_length, hidden_size = last_hidden_state.shape
print(f"Dettagli Dimensioni: Batch Size={batch_size}, Lunghezza Sequenza={sequence_length}, Dimensione Hidden State={hidden_size}")

print("\nEsempio di valori dell'ultimo hidden state (primo token):")
print(last_hidden_state[0, 0, :10])