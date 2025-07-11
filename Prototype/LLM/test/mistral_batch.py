import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import polars as pl
from tqdm import tqdm
import json
import pickle

# --- 0. PARAMETRI DI CONFIGURAZIONE ---
BATCH_SIZE = 8  # Prova ad aumentare questo valore (es. 32, 64) se la VRAM lo permette.
                 # Un batch più grande di solito significa maggiore velocità.

# --- 1. CARICAMENTO DEL MODELLO E DEL TOKENIZER ---
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

print("Caricamento del modello in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Impostazioni del tokenizer per il batching
tokenizer.padding_side = "left"  # Fondamentale per la generazione con modelli causali
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Modello caricato con successo.")

# --- 2. PREPARAZIONE DEI DATI ---
data_dir = "Dataset/steam/descr_filtering"
try:
    user_df = pl.read_csv(f"{data_dir}/users.csv")
    user_ids = user_df["user_id"].to_list()
    with open(f'{data_dir}/user_prompts.json', 'r', encoding='utf-8') as f:
        user_taste_prompts = json.load(f)
except FileNotFoundError as e:
    print(f"Errore: File non trovato - {e}. Assicurati che i percorsi siano corretti.")
    exit()

output_dict = {}

# --- 3. LOOP DI INFERENZA IN BATCH (LOGICA CORRETTA) ---
print(f"Inizio analisi di {len(user_ids)} profili con una batch size di {BATCH_SIZE}...")

for i in tqdm(range(0, len(user_ids), BATCH_SIZE), desc="Processando in batch"):
    batch_user_ids = user_ids[i:i + BATCH_SIZE]
    
    batch_prompts_with_template = []
    original_prompts_batch = []
    valid_batch_ids = []

    for user_id in batch_user_ids:
        prompt = user_taste_prompts.get(str(user_id))
        if prompt:
            batch_prompts_with_template.append(f"[INST] {prompt} [/INST]")
            original_prompts_batch.append(prompt)
            valid_batch_ids.append(user_id)

    if not batch_prompts_with_template:
        continue

    # Tokenizza l'intero batch in una sola volta
    inputs = tokenizer(
        batch_prompts_with_template,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048 # Limite di sicurezza per la memoria
    ).to(model.device)

    # Esegui la generazione e il calcolo degli embedding in un'unica passata
    with torch.no_grad():
        # **Non usiamo più .generate() che è più lento e complesso per gli embedding**
        # Eseguiamo una singola "forward pass" per ottenere gli stati nascosti
        model_outputs = model(**inputs, output_hidden_states=True)
        
        # ORA ESEGUIAMO UNA GENERAZIONE SEPARATA E VELOCE
        # Questo è più rapido perché non deve calcolare gli hidden states
        generated_sequences = model.generate(
            **inputs,
            max_new_tokens=10
        )

    # --- ESTRAZIONE OTTIMIZZATA DEI RISULTATI ---

    # 1. Calcolo degli Embedding in Batch (Masked Average Pooling)
    last_hidden_state = model_outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask']
    
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled_embeddings = sum_embeddings / sum_mask

    # 2. Decodifica del Testo Generato in Batch
    generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

    # Salva i risultati per ogni elemento del batch
    for j in range(len(valid_batch_ids)):
        current_user_id = valid_batch_ids[j]
        # Pulisci il testo generato per rimuovere il prompt originale
        clean_text = generated_texts[j].replace(batch_prompts_with_template[j], "").strip()
        
        output_dict[str(current_user_id)] = {
            "input_prompt": original_prompts_batch[j],
            "generated_text": clean_text,
            "embedding": mean_pooled_embeddings[j].cpu().tolist()
        }

print("Analisi completata.")

# --- 4. SALVATAGGIO FINALE ---
output_file_path = "user_embeddings_and_summaries_batch_fixed.pkl"
print(f"Salvataggio dei risultati in '{output_file_path}'...")
try:
    with open(output_file_path, "wb") as f:
        pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Salvataggio completato con successo.")
except Exception as e:
    print(f"Errore durante il salvataggio del file: {e}")
