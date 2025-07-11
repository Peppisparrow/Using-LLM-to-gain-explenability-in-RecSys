import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import polars as pl
from tqdm import tqdm
import json
# 1. Caricamento del Modello e del Tokenizer
# Usiamo un modello della famiglia Mistral.
# "Instruct" significa che è stato ottimizzato per seguire istruzioni.
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # Tipo di dato per i calcoli interni
    bnb_4bit_quant_type="nf4"            # Tipo di quantizzazione (standard)
)
# NOTA SULLA MEMORIA: Mistral-7B è un modello grande (~14GB in half-precision).
# Se hai problemi di memoria, puoi caricarlo in 4-bit usando bitsandbytes:
# model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
# Altrimenti, caricalo in half-precision (bfloat16) se hai una GPU compatibile.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",           # Distribuisce automaticamente il modello su GPU/CPU
    attn_implementation="sdpa"  # Usa l'implementazione SdpAttn per efficienza
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 2. Preparazione del Prompt di Input
# I modelli "Instruct" funzionano meglio con un template specifico.
# Per Mistral, il template è [INST] ... [/INST].
#input_prompt = generate_user_taste_prompt(10127955)
data_dir = "Dataset/steam/descr_filtering"
user = pl.read_csv(f"{data_dir}/users.csv")
user_ids = user["user_id"].to_list()
with open(f'{data_dir}/user_prompts.json', 'r') as f:
        user_taste_prompts = json.load(f)

output_dict = {}

for user_id in tqdm(user_ids, desc="Analizzando i profili utente"):
    prompt = user_taste_prompts.get(str(user_id))
    if not prompt:
        continue

    formatted_prompt = f"[INST] {prompt} [/INST]"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        model_outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id, 
            output_hidden_states=True,
            return_dict_in_generate=True
        )

    generated_ids = model_outputs.sequences[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    last_hidden_state = model_outputs.hidden_states[0][-1]
    prompt_hidden_states = last_hidden_state[:, :input_len, :]
    embedding_mean_pooling = prompt_hidden_states.mean(dim=1)

    output_dict[str(user_id)] = {
        "input_prompt": prompt,
        "generated_text": generated_text.strip(),
        "embedding": embedding_mean_pooling.cpu().tolist()[0]
    }

# Save as pickle with highest protocol
import pickle
with open("user_embeddings.pkl", "wb") as f:
    pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)




# input_prompt = user_taste_prompts[str(6766810)]
# formatted_prompt = f"[INST] {input_prompt} [/INST]"

# input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
# input_len = input_ids.shape[1]

# output_sequences = model.generate(
#     input_ids=input_ids,
#     max_new_tokens=200,
#     pad_token_id=tokenizer.eos_token_id # Imposta il pad token per evitare warning
# )


# generated_ids = output_sequences[0, input_ids.shape[-1]:]
# generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

# with torch.no_grad():
#     model_output = model(
#         input_ids=input_ids,
#         output_hidden_states=True # Parametro FONDAMENTALE
#     )
# all_hidden_states = model_output.hidden_states
# last_hidden_state = all_hidden_states[-1]
# embedding_mean_pooling = last_hidden_state.mean(dim=1)
# output_dict['user_id'] = {
#     "input_prompt": input_prompt,
#     "generated_text": generated_text.strip(),
#     "embedding_mean_pooling": embedding_mean_pooling # Converti in lista per serializzazione
# }
# print(input_prompt)
# print(generated_text.strip())
# print(embedding_mean_pooling)
