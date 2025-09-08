import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
# <-- MODIFICA: Importazioni aggiornate
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer # <-- MODIFICA: Useremo SFTTrainer per semplicità

# --- 1. Caricamento e Pulizia dei Dati ---
datapath = 'Dataset/ml/ml-latest-small/tuning'
app_plots_path = f'{datapath}/app_plots.csv'
# <-- MODIFICA: Nuovo output_dir per Gemma
output_dir = "Dataset/ml/ml-latest-small/tuning/gemma2b/first_experiments_LORA"
print("📂 Caricamento del dataset dei film...")

try:
    df_plots = pd.read_csv(app_plots_path)
except FileNotFoundError:
    print(f"❌ Errore: '{app_plots_path}' non trovato.")
    exit()

df_plots.dropna(subset=['title', 'genres', 'plot'], inplace=True)
df_plots['title'] = df_plots['title'].str.strip()
df_plots['genres'] = df_plots['genres'].str.strip()
df_plots['plot'] = df_plots['plot'].str.strip()
print(f"✅ Dati caricati e puliti. Numero di film validi: {len(df_plots)}")

# --- 2. Creazione dei Dati per le Task di SFT ---
# <-- MODIFICA: Formattiamo i dati per un Chat Template
sft_data = []
print("⚙️  Creazione degli esempi per le tre task di SFT nel formato chat...")
for _, row in df_plots.iterrows():
    title = row['title']
    genres = row['genres']
    plot = row['plot']

    # Task 1: Predire il titolo
    prompt1 = f"You are an expert in movie recommendations. Given the following information:\nGenres: {genres}\nPlot: {plot}\nYour task is to predict the title of the movie."
    sft_data.append({
        "messages": [
            {"role": "user", "content": prompt1},
            {"role": "assistant", "content": title}
        ]
    })

    # Task 2: Predire il genere
    prompt2 = f"You are an expert in movie recommendations. Given the following information:\nTitle: {title}\nPlot: {plot}\nYour task is to predict the genres of the movie."
    sft_data.append({
        "messages": [
            {"role": "user", "content": prompt2},
            {"role": "assistant", "content": genres}
        ]
    })

    # Task 3: Completare la trama
    if len(plot.split()) > 20:
        plot_words = plot.split()
        mid_point = len(plot_words) // 2
        first_half = " ".join(plot_words[:mid_point])
        second_half = " ".join(plot_words[mid_point:])
        prompt3 = f"You are an expert in movie recommendations. Given the following information:\nTitle: {title}\nGenre: {genres}\nPlot: {first_half}\nYour task is to complete the plot."
        sft_data.append({
            "messages": [
                {"role": "user", "content": prompt3},
                {"role": "assistant", "content": second_half}
            ]
        })

# Creiamo un DataFrame e poi un Dataset di Hugging Face
df_sft = pd.DataFrame(sft_data)
# Mescoliamo il dataframe prima dello split per assicurare una distribuzione casuale
df_sft = df_sft.sample(frac=1, random_state=42).reset_index(drop=True)

# Split manuale dei dati
split_index = int(len(df_sft) * 0.9)
train_df = df_sft.iloc[:split_index]
eval_df = df_sft.iloc[split_index:]

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

print(f"✅ Creati {len(df_sft)} esempi, divisi in {len(train_dataset)} (train) e {len(eval_dataset)} (eval).")

# --- 3. Setup del Modello e del Tokenizer ---
# <-- MODIFICA: Carichiamo Gemma-2
model_name = 'google/gemma-2-2b-it'
print(f"🔄 Caricamento del modello '{model_name}' e del tokenizer...")

# Usiamo bfloat16 per efficienza e compatibilità con le GPU moderne
torch_dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map="auto" # Lasciamo che accelerate gestisca il posizionamento su GPU
)

# --- NUOVA SEZIONE: Configurazione di PEFT (LoRA) per Gemma-2 ---
print("⚙️ Configurazione di LoRA per Gemma-2...")

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    # <-- MODIFICA: Target modules specifici per Gemma-2
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    # <-- MODIFICA: Task type per modelli decoder-only
    task_type="CAUSAL_LM"
)

# Applica la configurazione LoRA al modello
model = get_peft_model(model, lora_config)
print("✅ Modello configurato con adattatori LoRA.")
model.print_trainable_parameters()

# --- 5. Configurazione del Training con SFTTrainer ---
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=500,
    learning_rate=2e-5, # <-- MODIFICA: Un learning rate più basso è spesso migliore per SFT
    weight_decay=0.01,
    logging_dir=f'{output_dir}/logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none", # Disabilita wandb/tensorboard se non configurati
    bf16=True # Abilita bfloat16 per il training
)

# <-- MODIFICA: Usiamo SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config
)

# --- 6. Avvio del Training ---
print("\n🚀 Inizio del fine-tuning del modello Gemma-2 con LoRA...")
trainer.train()
print("✨ Training completato!")

# --- 7. Salvataggio del Modello Finale ---
final_model_path = f"{output_dir}/final"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"✅ Adattatori LoRA e tokenizer salvati in '{final_model_path}'")