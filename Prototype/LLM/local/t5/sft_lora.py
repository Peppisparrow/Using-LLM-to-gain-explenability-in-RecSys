import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

# --- 1. Caricamento e Pulizia dei Dati ---
datapath = 'Dataset/ml/ml-latest-small/tuning'
app_plots_path = f'{datapath}/app_plots.csv'
output_dir = "Dataset/ml/ml-latest-small/tuning/t5/first_experiments_LORA" # <-- MODIFICA: Nuovo output_dir
print("📂 Caricamento del dataset dei film...")

try:
    df_plots = pd.read_csv(app_plots_path)
except FileNotFoundError:
    print("❌ Errore: 'app_plots.csv' non trovato. Assicurati che il file sia nella directory corretta.")
    exit()

df_plots.dropna(subset=['title', 'genres', 'plot'], inplace=True)
df_plots['title'] = df_plots['title'].str.strip()
df_plots['genres'] = df_plots['genres'].str.strip()
df_plots['plot'] = df_plots['plot'].str.strip()
print(f"✅ Dati caricati e puliti. Numero di film validi: {len(df_plots)}")

# --- 2. Creazione dei Dati per le Task di SFT ---
# (Questa sezione rimane identica)
sft_data = []
print("⚙️  Creazione degli esempi per le tre task di SFT...")
for _, row in df_plots.iterrows():
    title = row['title']
    genres = row['genres']
    plot = row['plot']
    sft_data.append({'input_text': f"You are an expert in movie recommendations. Given the following information:\nGenres: {genres}\nPlot: {plot}\nYour task is to predict the title of the movie.",'target_text': title})
    sft_data.append({'input_text': f"You are an expert in movie recommendations. Given the following information:\nTitle: {title}\nPlot: {plot}\nYour task is to predict the genres of the movie.",'target_text': genres})
    if len(plot.split()) > 20:
        plot_words = plot.split()
        mid_point = len(plot_words) // 2
        first_half = " ".join(plot_words[:mid_point])
        second_half = " ".join(plot_words[mid_point:])
        sft_data.append({'input_text': f"You are an expert in movie recommendations. Given the following information:\nTitle: {title}\nGenre: {genres}\nPlot: {first_half}\nYour task is to complete the plot.",'target_text': second_half})

df_sft = pd.DataFrame(sft_data)
train_df, eval_df = train_test_split(df_sft, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
print(f"✅ Creati {len(df_sft)} esempi di training, divisi in {len(train_dataset)} per il training e {len(eval_dataset)} per la validazione.")

# --- 3. Setup del Modello e del Tokenizer ---
model_name = 't5-large'
print(f"🔄 Caricamento del modello '{model_name}' e del tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# --- NUOVA SEZIONE: Configurazione di PEFT (LoRA) ---
print("⚙️ Configurazione di LoRA...")

# Definiamo la configurazione di LoRA
lora_config = LoraConfig(
    r=32,  # Rank della decomposizione. Valori comuni sono 8, 16, 32. Più alto è, più parametri addestrabili.
    lora_alpha=64,  # Parametro di scaling. Spesso è il doppio di `r`.
    target_modules=["q", "v"],  # Applica LoRA ai moduli di query e value nell'attention.
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM  # Fondamentale per modelli encoder-decoder come T5
)

# Applica la configurazione LoRA al modello
model = get_peft_model(model, lora_config)
print("✅ Modello configurato con adattatori LoRA.")

# Funzione helper per mostrare quanti parametri stiamo addestrando
def print_trainable_parameters(model):
    """
    Stampa il numero di parametri addestrabili nel modello.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )

print_trainable_parameters(model) # Vedrai che i parametri addestrabili sono una frazione piccolissima!

# Verifichiamo se è disponibile una GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device) # Non è più necessario con PEFT e Accelerate, il Trainer lo gestisce
print(f"⚡️ Il modello verrà addestrato su: {device}")

# --- 4. Tokenizzazione dei Dati ---
# (Questa sezione rimane identica)
def tokenize_function(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True)
    labels = tokenizer(text_target=examples['target_text'], max_length=256, truncation=True) # Leggermente ottimizzato
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

print(" tokenize dei dataset...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])

# --- 5. Configurazione del Training ---
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=2,  # <-- MODIFICA: Con LoRA, possiamo spesso usare un batch size più grande
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f'{output_dir}/logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True if device == 'cuda' else False,
)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator
)

# --- 6. Avvio del Training ---
print("\n🚀 Inizio del fine-tuning del modello T5 con LoRA...")
trainer.train()
print("✨ Training completato!")

# --- 7. Salvataggio del Modello Finale ---
# Il trainer di PEFT salva automaticamente solo gli adattatori, che è ciò che vogliamo
final_model_path = f"{output_dir}/final"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"✅ Adattatori LoRA e tokenizer salvati in '{final_model_path}'")