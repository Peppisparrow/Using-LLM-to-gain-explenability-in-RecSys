import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset


import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# --- 1. Caricamento e Pulizia dei Dati ---
datapath = 'Dataset/ml/ml-latest-small/tuning'
app_plots_path = f'{datapath}/app_plots.csv'
output_dir = "Dataset/ml/ml-latest-small/tuning/t5/first_experiments"
print("📂 Caricamento del dataset dei film...")

try:
    df_plots = pd.read_csv(app_plots_path)
except FileNotFoundError:
    print("❌ Errore: 'app_plots.csv' non trovato. Assicurati che il file sia nella directory corretta.")
    exit()

# Rimuoviamo i film senza informazioni cruciali
df_plots.dropna(subset=['title', 'genres', 'plot'], inplace=True)

# Pulizia di base: rimuoviamo spazi extra
df_plots['title'] = df_plots['title'].str.strip()
df_plots['genres'] = df_plots['genres'].str.strip()
df_plots['plot'] = df_plots['plot'].str.strip()

print(f"✅ Dati caricati e puliti. Numero di film validi: {len(df_plots)}")

# --- 2. Creazione dei Dati per le Task di SFT ---
sft_data = []

print("⚙️  Creazione degli esempi per le tre task di SFT...")
for _, row in df_plots.iterrows():
    title = row['title']
    genres = row['genres']
    plot = row['plot']

    # Task 1: Predire il titolo da genere e trama
    sft_data.append({
        'input_text': f"You are an expert in movie recommendations. Given the following information:\nGenres: {genres}\nPlot: {plot}\nYour task is to predict the title of the movie.",
        'target_text': title
    })

    # Task 2: Predire il genere da titolo e trama
    sft_data.append({
        'input_text': f"You are an expert in movie recommendations. Given the following information:\nTitle: {title}\nPlot: {plot}\nYour task is to predict the genres of the movie.",
        'target_text': genres
    })

    # Task 3: Completare la trama
    # Dividiamo la trama a metà per creare un compito di completamento
    if len(plot.split()) > 20: # Assicuriamoci che la trama sia abbastanza lunga
        plot_words = plot.split()
        mid_point = len(plot_words) // 2
        first_half = " ".join(plot_words[:mid_point])
        second_half = " ".join(plot_words[mid_point:])
        
        sft_data.append({
            'input_text': f"You are an expert in movie recommendations. Given the following information:\nTitle: {title}\nGenre: {genres}\nPlot: {first_half}\nYour task is to complete the plot.",
            'target_text': second_half
        })

# Creiamo un DataFrame e poi un Dataset di Hugging Face
df_sft = pd.DataFrame(sft_data)
train_df, eval_df = train_test_split(df_sft, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

print(f"✅ Creati {len(df_sft)} esempi di training, divisi in {len(train_dataset)} per il training e {len(eval_dataset)} per la validazione.")





# --- 3. Setup del Modello e del Tokenizer ---

model_name = 't5-large' # Puoi usare 't5-base' per un modello più grande e performante
print(f"🔄 Caricamento del modello '{model_name}' e del tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Verifichiamo se è disponibile una GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"⚡️ Il modello verrà addestrato su: {device}")

# --- 4. Tokenizzazione dei Dati ---
def tokenize_function(examples):
    # Tokenizza gli input
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True)
    # Tokenizza i target (le etichette) usando l'argomento `text_target`
    labels = tokenizer(text_target=examples['target_text'], max_length=512, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

print(" tokenize dei dataset...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,              # Numero di epoche (3 è un buon punto di partenza)
    per_device_train_batch_size=4,   # Riduci se hai problemi di memoria VRAM
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f'{output_dir}/logs',
    logging_steps=100,
    eval_strategy="epoch",     # Valuta alla fine di ogni epoca
    save_strategy="epoch",           # Salva il modello alla fine di ogni epoca
    load_best_model_at_end=True,
    fp16=True if device == 'cuda' else False, # Abilita il mixed precision training su GPU
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
print("\n🚀 Inizio del fine-tuning del modello T5...")
trainer.train()
print("✨ Training completato!")

# --- 7. Salvataggio del Modello Finale ---
final_model_path = f"{output_dir}/final"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"✅ Modello e tokenizer salvati in '{final_model_path}'")
