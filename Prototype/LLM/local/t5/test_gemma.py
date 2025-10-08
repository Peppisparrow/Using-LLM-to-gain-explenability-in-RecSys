import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from datasets import Dataset

def test_finetuned_model(model_path, eval_dataframe, num_examples=10):
    """
    Carica un modello T5 addestrato e lo testa su un numero specifico di esempi per ogni task.

    Args:
        model_path (str): Il percorso alla directory del modello salvato.
        eval_dataframe (pd.DataFrame): Il DataFrame contenente i dati di valutazione.
        num_examples (int): Il numero di esempi da testare per ogni task.
    """
    print(f"📂 Caricamento del modello e tokenizer da '{model_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" # Lasciamo che accelerate gestisca il posizionamento su GPU
        )
    except OSError:
        print(f"❌ Errore: Modello non trovato in '{model_path}'.")
        print("Assicurati che il percorso sia corretto e che il training sia stato completato.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Imposta il modello in modalità di valutazione

    print("\n" + "="*50)
    print("🧪 Inizio testing del modello addestrato...")
    print("="*50)

    # Definiamo i prefissi per identificare ogni task
    task_prefixes = {
        "Predict Title": "Your task is to predict the title of the movie",
        "Predict Genres": "Predict only the genres of the film",
        "Complete Plot": "Continue the plot of the film,"
    }

    with torch.no_grad(): # Disabilita il calcolo dei gradienti per l'inferenza
        for task_name, prefix in task_prefixes.items():
            print(f"\n--- TASK: {task_name.upper()} ---\n")
            
            # Filtra il dataframe per la task corrente
            task_df = eval_dataframe[eval_dataframe['input_text'].str.contains(prefix, regex=False)]
            
            if len(task_df) == 0:
                print("Nessun esempio trovato per questa task.")
                continue

            # Seleziona N esempi casuali
            sample_df = task_df.sample(n=min(num_examples, len(task_df)), random_state=42)

            for i, row in enumerate(sample_df.iterrows()):
                index, data = row
                input_text = data['input_text']
                expected_output = data['target_text']

                # Tokenizza l'input e spostalo sul dispositivo corretto
                inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)

                # Genera l'output dal modello
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,       # Lunghezza massima per l'output
                    num_beams=5,          # Usa beam search per risultati migliori
                    early_stopping=True
                )

                # Decodifica la predizione
                input_length = inputs.input_ids.shape[1] 
                generated_tokens = outputs[0][input_length:]

                # 3. Decodifica solo la nuova predizione
                prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                print(f"▶️  ESEMPIO #{i+1}")
                print(f"   INPUT: \n{input_text}\n")
                print(f"   ✅ ATTESO   : {expected_output}")
                print(f"   🤖 PREDETTO : {prediction}")
                print("-" * 40)


# --- Sezione Principale ---
if __name__ == '__main__':
    # Percorso dove hai salvato il modello finale
    final_model_path = "Dataset/ml/ml-latest-small/tuning/gemmasft/onlu_title_lora/final"

    # Ricostruiamo l'eval_df nel caso questo script venga eseguito separatamente
    print("🔄 Ricostruzione del dataset di valutazione...")
    datapath = 'Dataset/ml_small/tuning'
    app_plots_path = f'{datapath}/app_plots.csv'
    df_plots = pd.read_csv(app_plots_path)
    df_plots.dropna(subset=['title', 'genres', 'plot'], inplace=True)
    df_plots['title'] = df_plots['title'].str.strip()
    df_plots['genres'] = df_plots['genres'].str.strip()
    df_plots['plot'] = df_plots['plot'].str.strip()
    
    sft_data = []
    for _, row in df_plots.iterrows():
        title = row['title']
        genres = row['genres']
        plot = row['plot']
        prompt1 = f"You are an expert in movie recommendations.\nYour task is to predict the title of the movie.\n Given the following information:\nGenres: {genres}\nPlot: {plot}\nPlease provide the title..."
        sft_data.append({'input_text': prompt1, 'target_text': title})
        # sft_data.append({'input_text': f"You are an expert in movie recommendations. Predict only the genres of the film, given the following context. Answer only with the genres separated by | (ex. genres1|genres2|genres3).\n"
        #                                 f"###Context\nTitle: {title}\nPlot: {plot}\nWhat are the genres of the film? Provide only the genres.", 'target_text': genres})
        # if len(plot.split()) > 20:
        #     plot_words = plot.split()
        #     mid_point = len(plot_words) // 2
        #     first_half = " ".join(plot_words[:mid_point])
        #     second_half = " ".join(plot_words[mid_point:])
        #     sft_data.append({'input_text': f"You are an expert in movie recommendations. Continue the plot of the film, given the following context.\n"
        #                                     f"###Context\nTitle: {title}\nGenre: {genres}\nPlot: {first_half}...\n", 'target_text': second_half})

    df_sft = pd.DataFrame(sft_data)
    _, eval_df = train_test_split(df_sft, test_size=0.1, random_state=42)
    print("✅ Dataset di valutazione pronto.")
    
    # Esegui la funzione di test
    test_finetuned_model(model_path=final_model_path, eval_dataframe=eval_df, num_examples=3)