import pandas as pd
import numpy as np
import time
from tqdm import tqdm
MODEL = 'mxbai'  # Modello di embedding da utilizzare
DATA= 'tuning/'  # Tipo di dati, può essere 'ml' o 'steam'
def calculate_user_embeddings():
    """
    Calcola gli embedding degli utenti come media degli embedding dei giochi con cui hanno
    interagito, utilizzando operazioni vettorizzate per massima efficienza e mostrando
    una barra di avanzamento durante il calcolo.

    Input:
    - embeddings.npz: File con ['app_id'] e ['embeddings'] dei giochi.
    - train_recommendations.csv: File con ['user_id', 'app_id'] delle interazioni.

    Output:
    - game_user_embeddings.npz: File con ['user_id'] e ['embeddings'] degli utenti.
    """
    print("🚀 Inizio del processo di calcolo...")
    start_time = time.time()

    ## 1. Caricamento Dati
    print(" Fase 1/4: Caricamento dei file...")
    
    # Usa il nome corretto del tuo file di embedding
    with np.load(f'{DATA}item_embeddings_{MODEL}.npz') as data:
        embeddings_df = pd.DataFrame({
            'app_id': data['item_id'],
            'embedding': list(data['embeddings'])
        })
    # salva data come npz con files item_id invece di app_id
        
    
    embeddings_df['app_id'] = embeddings_df['app_id'].astype(np.uint32)

    interactions_df = pd.read_csv(f'{DATA}train_recommendations.csv', dtype={'user_id': np.uint32, 'app_id': np.uint32})
    
    print(f"   - Trovati {len(embeddings_df)} embedding di giochi.")
    print(f"   - Trovate {len(interactions_df)} interazioni utente-gioco.")

    ## 2. Unione dei Dati
    print(" Fase 2/4: Associazione degli embedding alle interazioni...")
    
    merged_df = pd.merge(interactions_df, embeddings_df, on='app_id', how='inner')
    
    print(f"   - Associazione completata. {len(merged_df)} interazioni valide con embedding.")

    ## 3. Calcolo della Media per Utente con Barra di Avanzamento
    print(" Fase 3/4: Calcolo dell'embedding medio per ogni utente...")

    # Inizializza tqdm per l'integrazione con pandas
    tqdm.pandas(desc="Calcolo embedding utenti")

    # Usa .progress_apply() invece di .apply() per visualizzare la barra
    user_embeddings_series = merged_df.groupby('user_id')['embedding'].progress_apply(
        lambda x: np.mean(np.stack(x), axis=0)
    )
    
    print(f"   - Calcolati gli embedding per {len(user_embeddings_series)} utenti unici.")

    ## 4. Salvataggio dei Risultati
    print(" Fase 4/4: Salvataggio dei risultati nel file NPZ...")
    
    user_ids = user_embeddings_series.index.to_numpy()
    final_embeddings = np.stack(user_embeddings_series.values)

    np.savez_compressed(
        f'{DATA}mean_user_embeddings_{MODEL}.npz',
        user_id=user_ids,
        embeddings=final_embeddings
    )

    end_time = time.time()
    print("\n✅ Operazione completata con successo!")
    print(f"   - File salvato: 'game_user_embeddings.npz'")
    print(f"   - Shape degli embedding finali: {final_embeddings.shape}")
    print(f"   - Tempo totale di esecuzione: {end_time - start_time:.2f} secondi.")

# Esegui la funzione
if __name__ == "__main__":
    calculate_user_embeddings()