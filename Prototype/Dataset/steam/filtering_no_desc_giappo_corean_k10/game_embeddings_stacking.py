import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import pickle  # <-- 1. Importa il modulo pickle

MODEL = "t5"
DATA = "small"
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

    # Le Fasi 1 e 2 restano invariate
    ## 1. Caricamento Dati
    print(" Fase 1/4: Caricamento dei file...")
    with np.load(f'{DATA}/game_embeddings/game_embeddings_{MODEL}.npz') as data:
        embeddings_df = pd.DataFrame({
            'app_id': data['app_id'],
            'embedding': list(data['embeddings'])
        })
    embeddings_df['app_id'] = embeddings_df['app_id'].astype(np.uint32)
    interactions_df = pd.read_csv(f'{DATA}/train_recommendations.csv', dtype={'user_id': np.uint32, 'app_id': np.uint32})

    ## 2. Unione dei Dati
    print(" Fase 2/4: Associazione degli embedding alle interazioni...")
    merged_df = pd.merge(interactions_df, embeddings_df, on='app_id', how='inner')

    ## 3. Raggruppamento delle Cronologie
    print(" Fase 3/4: Raggruppamento della cronologia di embedding per ogni utente...")
    tqdm.pandas(desc="Raggruppamento utenti")
    # Questa è la tua logica originale per ottenere la cronologia completa
    user_embeddings_series = merged_df.groupby('user_id')['embedding'].progress_apply(
        lambda x: np.stack(x.values)
    )
    print(f"   - Raggruppate le cronologie per {len(user_embeddings_series)} utenti unici.")

    ## 4. Salvataggio dei Risultati in un Dizionario (.pkl)
    print(" Fase 4/4: Salvataggio dei risultati in un file dizionario (.pkl)...")
    dict_user = user_embeddings_series.to_dict()
    with open( f'{DATA}/game_embeddings/game_embeddings_{MODEL}_MATRIX_USER.pkl', 'wb') as f:
        pickle.dump(dict_user, f)


# Esegui la funzione
if __name__ == "__main__":
    calculate_user_embeddings()