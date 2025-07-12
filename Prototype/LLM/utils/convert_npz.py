import numpy as np
def convert_to_npz(data, output_filename):
    """
    Estrae user_id e embedding da un file pickle e li salva
    in un file .npz compresso per la massima efficienza di spazio.

    Args:
        input_filename (str): Il nome del file .pkl di input.
        output_filename (str): Il nome del file .npz di output.
    """
    try:
        print(f"File caricato. Trovati utenti.")

        # Prepara due liste per mantenere l'ordine corrispondente
        user_ids = []
        embeddings_list = []

        for user_id, user_info in data.items():
            if 'embedding' in user_info:
                user_ids.append(user_id)
                embeddings_list.append(user_info['embedding'])

        # Converti le liste in array NumPy
        # Gli embedding diventano una matrice 2D [N_utenti x Dim_embedding]
        # Gli ID utente diventano un array 1D
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        user_ids_array = np.array(user_ids)

        # Salva entrambi gli array in un singolo file .npz compresso
        # 'embeddings' e 'user_ids' saranno le chiavi per recuperare i dati
        np.savez_compressed(output_filename, embeddings=embeddings_matrix, user_ids=user_ids_array)
            
        print(f"Conversione completata. Dati salvati in '{output_filename}.npz'.")
        print(f"Dimensioni della matrice embeddings: {embeddings_matrix.shape}")
    except FileNotFoundError:
        print(f"Errore: Il file  non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")