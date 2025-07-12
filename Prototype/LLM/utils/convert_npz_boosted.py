import numpy as np
import argparse
import pickle
import sys

def convert_to_npz(data, output_filename, id_key="user_ids", embedding_key="embeddings", data_field="embedding"):
    """
    Estrae ID ed embedding da un dizionario di dati e li salva
    in un file .npz compresso per la massima efficienza di spazio.
    
    Args:
        data (dict): I dati da convertire (dizionario caricato da pickle)
        output_filename (str): Il nome del file .npz di output
        id_key (str): Nome della chiave per gli ID nel file .npz (default: "user_ids")
        embedding_key (str): Nome della chiave per gli embeddings nel file .npz (default: "embeddings")
        data_field (str): Nome del campo embedding nei dati di input (default: "embedding")
    """
    try:
        print(f"File caricato. Trovati {len(data)} elementi.")
        
        # Prepara due liste per mantenere l'ordine corrispondente
        ids = []
        embeddings_list = []
        
        for item_id, item_info in data.items():
            if data_field in item_info:
                ids.append(item_id)
                embeddings_list.append(item_info[data_field])
        
        if not ids:
            print(f"Errore: Nessun elemento con campo '{data_field}' trovato nei dati.")
            return False
        
        # Converti le liste in array NumPy
        # Gli embedding diventano una matrice 2D [N_elementi x Dim_embedding]
        # Gli ID diventano un array 1D
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        ids_array = np.array(ids)
        
        # Salva entrambi gli array in un singolo file .npz compresso
        # Usa le chiavi personalizzate specificate dall'utente
        np.savez_compressed(
            output_filename, 
            **{embedding_key: embeddings_matrix, id_key: ids_array}
        )
        
        print(f"Conversione completata. Dati salvati in '{output_filename}.npz'.")
        print(f"Dimensioni della matrice embeddings: {embeddings_matrix.shape}")
        print(f"Numero di elementi processati: {len(ids)}")
        print(f"Chiavi utilizzate: ID='{id_key}', Embeddings='{embedding_key}'")
        
        return True
        
    except Exception as e:
        print(f"Si è verificato un errore durante la conversione: {e}")
        return False

def load_pickle_data(input_filename):
    """
    Carica i dati da un file pickle.
    
    Args:
        input_filename (str): Il nome del file .pkl di input
        
    Returns:
        dict: I dati caricati dal file pickle
    """
    try:
        with open(input_filename, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Errore: Il file '{input_filename}' non è stato trovato.")
        return None
    except Exception as e:
        print(f"Errore durante il caricamento del file pickle: {e}")
        return None

def main():
    """Funzione principale con parsing degli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(description="Converte dati pickle in formato NPZ compresso con chiavi personalizzabili")
    
    parser.add_argument(
        "--input_file",
        help="File pickle di input contenente i dati da convertire"
    )
    
    parser.add_argument(
        "--output_file", 
        help="Nome del file NPZ di output (senza estensione .npz)"
    )
    
    parser.add_argument(
        "--id_key",
        default="user_ids",
        help="Nome della chiave per gli ID nel file NPZ (default: 'user_ids')"
    )
    
    parser.add_argument(
        "--embedding-key",
        default="embeddings", 
        help="Nome della chiave per gli embeddings nel file NPZ (default: 'embeddings')"
    )
    
    parser.add_argument(
        "--data-field",
        default="embedding",
        help="Nome del campo embedding nei dati di input (default: 'embedding')"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Abilita output verboso"
    )
    
    args = parser.parse_args()
    
    # Carica i dati dal file pickle
    if args.verbose:
        print(f"Caricamento dati da '{args.input_file}'...")
    
    data = load_pickle_data(args.input_file)
    if data is None:
        sys.exit(1)
    
    # Converti i dati in formato NPZ
    if args.verbose:
        print(f"Conversione in corso...")
        print(f"Chiavi utilizzate: ID='{args.id_key}', Embeddings='{args.embedding_key}'")
        print(f"Campo dati cercato: '{args.data_field}'")
    
    success = convert_to_npz(
        data=data,
        output_filename=args.output_file,
        id_key=args.id_key,
        embedding_key=args.embedding_key,
        data_field=args.data_field
    )
    
    if success:
        print("Operazione completata con successo!")
        sys.exit(0)
    else:
        print("Operazione fallita.")
        sys.exit(1)

if __name__ == "__main__":
    main()