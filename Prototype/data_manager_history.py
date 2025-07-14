import pandas as pd
import numpy as np
import scipy.sparse as sps
from pathlib import Path
import pickle # MODIFICA 1: Aggiunto l'import di pickle

class DataManger:
    # MODIFICA 2: Aggiunto user_history_path per il file .pkl
    def __init__(self, data_path: Path, user_embedding_path: Path = None, item_embeddings_path: Path = None, user_history_path: Path = None):
        (self.URM_train, self.URM_test, 
         self.user_embeddings, self.item_embeddings,
         self.user_history_embeddings) = self.load_data(data_path, user_embedding_path, item_embeddings_path, user_history_path)
        
    def get_URM_train(self): return self.URM_train
    def get_URM_test(self): return self.URM_test
    def get_user_embeddings(self): return self.user_embeddings
    def get_item_embeddings(self): return self.item_embeddings
    def get_user_mapping(self): return self.user_id_to_index
    def get_item_mapping(self): return self.item_id_to_index
    def get_user_history_embeddings(self): return self.user_history_embeddings
    

    # MODIFICA 4: Aggiunto user_history_path alla firma del metodo
    def load_data(self, data_path: Path, user_embeddings_path: Path = None, item_embeddings_path: Path = None, user_history_path: Path = None):
        train_path = data_path / 'train_recommendations.csv'
        test_path = data_path / 'test_recommendations.csv'

        train_data = pd.read_csv(train_path)[['user_id', 'app_id']]
        test_data = pd.read_csv(test_path)[['user_id', 'app_id']]
        
        unique_user_ids = np.sort(np.unique(np.concatenate((train_data['user_id'].values, test_data['user_id'].values))))
        unique_item_ids = np.sort(np.unique(np.concatenate((train_data['app_id'].values, test_data['app_id'].values))))
        
        # La costruzione delle mappe ora avviene prima, per essere disponibile a tutte le sezioni
        self.user_id_to_index = {uid: i for i, uid in enumerate(unique_user_ids)}
        self.item_id_to_index = {iid: i for i, iid in enumerate(unique_item_ids)}
        n_users = len(unique_user_ids)
        n_items = len(unique_item_ids)

        # Logica per user_embeddings (aggregati) - INVARIATA
        user_embeddings = None
        user_embeddings = None
        if user_embeddings_path:
            x_user = np.load(user_embeddings_path)
            all_user_vectors = x_user['embeddings']
            embedding_user_ids = np.array([int(i) for i in x_user['user_id']])
            
            required_user_set = set(unique_user_ids)
            provided_user_set = set(embedding_user_ids)

            if not required_user_set.issubset(provided_user_set):
                missing_users = required_user_set - provided_user_set
                raise ValueError(f"CRITICAL ERROR: Missing embeddings for {len(missing_users)} users. Examples: {list(missing_users)[:10]}")
            
            user_embedding_map = {id_val: vec for id_val, vec in zip(embedding_user_ids, all_user_vectors)}
            user_embeddings = np.array([user_embedding_map[id_val] for id_val in unique_user_ids])


        # Logica per item_embeddings - INVARIATA
        item_embeddings = None
        if item_embeddings_path:
            x_item = np.load(item_embeddings_path)
            all_item_vectors = x_item['embeddings']
            embedding_item_ids = np.array([int(i) for i in x_item['app_id']])

            required_item_set = set(unique_item_ids)
            provided_item_set = set(embedding_item_ids)

            if not required_item_set.issubset(provided_item_set):
                missing_items = required_item_set - provided_item_set
                raise ValueError(f"CRITICAL ERROR: Missing embeddings for {len(missing_items)} items. Examples: {list(missing_items)[:10]}")
            
            item_embedding_map = {id_val: vec for id_val, vec in zip(embedding_item_ids, all_item_vectors)}
            item_embeddings = np.array([item_embedding_map[id_val] for id_val in unique_item_ids])

        # MODIFICA 5: Aggiunta la logica per caricare le cronologie dal file .pkl
        user_history_embeddings = None
        if user_history_path:
            print(f"INFO: Caricamento delle cronologie utenti dal file .pkl: {user_history_path}")
            with open(user_history_path, 'rb') as f:
                user_histories_dict = pickle.load(f)

            # Prepara una lista per contenere le matrici di embedding, ordinata per il nuovo indice
            # Inizializziamo con None per assicurarci che tutti gli slot siano riempiti
            user_history_embeddings = [None] * n_users
            
            found_users = 0
            for original_user_id, history_matrix in user_histories_dict.items():
                # Controlla se l'utente dal file pkl esiste nel nostro universo di utenti
                if original_user_id in self.user_id_to_index:
                    # Ottieni l'indice mappato corretto
                    mapped_user_index = self.user_id_to_index[original_user_id]
                    # Inserisci la matrice nella posizione corretta della lista
                    user_history_embeddings[mapped_user_index] = history_matrix
                    found_users += 1
            
            print(f"INFO: Associate {found_users}/{n_users} cronologie utenti agli utenti presenti nel dataset.")
            # Controlla se qualche utente non ha avuto una cronologia associata
            if found_users < n_users:
                print(f"WARNING: {n_users - found_users} utenti non avevano una cronologia nel file .pkl.")

       # La costruzione delle mappe e della URM funziona indipendentemente dalla presenza degli embedding
        self.user_id_to_index = {uid: i for i, uid in enumerate(unique_user_ids)}
        self.item_id_to_index = {iid: i for i, iid in enumerate(unique_item_ids)}
        
        train_data['user_id'] = train_data['user_id'].map(self.user_id_to_index)
        train_data['app_id'] = train_data['app_id'].map(self.item_id_to_index)
        
        train_data.dropna(inplace=True)
        train_data = train_data.astype(int)

        test_data['user_id'] = test_data['user_id'].map(self.user_id_to_index)
        test_data['app_id'] = test_data['app_id'].map(self.item_id_to_index)
        test_data.dropna(inplace=True)
        test_data = test_data.astype(int)
        
        train_data['interaction'] = 1
        test_data['interaction'] = 1

        n_users = len(unique_user_ids)
        n_items = len(unique_item_ids)

        URM_train = sps.csr_matrix((train_data['interaction'].values, 
                                (train_data['user_id'].values, train_data['app_id'].values)),
                                shape=(n_users, n_items))
        URM_test = sps.csr_matrix((test_data['interaction'].values, 
                                (test_data['user_id'].values, test_data['app_id'].values)),
                                shape=(n_users, n_items))
        
        # MODIFICA 6: Restituisce la nuova struttura dati
        return URM_train, URM_test, user_embeddings, item_embeddings, user_history_embeddings