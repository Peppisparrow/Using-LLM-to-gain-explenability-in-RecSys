import pandas as pd
import numpy as np
import scipy.sparse as sps
from pathlib import Path

class DataManger:
    def __init__(self, user_embedding_path: Path, data_path: Path, item_embeddings_path: Path = None):
        (self.URM_train, self.URM_test, 
         self.user_embeddings, self.item_embeddings) = self.load_data(user_embedding_path, data_path, item_embeddings_path)
        
    def get_URM_train(self): return self.URM_train
    def get_URM_test(self): return self.URM_test
    def get_user_embeddings(self): return self.user_embeddings
    def get_item_embeddings(self): return self.item_embeddings
    def get_user_mapping(self): return self.user_id_to_index
    def get_item_mapping(self): return self.item_id_to_index

    def load_data(self, user_embeddings_path: Path, data_path: Path, item_embeddings_path: Path = None):
        train_path = data_path / 'train_recommendations.csv'
        test_path = data_path / 'test_recommendations.csv'

        train_data = pd.read_csv(train_path)[['user_id', 'app_id']]
        test_data = pd.read_csv(test_path)[['user_id', 'app_id']]
        
        # LOGICA CORRETTA: L'universo di utenti e item viene definito dai file CSV.
        unique_user_ids = np.sort(np.unique(np.concatenate((train_data['user_id'].values, test_data['user_id'].values))))
        unique_item_ids = np.sort(np.unique(np.concatenate((train_data['app_id'].values, test_data['app_id'].values))))
        
        # --- VALIDAZIONE UTENTI ---
        x_user = np.load(user_embeddings_path)
        all_user_vectors = x_user['embeddings']
        embedding_user_ids = np.array([int(i) for i in x_user['user_ids']])
        
        required_user_set = set(unique_user_ids)
        provided_user_set = set(embedding_user_ids)

        if not required_user_set.issubset(provided_user_set):
            missing_users = required_user_set - provided_user_set
            raise ValueError(f"CRITICAL ERROR: Missing embeddings for {len(missing_users)} users. Examples: {list(missing_users)[:10]}")
        
        user_embedding_map = {id_val: vec for id_val, vec in zip(embedding_user_ids, all_user_vectors)}
        user_embeddings = np.array([user_embedding_map[id_val] for id_val in unique_user_ids])

        # --- VALIDAZIONE ITEM ---
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

        # Costruzione mappe e URM
        self.user_id_to_index = {uid: i for i, uid in enumerate(unique_user_ids)}
        self.item_id_to_index = {iid: i for i, iid in enumerate(unique_item_ids)}
        
        train_data['user_id'] = train_data['user_id'].map(self.user_id_to_index)
        train_data['app_id'] = train_data['app_id'].map(self.item_id_to_index)
        
        train_data.dropna(inplace=True)
        train_data['user_id'] = train_data['user_id'].astype(int)
        train_data['app_id'] = train_data['app_id'].astype(int)

        test_data['user_id'] = test_data['user_id'].map(self.user_id_to_index)
        test_data['app_id'] = test_data['app_id'].map(self.item_id_to_index)
        test_data.dropna(inplace=True)
        test_data['user_id'] = test_data['user_id'].astype(int)
        test_data['app_id'] = test_data['app_id'].astype(int)
        
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
        
        return URM_train, URM_test, user_embeddings, item_embeddings