
import os
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sps
from pathlib import Path

class DataManger:
    def __init__(self, user_embedding_path: Path, data_path: Path = Path('Prototype/data')):
        self.data_path = data_path
        self.URM_train, self.URM_test, self.user_embeddings = self.load_data(user_embedding_path, data_path)
        
    def get_URM_train(self):
        """
        Returns the user-item interaction matrix for training.
        """
        return self.URM_train
    
    def get_URM_test(self):
        """
        Returns the user-item interaction matrix for testing.
        """
        return self.URM_test
    
    def get_user_embeddings(self):
        """
        Returns the user embeddings.
        """
        return self.user_embeddings

    def load_data(self,
                  user_embeddings_path: Path,
                  data_path: Path):
        """
        Loads URM_train, URM_test and user embeddings
        """

        train_path = data_path / 'train_recommendations.csv'
        test_path = data_path / 'test_recommendations.csv'


        train_data = pd.read_csv(train_path)[['user_id', 'app_id']]
        test_data = pd.read_csv(test_path)[['user_id', 'app_id']]

        x = np.load(user_embeddings_path)
        user_embeddings = x['embeddings']
        user_ids = x['user_ids']

        # Since user_ids are stored as strings, we need to convert them to integers
        user_ids = [int(i) for i in user_ids]
        # Convert user_ids to numpy array to use with sorted_indices
        user_ids = np.array(user_ids)
        # Now we sort it to ensure consistency
        sorted_indices = np.argsort(user_ids)
        user_embeddings = user_embeddings[sorted_indices]
        user_ids = user_ids[sorted_indices]


        unique_user_ids = np.unique(np.concatenate((train_data['user_id'].values, test_data['user_id'].values)))
        unique_item_ids = np.unique(np.concatenate((train_data['app_id'].values, test_data['app_id'].values)))

        unique_user_ids = np.array(sorted(unique_user_ids))
        unique_item_ids = np.array(sorted(unique_item_ids))

        # Mapping user_ids and review_ids to indices
        user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
        item_id_to_index = {review_id: index for index, review_id in enumerate(unique_item_ids)}

        train_data['user_id'] = train_data['user_id'].map(user_id_to_index)
        train_data['review_id'] = train_data['app_id'].map(item_id_to_index)

        test_data['user_id'] = test_data['user_id'].map(user_id_to_index)
        test_data['review_id'] = test_data['app_id'].map(item_id_to_index)

        train_data['interaction'] = 1
        test_data['interaction'] = 1

        n_users = len(user_ids)
        n_items = len(unique_item_ids)

        URM_train = sps.coo_matrix((train_data['interaction'].values, 
                                (train_data['user_id'].values, train_data['review_id'].values)),
                                shape=(n_users, n_items))

        URM_test = sps.coo_matrix((test_data['interaction'].values, 
                                (test_data['user_id'].values, test_data['review_id'].values)),
                                shape=(n_users, n_items))

        URM_train = URM_train.tocsr()
        URM_test = URM_test.tocsr()

        print(f"URM train shape: {URM_train.shape}, nonzero: {URM_train.nnz}")
        print(f"URM test shape: {URM_test.shape}, nonzero: {URM_test.nnz}")
        
        return URM_train, URM_test, user_embeddings