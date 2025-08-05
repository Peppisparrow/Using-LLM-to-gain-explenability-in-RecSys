#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Luca Pagano & Giuseppe Vitello
"""
import numpy as np
import os

# Assuming BaseRecommender is in the specified path.
from RecSysFramework.Recommenders.BaseRecommender import BaseRecommender
from RecSysFramework.Recommenders.DataIO import DataIO


class EmbeddingSimilarityRecommender(BaseRecommender):
    """
    EmbeddingSimilarityRecommender

    This recommender computes recommendations for a given user by calculating the cosine similarity
    between the user's embedding and all item embeddings. It uses numpy for all vector operations.
    """

    RECOMMENDER_NAME = "EmbeddingSimilarityRecommender"

    def __init__(self, URM_train, user_embeddings, item_embeddings, verbose=True):
        """
        Initializes the recommender.

        Args:
            URM_train (scipy.sparse.spmatrix): The User-Rating Matrix for training.
            user_embeddings (np.ndarray): A numpy array of user embeddings.
            item_embeddings (np.ndarray): A numpy array of item embeddings.
            verbose (bool, optional): If True, prints status messages. Defaults to True.
        """
        super(EmbeddingSimilarityRecommender, self).__init__(URM_train, verbose=verbose)

        # --- Validate and store embeddings ---
        assert self.n_users == user_embeddings.shape[0], \
            f"{self.RECOMMENDER_NAME}: User count mismatch. URM has {self.n_users}, embeddings have {user_embeddings.shape[0]}"
        assert self.n_items == item_embeddings.shape[0], \
            f"{self.RECOMMENDER_NAME}: Item count mismatch. URM has {self.n_items}, embeddings have {item_embeddings.shape[0]}"
        assert user_embeddings.shape[1] == item_embeddings.shape[1], \
            f"{self.RECOMMENDER_NAME}: Embedding dimension mismatch."

        self.user_embeddings = user_embeddings.astype(np.float32)
        self.item_embeddings = item_embeddings.astype(np.float32)
        self.embedding_dim = self.user_embeddings.shape[1]
        
        # --- Pre-normalize item embeddings for efficient cosine similarity calculation ---
        self._print("Normalizing item embeddings...")
        item_norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        # Avoid division by zero for zero-vectors
        item_norms[item_norms == 0] = 1e-10
        self.item_embeddings_normalized = self.item_embeddings / item_norms
        self._print("Normalization of item embeddings complete.")


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        Computes the cosine similarity scores for all items for the given users.
        This is achieved by a dot product between the normalized user embeddings
        and the pre-normalized item embeddings.

        Args:
            user_id_array (np.ndarray): Array of user IDs for which to compute scores.
            items_to_compute (np.ndarray, optional): Not used in this implementation.

        Returns:
            np.ndarray: A matrix of scores of shape (len(user_id_array), n_items).
        """
        # Get the embeddings for the batch of users
        user_batch_embeddings = self.user_embeddings[user_id_array]

        # L2-normalize the user embeddings to prepare for cosine similarity calculation
        user_norms = np.linalg.norm(user_batch_embeddings, axis=1, keepdims=True)
        # Avoid division by zero for zero-vectors
        user_norms[user_norms == 0] = 1e-10
        user_batch_embeddings_normalized = user_batch_embeddings / user_norms
        
        # Compute scores by dot product with the pre-normalized item embeddings.
        # This is the most efficient way to get the full score matrix.
        scores = user_batch_embeddings_normalized @ self.item_embeddings_normalized.T

        if items_to_compute is not None:
            full_scores = np.full((len(user_id_array), self.n_items), -np.inf, dtype=np.float32)
            full_scores[:, items_to_compute] = scores[:, items_to_compute]
            return full_scores

        return scores


    def save_model(self, folder_path, file_name=None):
        """
        Saves the model to disk.
        """
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print(f"Saving model in folder '{folder_path}'")
        os.makedirs(folder_path, exist_ok=True)

        # Save other data
        data_dict = {
            'user_embeddings': self.user_embeddings,
            'item_embeddings': self.item_embeddings,
        }
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict)

        self._print("Saving complete.")


    def load_model(self, folder_path, file_name=None):
        """
        Loads the model from disk, reconstructing all necessary components.
        """
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print(f"Loading model from folder '{folder_path}'")

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        # Re-normalize item embeddings after loading
        self._print("Re-normalizing item embeddings after loading...")
        item_norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        item_norms[item_norms == 0] = 1e-10
        self.item_embeddings_normalized = self.item_embeddings / item_norms
        
        self._print("Loading complete.")

