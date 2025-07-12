from RecSysFramework.Recommenders.Neural.DeepLearningRecommender import DeepLearningRecommender
import torch.nn as nn
import torch
import numpy as np

class TwoTowerRecProductNorm(DeepLearningRecommender):

    RECOMMENDER_NAME = """TwoTowerRecommenderProduct"""

    def __init__(self, URM_train, num_users, num_items, user_embeddings_dim=None, item_embeddings_dim=None, first_dim_layer=10, layers=[10], verbose = True):
        super().__init__(URM_train, verbose)

        self.mlp_embedding_user = None
        self.mlp_embedding_item = None

        if user_embeddings_dim is None and item_embeddings_dim is None:
            self.mlp_embedding_user = nn.Embedding(num_users, first_dim_layer, device=self.device)
            self.mlp_embedding_item = nn.Embedding(num_items, first_dim_layer, device=self.device)
            input_size_tower_user = first_dim_layer
            input_size_tower_item = first_dim_layer

        elif user_embeddings_dim is None and item_embeddings_dim is not None:
            self.mlp_embedding_user = nn.Embedding(num_users, item_embeddings_dim, device=self.device)
            input_size_tower_user = item_embeddings_dim
            input_size_tower_item = item_embeddings_dim

        elif user_embeddings_dim is not None and item_embeddings_dim is None:
            self.mlp_embedding_item = nn.Embedding(num_items, user_embeddings_dim, device=self.device)
            input_size_tower_user = user_embeddings_dim
            input_size_tower_item = user_embeddings_dim
            
        else:
            input_size_tower_user = user_embeddings_dim
            input_size_tower_item = item_embeddings_dim
        
        self.mlp_layers_tower_user = nn.ModuleList()
        for output_size in layers:
            self.mlp_layers_tower_user.append(nn.Linear(input_size_tower_user, output_size, device=self.device))
            self.mlp_layers_tower_user.append(nn.BatchNorm1d(output_size, device=self.device))
            self.mlp_layers_tower_user.append(nn.ReLU())
            input_size_tower_user = output_size
        
        self.mlp_layers_tower_item = nn.ModuleList()
        for output_size in layers:
            self.mlp_layers_tower_item.append(nn.Linear(input_size_tower_item, output_size, device=self.device))
            self.mlp_layers_tower_item.append(nn.BatchNorm1d(output_size, device=self.device))
            self.mlp_layers_tower_item.append(nn.ReLU())
            input_size_tower_item = output_size
    
        
        self.to(self.device)

    def forward(self, user_input, item_input, user_embedding=None, item_embedding=None):

        if user_embedding is None and item_embedding is None:
            mlp_user_vector = self.mlp_embedding_user(user_input.long())
            mlp_item_vector = self.mlp_embedding_item(item_input.long())

        elif user_embedding is None and item_embedding is not None:
            mlp_user_vector = self.mlp_embedding_user(user_input.long())
            mlp_item_vector = item_embedding

        elif user_embedding is not None and item_embedding is None:
            mlp_user_vector = user_embedding
            mlp_item_vector = self.mlp_embedding_item(item_input.long())

        else:
            mlp_user_vector = user_embedding
            mlp_item_vector = item_embedding

        for layer in self.mlp_layers_tower_user:
            mlp_user_vector = layer(mlp_user_vector)

        for layer in self.mlp_layers_tower_item:
            mlp_item_vector = layer(mlp_item_vector)

        predict_vector = torch.sum(mlp_user_vector * mlp_item_vector, dim=1, keepdim=True)
        
        prediction = torch.sigmoid((predict_vector))
        return prediction