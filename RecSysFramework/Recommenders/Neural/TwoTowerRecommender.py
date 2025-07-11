from RecSysFramework.Recommenders.Neural.DeepLearningRecommender import DeepLearningRecommender
import torch.nn as nn
import torch
import numpy as np
class TwoTowerRecommenderConcat(DeepLearningRecommender):
    
    RECOMMENDER_NAME = """TwoTowerRecommenderConcat"""
    
    def __init__(self, URM_train, num_users, num_items, layers=[10], verbose = True):
        super().__init__(URM_train, verbose)
        self.mlp_embedding_user = nn.Embedding(num_users, int(layers[0]/2), device=self.device) # <- The input for each tower will be a learned latent representation,
        self.mlp_embedding_item = nn.Embedding(num_items, int(layers[0]/2), device=self.device) # <- sort of like what we have seen for Matrix Factorization.

        self.mlp_layers = nn.ModuleList([
            nn.Linear(layers[i-1], layers[i], bias=True, device=self.device) for i in range(1, len(layers))
            ])
        for i, layer in enumerate(self.mlp_layers):
            nn.init.normal_(layer.weight)
            layer.bias.data.zero_()
            # layer.weight_decay = reg_layers[i]

        self.prediction_layer = nn.Linear(layers[-1], 1, bias=True, device=self.device)
        nn.init.uniform_(self.prediction_layer.weight)
        self.prediction_layer.bias.data.zero_()
        self.to(self.device)

    def forward(self, user_input, item_input):
        mlp_user_latent = self.mlp_embedding_user(user_input.long().to(self.device)) # <- shallow tower: we just extract the embedding corresponding to the profiles
        mlp_item_latent = self.mlp_embedding_item(item_input.long().to(self.device))
        mlp_vector = torch.cat((mlp_user_latent, mlp_item_latent), dim=1) # <- Concatenate user and item embeddings
        for layer in self.mlp_layers:
            mlp_vector = torch.relu(layer(mlp_vector)) # <- MLP after-merge processing block

        predict_vector = mlp_vector
        prediction = torch.sigmoid(self.prediction_layer(predict_vector))
        return prediction

class TwoTowerRecommenderProduct(DeepLearningRecommender):

    RECOMMENDER_NAME = """TwoTowerRecommenderProduct"""

    def __init__(self, URM_train, num_users, num_items, layers=[10], verbose = True):
        super().__init__(URM_train, verbose)
        self.mlp_embedding_user = nn.Embedding(num_users, layers[0], device=self.device)
        self.mlp_embedding_item = nn.Embedding(num_items, layers[0], device=self.device) # <- It's possible to make the towers asymmetric! Mind the output dimension though

        self.mlp_layers_tower1 = nn.ModuleList([ # <- First tower MLP
            nn.Linear(
                layers[i-1],
                layers[i], bias=True, device=self.device
                ) for i in range(1, len(layers))
            ])
        
        self.mlp_layers_tower2 = nn.ModuleList([ # <- Second tower MLP
            nn.Linear(
                layers[i-1],
                layers[i], bias=True, device=self.device
                ) for i in range(1, len(layers))
            ])
        
        for i, layer in enumerate(self.mlp_layers_tower1):
            nn.init.normal_(layer.weight)
            layer.bias.data.zero_()
            # layer.weight_decay = reg_layers[i]

        for i, layer in enumerate(self.mlp_layers_tower2):
            nn.init.normal_(layer.weight)
            layer.bias.data.zero_()
            # layer.weight_decay = reg_layers[i]

        self.prediction_layer = nn.Linear(layers[-1], 1, bias=True, device=self.device) # <- shallow post-merge block: a simple linear layer with sigmoid activation
        nn.init.uniform_(self.prediction_layer.weight)
        self.prediction_layer.bias.data.zero_()
        self.to(self.device)

    def forward(self, user_input, item_input):
        mlp_user_latent = self.mlp_embedding_user(user_input.long().to(self.device))
        mlp_item_latent = self.mlp_embedding_item(item_input.long().to(self.device))

        mlp_user_vector = mlp_user_latent
        mlp_item_vector = mlp_item_latent

        for layer in self.mlp_layers_tower1:
            mlp_user_vector = torch.relu(layer(mlp_user_vector))

        for layer in self.mlp_layers_tower2:
            mlp_item_vector = torch.relu(layer(mlp_item_vector))

        predict_vector = mlp_user_vector * mlp_item_vector # <- Merge the tensors via element-wise multiplication
        prediction = torch.sigmoid(self.prediction_layer(predict_vector))
        return prediction