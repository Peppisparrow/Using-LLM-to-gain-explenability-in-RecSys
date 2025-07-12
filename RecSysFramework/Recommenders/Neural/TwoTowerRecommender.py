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
    

class TwoTowerRecConcatNorm(DeepLearningRecommender):
    
    RECOMMENDER_NAME = """TwoTowerRecConcatNorm"""
    
    def __init__(self, URM_train, num_users, num_items, layers=[10], verbose=True):
        super().__init__(URM_train, verbose)
        self.mlp_embedding_user = nn.Embedding(num_users, int(layers[0]/2), device=self.device)
        self.mlp_embedding_item = nn.Embedding(num_items, int(layers[0]/2), device=self.device)

        # --- Modifica all'architettura ---
        self.mlp_layers = nn.ModuleList()
        # Il primo layer è l'input concatenato
        input_size = layers[0]
        
        for output_size in layers[1:]:
            # Aggiungi un layer lineare
            self.mlp_layers.append(nn.Linear(input_size, output_size, device=self.device))
            # Aggiungi la Batch Normalization
            self.mlp_layers.append(nn.BatchNorm1d(output_size, device=self.device))
            # Aggiungi l'attivazione
            self.mlp_layers.append(nn.ReLU())
            # L'input del prossimo layer sarà l'output di questo
            input_size = output_size
        # --- Fine Modifica ---

        # Il prediction layer ora prende l'output dell'ultimo blocco
        self.prediction_layer = nn.Linear(layers[-1], 1, bias=True, device=self.device)
        
        # Inizializzazione (può essere omessa se si usano le default di PyTorch)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        
        self.to(self.device)

    def forward(self, user_input, item_input):
        mlp_user_latent = self.mlp_embedding_user(user_input.long())
        mlp_item_latent = self.mlp_embedding_item(item_input.long())
        
        mlp_vector = torch.cat((mlp_user_latent, mlp_item_latent), dim=1)
        
        # --- Modifica al forward pass ---
        # Applica sequenzialmente i blocchi (Linear -> BatchNorm -> ReLU)
        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)
        # --- Fine Modifica ---
        
        # L'output dell'MLP va al prediction layer
        prediction = torch.sigmoid(self.prediction_layer(mlp_vector))
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
class TwoTowerRecProductNorm(DeepLearningRecommender):

    RECOMMENDER_NAME = """TwoTowerRecommenderProduct"""

    def __init__(self, URM_train, num_users, num_items, layers=[10], verbose = True):
        super().__init__(URM_train, verbose)
        
        # Gli embedding iniziali per ogni torre
        self.mlp_embedding_user = nn.Embedding(num_users, layers[0], device=self.device)
        self.mlp_embedding_item = nn.Embedding(num_items, layers[0], device=self.device)

        # --- Aggiunta di BatchNorm e ReLU per la Torre 1 (Utenti) ---
        self.mlp_layers_tower1 = nn.ModuleList()
        input_size_tower1 = layers[0]
        for output_size in layers[1:]:
            self.mlp_layers_tower1.append(nn.Linear(input_size_tower1, output_size, device=self.device))
            self.mlp_layers_tower1.append(nn.BatchNorm1d(output_size, device=self.device))
            self.mlp_layers_tower1.append(nn.ReLU())
            input_size_tower1 = output_size
        
        # --- Aggiunta di BatchNorm e ReLU per la Torre 2 (Item) ---
        self.mlp_layers_tower2 = nn.ModuleList()
        input_size_tower2 = layers[0]
        for output_size in layers[1:]:
            self.mlp_layers_tower2.append(nn.Linear(input_size_tower2, output_size, device=self.device))
            self.mlp_layers_tower2.append(nn.BatchNorm1d(output_size, device=self.device))
            self.mlp_layers_tower2.append(nn.ReLU())
            input_size_tower2 = output_size
        
        # Il prediction layer prende l'output dopo la fusione delle torri
        # self.prediction_layer = nn.Linear(layers[-1], 1, bias=True, device=self.device)
        
        self.to(self.device)

    def forward(self, user_input, item_input):
        # Estrazione degli embedding iniziali
        mlp_user_vector = self.mlp_embedding_user(user_input.long())
        mlp_item_vector = self.mlp_embedding_item(item_input.long())

        # Passaggio attraverso la prima torre (utente)
        for layer in self.mlp_layers_tower1:
            mlp_user_vector = layer(mlp_user_vector)

        # Passaggio attraverso la seconda torre (item)
        for layer in self.mlp_layers_tower2:
            mlp_item_vector = layer(mlp_item_vector)

        # Fusione tramite prodotto elemento per elemento
        predict_vector = torch.sum(mlp_user_vector * mlp_item_vector, dim=1, keepdim=True)
        
        # Predizione finale
        prediction = torch.sigmoid((predict_vector))
        return prediction