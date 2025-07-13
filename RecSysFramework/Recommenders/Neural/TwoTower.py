from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F
class TwoTowerRecommender(nn.Module, BaseMatrixFactorizationRecommender):
    """
    Implementazione di un modello Two-Tower Recommender con PyTorch.

    Ogni torre (una per gli utenti, una per gli item) è una rete neurale
    la cui architettura è definita dinamicamente. Il modello calcola gli 
    embedding, e la predizione è data dal prodotto scalare di questi.
    """

    def __init__(self, 
                 URM_train, 
                 num_users, 
                 num_items, 
                 layers=[10], 
                 verbose=True):
        """
        Costruttore della classe con l'interfaccia richiesta.

        :param URM_train: Matrice di interazione sparse (CSR).
        :param num_users: Numero totale di utenti nel dataset.
        :param num_items: Numero totale di item nel dataset.
        :param layers: Lista di interi che definisce le dimensioni dei layer
                       delle torri. La prima dimensione (layers[0]) definisce
                       la dimensione degli embedding. Le dimensioni successive
                       definiscono i layer densi.
                       Se len(layers) == 1, il modello si comporta come una
                       semplice Matrix Factorization.
        :param verbose: Flag per abilitare i messaggi di log.
        """
        super().__init__()
        BaseMatrixFactorizationRecommender.__init__(self, URM_train, verbose)

        # 1. Sovrascriviamo n_users e n_items con i valori passati
        self.n_users = num_users
        self.n_items = num_items
        
        # 2. Scelta del device (GPU/MPS/CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self._print(f"Device selezionato: {self.device}")

        embedding_dim = layers[0]
        self.layers = layers

        self.user_embedding = nn.Embedding(
            num_embeddings=self.n_users, 
            embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=self.n_items, 
            embedding_dim=embedding_dim
        )

        self.user_tower = self._create_tower(self.layers)
        self.item_tower = self._create_tower(self.layers)
        
        self.to(self.device)

        self.USER_factors = None
        self.ITEM_factors = None


    def _create_tower(self, layer_sizes):
        """
        Funzione helper per creare una singola torre (user o item).
        """
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)

    def forward(self, user_input, item_input):
        """
        Esegue la forward pass del modello.
        
        :param user_input: Tensor di ID utente.
        :param item_input: Tensor di ID item.
        :return: Predizioni (punteggi tra 0 e 1).
        """
        # Calcola gli embedding per la torre utente
        user_embedding_vector = self.user_embedding(user_input)
        user_tower_output = self.user_tower(user_embedding_vector)

        # Calcola gli embedding per la torre item
        item_embedding_vector = self.item_embedding(item_input)
        item_tower_output = self.item_tower(item_embedding_vector)

        # Prodotto scalare e sigmoide per la predizione
        # (u * i).sum(dim=1) calcola il dot product riga per riga
        prediction = (user_tower_output * item_tower_output).sum(dim=1, keepdim=True)
        
        return prediction

    def _data_generator_fixed(self, batch_size, num_negatives=1):
        """
        Generatore di dati ottimizzato per il training.
        """
        rows, cols = self.URM_train.nonzero()
        positive_pairs = list(zip(rows, cols))
        dok_train = self.URM_train.todok()
        n_positive_samples = len(positive_pairs)
        np.random.shuffle(positive_pairs)

        for start_pos in range(0, n_positive_samples, batch_size):
            user_input, item_input, labels = [], [], []
            
            batch_positive_pairs = positive_pairs[start_pos : start_pos + batch_size]

            for u, i in batch_positive_pairs:
                # Campione positivo
                user_input.append(u)
                item_input.append(i)
                labels.append(1)

                # Campioni negativi
                for _ in range(num_negatives):
                    j = np.random.randint(self.n_items)
                    while (u, j) in dok_train: 
                        j = np.random.randint(self.n_items)
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)
            
            user_tensor = torch.tensor(user_input, dtype=torch.long, device=self.device)
            item_tensor = torch.tensor(item_input, dtype=torch.long, device=self.device)
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device).reshape((-1, 1))
            
            yield user_tensor, item_tensor, labels_tensor

    def fit(self, epochs=10, batch_size=2048, num_negatives=1, optimizer=None):
        """
        Esegue il training del modello.
        """
        self.train() # Imposta il modello in modalità training
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        num_batches = math.ceil(self.URM_train.nnz / batch_size)

        for i in range(epochs):
            self._print(f"Epoch {i+1}/{epochs} start")
            
            data_generator = self._data_generator_fixed(batch_size, num_negatives) 
            
            progress_bar = tqdm(
                iterable=data_generator, 
                total=num_batches,
                desc=f"Epoch {i+1}/{epochs}",
                disable=not self.verbose
            )
            
            total_loss = 0
            for user_input_ids, item_input_ids, labels in progress_bar:
                optimizer.zero_grad()

                predictions = self.forward(user_input_ids, item_input_ids)
                
                loss = loss_fn(predictions, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            avg_loss = total_loss / num_batches
            self._print(f"Epoch {i+1}/{epochs} finished. Average Loss: {avg_loss:.4f}\n")


    def compute_user_embeddings(self, batch_size=4096):
        """
        Calcola gli embedding per tutti gli utenti in batch e li salva 
        in self.USER_factors.

        :param batch_size: La dimensione dei lotti per il calcolo.
        :return: La matrice numpy degli embedding degli utenti (n_users, embedding_dim).
        """
        self.eval() # Imposta il modello in modalità valutazione
        self._print(f"Inizio calcolo user embeddings con batch_size: {batch_size}")

        user_ids_tensor = torch.arange(self.n_users, device=self.device)
        user_embeddings_list = []

        with torch.no_grad():
            for start_pos in range(0, self.n_users, batch_size):
                end_pos = min(start_pos + batch_size, self.n_users)
                user_batch_ids = user_ids_tensor[start_pos:end_pos]
                
                # Calcolo degli embedding per il batch corrente
                user_initial_emb = self.user_embedding(user_batch_ids)
                user_batch_embeddings = self.user_tower(user_initial_emb)
                
                user_embeddings_list.append(user_batch_embeddings.cpu().numpy())
        
        # Concatena i risultati di tutti i batch
        self.USER_factors = np.concatenate(user_embeddings_list, axis=0)
        self._print(f"Calcolo user embeddings completato. Shape: {self.USER_factors.shape}")
        return self.USER_factors


    def compute_item_embeddings(self, batch_size=4096):
        """
        Calcola gli embedding per tutti gli item in batch e li salva
        in self.ITEM_factors.

        :param batch_size: La dimensione dei lotti per il calcolo.
        :return: La matrice numpy degli embedding degli item (n_items, embedding_dim).
        """
        self.eval() # Imposta il modello in modalità valutazione
        self._print(f"Inizio calcolo item embeddings con batch_size: {batch_size}")

        item_ids_tensor = torch.arange(self.n_items, device=self.device)
        item_embeddings_list = []

        with torch.no_grad():
            for start_pos in range(0, self.n_items, batch_size):
                end_pos = min(start_pos + batch_size, self.n_items)
                item_batch_ids = item_ids_tensor[start_pos:end_pos]
                
                # Calcolo degli embedding per il batch corrente
                item_initial_emb = self.item_embedding(item_batch_ids)
                item_batch_embeddings = self.item_tower(item_initial_emb)
                
                item_embeddings_list.append(item_batch_embeddings.cpu().numpy())
        
        # Concatena i risultati di tutti i batch
        self.ITEM_factors = np.concatenate(item_embeddings_list, axis=0)
        self._print(f"Calcolo item embeddings completato. Shape: {self.ITEM_factors.shape}")
        return self.ITEM_factors
    
    def compute_all_embeddings(self, batch_size=4096):
        """
        Calcola e salva gli embedding per tutti gli utenti e item
        chiamando le funzioni di calcolo specifiche in batch.
        """
        self.compute_user_embeddings(batch_size)
        self.compute_item_embeddings(batch_size)

    # def _compute_item_score(self, user_id_array, items_to_compute=None):
    #     """
    #     Calcola i punteggi di raccomandazione per un dato array di utenti.
    #     Utilizza gli embedding pre-calcolati (USER_factors, ITEM_factors).
    #     """
    #     # Se gli embedding non sono stati calcolati, lanciali ora
    #     if self.USER_factors is None or self.ITEM_factors is None:
    #         self._print("Embedding non ancora calcolati. Avvio di compute_all_embeddings...")
    #         self.compute_all_embeddings()

    #     if items_to_compute is None:
    #         items_to_compute = np.arange(self.n_items)

    #     # Estrai gli embedding per gli utenti e item richiesti
    #     user_factors = self.USER_factors[user_id_array]
    #     item_factors = self.ITEM_factors[items_to_compute]

    #     # Calcola i punteggi tramite prodotto matriciale
    #     # (N_users, Emb_size) @ (Emb_size, N_items) -> (N_users, N_items)
    #     scores = user_factors @ item_factors.T
        
    #     return scores

    # def _compute_item_score(self, user_id_array, items_to_compute=None):
    #     """
    #     Calcola i punteggi di raccomandazione normalizzando gli embedding.
    #     Questo produce una similarità cosenica, rendendo i punteggi stabili.
    #     """
    #     if self.USER_factors is None or self.ITEM_factors is None:
    #         self._print("Embedding non ancora calcolati. Avvio di compute_all_embeddings...")
    #         self.compute_all_embeddings()

    #     if items_to_compute is None:
    #         items_to_compute = np.arange(self.n_items)

    #     # Estrai i fattori (che sono array numpy) e riconvertili in tensori
    #     user_factors = torch.from_numpy(self.USER_factors[user_id_array])
    #     item_factors = torch.from_numpy(self.ITEM_factors[items_to_compute])

    #     # --- NORMALIZZAZIONE L2 ---
    #     # Questa è la modifica cruciale
    #     user_factors = F.normalize(user_factors, p=2, dim=1)
    #     item_factors = F.normalize(item_factors, p=2, dim=1)

    #     # Calcola i punteggi tramite prodotto matriciale
    #     scores = user_factors @ item_factors.T
        
    #     # Riconverti in array numpy come si aspetta l'evaluator
    #     return scores.numpy()