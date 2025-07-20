import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F
from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

class TwoTowerRecommender(nn.Module, BaseMatrixFactorizationRecommender):
    """
    Versione modificata del TwoTowerRecommender.
    - Accetta embedding pre-calcolati per utenti e/o item.
    - Se gli embedding non sono forniti, li impara tramite un layer nn.Embedding.
    - Supporta sia BPR loss che BCE loss in modo modulare.
    - Include stampe di DEBUG per tracciare il flusso dei dati.
    """
    RECOMMENDER_NAME = "TwoTowerRecommender"
    
    def __init__(self, 
                 URM_train, 
                 num_users, 
                 num_items, 
                 layers=[10], 
                 user_embeddings=None,
                 item_embeddings=None,
                 verbose=True):
        
        super().__init__()
        BaseMatrixFactorizationRecommender.__init__(self, URM_train, verbose)

        self.n_users = num_users
        self.n_items = num_items
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self._print(f"Device selezionato: {self.device}")

        # --- AGGIUNTA FLAG DI DEBUG ---
        self.DEBUG = False
        self.oneprint_forward = True
        self.oneprint_get_user_emb = True
        self.oneprint_get_item_emb = True
        # --- FINE AGGIUNTA ---

        self.user_embedding = None
        self.item_embedding = None
        
        if user_embeddings is not None:
            self._print("Embedding utenti pre-calcolati forniti. La torre utenti userà questi come input.")
            self.register_buffer('pretrained_user_embeddings', torch.tensor(user_embeddings, dtype=torch.float32))
            user_tower_input_dim = self.pretrained_user_embeddings.shape[1]
        else:
            self._print("Nessun embedding utente fornito. Creazione di un layer nn.Embedding addestrabile.")
            user_tower_input_dim = layers[0]
            self.user_embedding = nn.Embedding(self.n_users, user_tower_input_dim)
        
        if item_embeddings is not None:
            self._print("Embedding item pre-calcolati forniti. La torre item userà questi come input.")
            self.register_buffer('pretrained_item_embeddings', torch.tensor(item_embeddings, dtype=torch.float32))
            item_tower_input_dim = self.pretrained_item_embeddings.shape[1]
        else:
            self._print("Nessun embedding item fornito. Creazione di un layer nn.Embedding addestrabile.")
            item_tower_input_dim = layers[0] if len(layers) > 0 else user_tower_input_dim
            self.item_embedding = nn.Embedding(self.n_items, item_tower_input_dim)

        self.user_tower = self._create_tower(user_tower_input_dim, layers)
        self.item_tower = self._create_tower(item_tower_input_dim, layers)
        
        self.to(self.device)
        self.USER_factors = None
        self.ITEM_factors = None

    def _create_tower(self, input_dim, hidden_layers):
        layers = []
        current_dim = input_dim
        for layer_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, layer_dim))
            layers.append(nn.BatchNorm1d(layer_dim))
            layers.append(nn.ReLU())
            current_dim = layer_dim
        return nn.Sequential(*layers)

    def forward(self, user_input, item_input):
        # Ottieni il vettore iniziale per l'utente
        if self.user_embedding is not None:
            if self.DEBUG and self.oneprint_forward:
                print("DEBUG: Forward pass usa `self.user_embedding` (addestrabile).")
            user_vector = self.user_embedding(user_input)
        else:
            if self.DEBUG and self.oneprint_forward:
                print("DEBUG: Forward pass usa `self.pretrained_user_embeddings` (pre-calcolati).")
            user_vector = self.pretrained_user_embeddings[user_input]

        # Ottieni il vettore iniziale per l'item
        if self.item_embedding is not None:
            if self.DEBUG and self.oneprint_forward:
                print("DEBUG: Forward pass usa `self.item_embedding` (addestrabile).")
            item_vector = self.item_embedding(item_input)
        else:
            if self.DEBUG and self.oneprint_forward:
                print("DEBUG: Forward pass usa `self.pretrained_item_embeddings` (pre-calcolati).")
            item_vector = self.pretrained_item_embeddings[item_input]

        # --- AGGIUNTA FLAG DI DEBUG ---
        # Disattiva le stampe dopo la prima esecuzione del forward
        if self.DEBUG and self.oneprint_forward:
            self.oneprint_forward = False
        # --- FINE AGGIUNTA ---

        user_tower_output = self.user_tower(user_vector)
        item_tower_output = self.item_tower(item_vector)

        prediction_logits = (user_tower_output * item_tower_output).sum(dim=1, keepdim=True)
        
        return prediction_logits

    def _get_initial_embeddings(self, user_ids=None, item_ids=None):
        """Helper per ottenere i vettori iniziali prima delle torri, con stampe di DEBUG."""
        
        if user_ids is not None:
            # Blocco di DEBUG per gli utenti (eseguito una sola volta)
            if self.DEBUG and self.oneprint_get_user_emb:
                source = "`self.user_embedding` (addestrabile)" if self.user_embedding is not None else "`self.pretrained_user_embeddings` (pre-calcolati)"
                print(f"DEBUG: Chiamata a `_get_initial_embeddings` per USERS. Sorgente di input: {source}")
                self.oneprint_get_user_emb = False
            
            # Logica originale per restituire gli embedding
            if self.user_embedding is not None:
                return self.user_embedding(user_ids)
            else:
                return self.pretrained_user_embeddings[user_ids]
        
        if item_ids is not None:
            # Blocco di DEBUG per gli item (eseguito una sola volta)
            if self.DEBUG and self.oneprint_get_item_emb:
                source = "`self.item_embedding` (addestrabile)" if self.item_embedding is not None else "`self.pretrained_item_embeddings` (pre-calcolati)"
                print(f"DEBUG: Chiamata a `_get_initial_embeddings` per ITEMS. Sorgente di input: {source}")
                self.oneprint_get_item_emb = False

            # Logica originale per restituire gli embedding
            if self.item_embedding is not None:
                return self.item_embedding(item_ids)
            else:
                return self.pretrained_item_embeddings[item_ids]

    def compute_user_embeddings(self, batch_size=4096):
        self.eval()
        user_ids_tensor = torch.arange(self.n_users, device=self.device)
        embeddings_list = []
        with torch.no_grad():
            for start_pos in range(0, self.n_users, batch_size):
                end_pos = min(start_pos + batch_size, self.n_users)
                batch_ids = user_ids_tensor[start_pos:end_pos]
                initial_vectors = self._get_initial_embeddings(user_ids=batch_ids)
                final_embeddings = self.user_tower(initial_vectors)
                embeddings_list.append(final_embeddings.cpu().numpy())
        self.USER_factors = np.concatenate(embeddings_list, axis=0)
        return self.USER_factors

    def compute_item_embeddings(self, batch_size=4096):
        self.eval()
        item_ids_tensor = torch.arange(self.n_items, device=self.device)
        embeddings_list = []
        with torch.no_grad():
            for start_pos in range(0, self.n_items, batch_size):
                end_pos = min(start_pos + batch_size, self.n_items)
                batch_ids = item_ids_tensor[start_pos:end_pos]
                initial_vectors = self._get_initial_embeddings(item_ids=batch_ids)
                final_embeddings = self.item_tower(initial_vectors)
                embeddings_list.append(final_embeddings.cpu().numpy())
        self.ITEM_factors = np.concatenate(embeddings_list, axis=0)
        return self.ITEM_factors
    
    def compute_all_embeddings(self, batch_size=4096):
        self.compute_user_embeddings(batch_size)
        self.compute_item_embeddings(batch_size)

    def fit(self, epochs=10, batch_size=2048, num_negatives=1, optimizer=None, loss_type='bce'):
        """
        Addestra il modello con la loss function specificata.
        
        Args:
            epochs: Numero di epoche di training
            batch_size: Dimensione del batch
            num_negatives: Numero di campioni negativi per ogni positivo (utilizzato solo con BCE)
            optimizer: Optimizer da utilizzare (default: Adam)
            loss_type: Tipo di loss function ('bce' o 'bpr')
        """
        print(f"Training {self.RECOMMENDER_NAME} con {loss_type.upper()} loss per {epochs} epoche e batch_size={batch_size}")
        self.train() 
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        # Seleziona la loss function appropriata
        if loss_type.lower() == 'bce':
            loss_fn = torch.nn.BCEWithLogitsLoss()
            data_generator = self._data_generator_bce(batch_size, num_negatives)
            self._print(f"Training con BCE loss (num_negatives={num_negatives})")
        elif loss_type.lower() == 'bpr':
            loss_fn = self._bpr_loss
            data_generator = self._data_generator_bpr(batch_size)
            self._print(f"Training con BPR loss")
        else:
            raise ValueError(f"loss_type non supportato: {loss_type}. Usa 'bce' o 'bpr'.")
        
        num_batches = math.ceil(self.URM_train.nnz / batch_size)
        for i in range(epochs):
            self._print(f"Epoch {i+1}/{epochs} start")
            progress_bar = tqdm(iterable=data_generator, total=num_batches, desc=f"Epoch {i+1}/{epochs}", disable=not self.verbose)
            total_loss = 0
            
            for batch_data in progress_bar:
                optimizer.zero_grad()
                
                if loss_type.lower() == 'bce':
                    user_input_ids, item_input_ids, labels = batch_data
                    predictions = self.forward(user_input_ids, item_input_ids)
                    loss = loss_fn(predictions, labels)
                elif loss_type.lower() == 'bpr':
                    user_input_ids, pos_item_ids, neg_item_ids = batch_data
                    pos_predictions = self.forward(user_input_ids, pos_item_ids)
                    neg_predictions = self.forward(user_input_ids, neg_item_ids)
                    loss = loss_fn(pos_predictions, neg_predictions)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / num_batches
            self._print(f"Epoch {i+1}/{epochs} finished. Average Loss: {avg_loss:.4f}\n")

    def _bpr_loss(self, pos_predictions, neg_predictions):
        """
        Calcola la BPR (Bayesian Personalized Ranking) loss.
        
        Args:
            pos_predictions: Predizioni per gli item positivi
            neg_predictions: Predizioni per gli item negativi
            
        Returns:
            BPR loss value
        """
        diff = pos_predictions - neg_predictions
        return -torch.mean(torch.log(torch.sigmoid(diff)))

    def _data_generator_bce(self, batch_size, num_negatives=1):
        """
        Generatore di dati per BCE loss (stessa logica del _data_generator_fixed originale).
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
                user_input.append(u)
                item_input.append(i)
                labels.append(1)
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

    def _data_generator_bpr(self, batch_size):
        """
        Generatore di dati per BPR loss.
        Genera triple (user, positive_item, negative_item).
        """
        rows, cols = self.URM_train.nonzero()
        positive_pairs = list(zip(rows, cols))
        dok_train = self.URM_train.todok()
        n_positive_samples = len(positive_pairs)
        np.random.shuffle(positive_pairs)

        for start_pos in range(0, n_positive_samples, batch_size):
            user_input, pos_item_input, neg_item_input = [], [], []
            batch_positive_pairs = positive_pairs[start_pos : start_pos + batch_size]
            
            for u, i in batch_positive_pairs:
                # Aggiungi campione positivo
                user_input.append(u)
                pos_item_input.append(i)
                
                # Genera campione negativo
                j = np.random.randint(self.n_items)
                while (u, j) in dok_train: 
                    j = np.random.randint(self.n_items)
                neg_item_input.append(j)
            
            user_tensor = torch.tensor(user_input, dtype=torch.long, device=self.device)
            pos_item_tensor = torch.tensor(pos_item_input, dtype=torch.long, device=self.device)
            neg_item_tensor = torch.tensor(neg_item_input, dtype=torch.long, device=self.device)
            
            yield user_tensor, pos_item_tensor, neg_item_tensor

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if self.USER_factors is None or self.ITEM_factors is None:
            self.compute_all_embeddings()
        user_f = self.USER_factors[user_id_array]
        item_f = self.ITEM_factors
        if items_to_compute is not None:
            item_f = item_f[items_to_compute]
        scores = user_f @ item_f.T
        return scores