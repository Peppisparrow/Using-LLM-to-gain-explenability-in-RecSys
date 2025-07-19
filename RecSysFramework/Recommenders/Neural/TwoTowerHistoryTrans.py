import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.profiler
from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class TransformerUserEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, reduction_dim: int, num_heads: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.reduction_dim = reduction_dim
        self.reduction_layer = nn.Linear(embedding_dim, reduction_dim)
        self.pos_encoder = PositionalEncoding(reduction_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=reduction_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(reduction_dim)
        self.activation = nn.ReLU()

    def forward(self, item_embeddings_list: list[torch.Tensor]) -> torch.Tensor:
        padded_embeddings = pad_sequence(item_embeddings_list, batch_first=True, padding_value=0.0)
        reduced_embeddings = self.reduction_layer(padded_embeddings)
        reduced_embeddings = self.activation(reduced_embeddings)
        lengths = torch.tensor([len(seq) for seq in item_embeddings_list], device=padded_embeddings.device)
        max_len = reduced_embeddings.size(1)
        src_key_padding_mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]
        x = self.pos_encoder(reduced_embeddings)
        transformer_output = self.transformer_encoder(src=x, src_key_padding_mask=src_key_padding_mask)
        mask = ~src_key_padding_mask.unsqueeze(-1)
        transformer_output = transformer_output.masked_fill(~mask, 0.0)
        sum_embeddings = transformer_output.sum(dim=1)
        user_embeddings = sum_embeddings / (lengths.unsqueeze(1) + 1e-9)
        return self.layer_norm(user_embeddings)

    
class TwoTowerRecommender(nn.Module, BaseMatrixFactorizationRecommender):
    """
    Versione modificata del TwoTowerRecommender.
    - Accetta embedding pre-calcolati per utenti e/o item.
    - Se gli embedding non sono forniti, li impara tramite un layer nn.Embedding.
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
                 num_attention_layer=2,
                 num_heads=2,
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
        self.oneprint_get_user_emb = True  # <-- Aggiungi questa riga
        self.oneprint_get_item_emb = True  # <-- Aggiungi questa riga
        # --- FINE AGGIUNTA ---

        self.user_embedding = None
        self.item_embedding = None
        
        if user_embeddings is not None:
            self._print("Embedding utenti pre-calcolati forniti. La torre utenti userà questi come input.")
            self.user_history_embeddings = user_embeddings 
            user_tower_input_dim = layers[0]
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

        self.aggregation_model = TransformerUserEmbedding(
            embedding_dim=user_embeddings[0].shape[1], 
            reduction_dim=user_tower_input_dim,
            num_heads=num_heads, 
            num_layers=num_attention_layer, 
            dim_feedforward=user_tower_input_dim * 2, 
            dropout=0.1)
        
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
        if self.user_embedding is not None:
            user_vector = self.user_embedding(user_input)
        else:
            histories_for_batch = [torch.tensor(self.user_history_embeddings[i], dtype=torch.float32).to(self.device) for i in user_input]
            user_vector = self.aggregation_model(histories_for_batch)

        # Ottieni il vettore iniziale per l'item
        if self.item_embedding is not None:
            item_vector = self.item_embedding(item_input)
        else:
            item_vector = self.pretrained_item_embeddings[item_input]

        user_tower_output = self.user_tower(user_vector)
        item_tower_output = self.item_tower(item_vector)

        prediction_logits = (user_tower_output * item_tower_output).sum(dim=1, keepdim=True)
        
        return prediction_logits
    def _data_generator_fixed(self, batch_size, num_negatives=1):
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

    # def fit(self, epochs=10, batch_size=1024, num_negatives=1, optimizer=None, use_mixed_precision=True):
    #     self.train() 
    #     if optimizer is None:
    #         optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    #     loss_fn = torch.nn.BCEWithLogitsLoss()
        
    #     scaler = torch.amp.GradScaler(device='cuda') if use_mixed_precision and self.device.type == 'cuda' else None
        
    #     num_batches = math.ceil(self.URM_train.nnz / batch_size)
    #     self._print("Profiler enabled. Recording will start after 2 batches and last for 3 batches.")
        
        
    #         # Profileremo solo la prima epoca per semplicità
    #     for i in range(1):
    #         self._print(f"Epoch {i+1}/{epochs} start")
    #         data_generator = self._data_generator_fixed(batch_size, num_negatives) 
    #         progress_bar = tqdm(iterable=data_generator, total=num_batches, desc=f"Epoch {i+1}/{epochs}", disable=not self.verbose)
            
    #         for batch_idx, (user_input_ids, item_input_ids, labels) in enumerate(progress_bar):
    #             optimizer.zero_grad(set_to_none=True)
                
    #             if use_mixed_precision and self.device.type == 'cuda':
    #                 with torch.amp.autocast(device_type='cuda'):
    #                     predictions = self.forward(user_input_ids, item_input_ids)
    #                     loss = loss_fn(predictions, labels)
    #                 scaler.scale(loss).backward()
    #                 scaler.step(optimizer)
    #                 scaler.update()
    #             else:
    #                 pass
    #             if (batch_idx+1) % 1000 == 0:  # Inizia a profilare dopo 2 batch
                    
    #                 break
                
        
    def fit(self, epochs=10, batch_size=1024, num_negatives=1, optimizer=None, use_mixed_precision=True):
        self.train() 
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # Mixed precision training
        scaler = torch.amp.GradScaler(device='cuda') if use_mixed_precision else None
        num_batches = math.ceil(self.URM_train.nnz / batch_size)
        for i in range(epochs):
            self._print(f"Epoch {i+1}/{epochs} start")
            data_generator = self._data_generator_fixed(batch_size, num_negatives) 
            progress_bar = tqdm(iterable=data_generator, total=num_batches, desc=f"Epoch {i+1}/{epochs}", disable=not self.verbose)
            total_loss = 0
            
            for batch_idx, (user_input_ids, item_input_ids, labels) in enumerate(progress_bar):
                optimizer.zero_grad()
                
                if use_mixed_precision:
                    with torch.amp.autocast(device_type='cuda'):
                        predictions = self.forward(user_input_ids, item_input_ids)
                        loss = loss_fn(predictions, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = self.forward(user_input_ids, item_input_ids)
                    loss = loss_fn(predictions, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Pulizia memoria periodica
                if batch_idx % 100 == 0 and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_batches
            self._print(f"Epoch {i+1}/{epochs} finished. Average Loss: {avg_loss:.4f}\n")

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
                return self.user_history_embeddings[user_ids]
        
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
        """
        Calcola gli embedding finali per tutti gli utenti, applicando prima il
        modello di aggregazione e poi la user tower.
        """
        self.eval() # Imposta il modello in modalità valutazione
        user_ids_tensor = torch.arange(self.n_users, device=self.device)
        embeddings_list = []
        
        with torch.no_grad():
            for start_pos in range(0, self.n_users, batch_size):
                end_pos = min(start_pos + batch_size, self.n_users)
                batch_ids = user_ids_tensor[start_pos:end_pos]
                
                # --- LOGICA CORRETTA ---
                if self.aggregation_model is not None:
                    # 1. Recupera le cronologie per gli utenti del batch
                    histories_for_batch = [
                        torch.tensor(self.user_history_embeddings[i], dtype=torch.float32).to(self.device)
                        for i in batch_ids
                    ]
                    
                    # 2. Aggrega le cronologie per ottenere i vettori utente iniziali
                    #    (Questo è il passaggio che mancava)
                    initial_vectors = self.aggregation_model(histories_for_batch)
                else:
                    # Caso standard se non si usa l'aggregazione
                    initial_vectors = self.user_embedding(batch_ids)
                # --- FINE LOGICA CORRETTA ---

                # 3. Passa i vettori utente (ora aggregati e in un unico tensore) alla torre
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

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if self.USER_factors is None or self.ITEM_factors is None:
            self.compute_all_embeddings()
        user_f = self.USER_factors[user_id_array]
        item_f = self.ITEM_factors
        if items_to_compute is not None:
            item_f = item_f[items_to_compute]
        scores = user_f @ item_f.T
        return scores