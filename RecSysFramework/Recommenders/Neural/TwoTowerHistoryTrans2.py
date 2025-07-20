import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

class FastUserEmbedding(nn.Module):
    """
    Versione ottimizzata che usa una semplice media pesata invece del Transformer
    """
    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        # Parametri per la media pesata
        self.attention_weights = nn.Linear(embedding_dim, 1)
        
    def forward(self, padded_embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            padded_embeddings: [batch_size, max_len, embedding_dim]
            lengths: [batch_size] - lunghezze reali delle sequenze
        """
        # Calcola pesi di attenzione semplici
        attention_scores = self.attention_weights(padded_embeddings).squeeze(-1)  # [batch_size, max_len]
        
        # Maschera per ignorare il padding
        max_len = padded_embeddings.size(1)
        mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
        attention_scores = attention_scores.masked_fill(~mask, -float('inf'))
        
        # Applica softmax per ottenere pesi normalizzati
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, max_len]
        
        # Media pesata
        weighted_embeddings = torch.sum(padded_embeddings * attention_weights.unsqueeze(-1), dim=1)
        
        return self.layer_norm(self.dropout(weighted_embeddings))

class TwoTowerRecommender(nn.Module, BaseMatrixFactorizationRecommender):
    """
    Versione ottimizzata del TwoTowerRecommender con gestione intelligente della memoria
    """
    RECOMMENDER_NAME = "TwoTowerRecommender"
    
    def __init__(self, 
                 URM_train, 
                 num_users, 
                 num_items, 
                 layers=[10], 
                 user_embeddings=None,
                 item_embeddings=None,
                 verbose=True,
                 use_fast_aggregation=False,
                 max_memory_users=50000):  # Limita utenti in memoria
        
        super().__init__()
        BaseMatrixFactorizationRecommender.__init__(self, URM_train, verbose)

        self.n_users = num_users
        self.n_items = num_items
        self.use_fast_aggregation = use_fast_aggregation
        self.max_memory_users = max_memory_users
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self._print(f"Device selezionato: {self.device}")

        self.user_embedding = None
        self.item_embedding = None
        
        # Gestione delle cronologie utenti
        if user_embeddings is not None:
            self._print(f"Processamento embedding utenti per {num_users} utenti...")
            self.user_history_embeddings = user_embeddings 
            user_tower_input_dim = layers[0]
            
            # OTTIMIZZAZIONE: Crea un cache per batch processing
            self._prepare_user_cache()
            
            # Usa aggregazione veloce o Transformer
            if use_fast_aggregation:
                self.aggregation_model = FastUserEmbedding(user_tower_input_dim)
                self._print("Usando aggregazione veloce (media pesata)")
            else:
                self.aggregation_model = TransformerUserEmbedding(
                    embedding_dim=self.user_history_embeddings[0].shape[1], 
                    reduction_dim=user_tower_input_dim,
                    num_heads=2, 
                    num_layers=2,
                    dim_feedforward=user_tower_input_dim,
                    dropout=0.1
                )
                self._print("Usando Transformer (più lento)")
        else:
            self._print("Nessun embedding utente fornito. Creazione di un layer nn.Embedding addestrabile.")
            user_tower_input_dim = layers[0]
            self.user_embedding = nn.Embedding(self.n_users, user_tower_input_dim)
        
        if item_embeddings is not None:
            self._print("Embedding item pre-calcolati forniti. La torre item userà questi come input.")
            self.register_buffer('pretrained_item_embeddings', torch.tensor(item_embeddings, dtype=torch.float32))
            item_tower_input_dim = layers[0]
        else:
            self._print("Nessun embedding item fornito. Creazione di un layer nn.Embedding addestrabile.")
            item_tower_input_dim = layers[0]
            self.item_embedding = nn.Embedding(self.n_items, item_tower_input_dim)

        self.user_tower = self._create_tower(user_tower_input_dim, layers)
        self.item_tower = self._create_tower(item_tower_input_dim, layers)
        
        self.to(self.device)
        self.USER_factors = None
        self.ITEM_factors = None

    def _prepare_user_cache(self):
        """Prepara un cache per le cronologie utenti senza caricare tutto in memoria"""
        self._print("Preparazione cache utenti...")
        
        # Calcola statistiche per ottimizzare il processing
        self.user_lengths = []
        self.embedding_dim = None
        
        for i in range(self.n_users):
            if i < len(self.user_history_embeddings) and self.user_history_embeddings[i] is not None:
                history = self.user_history_embeddings[i]
                self.user_lengths.append(len(history))
                if self.embedding_dim is None:
                    self.embedding_dim = history.shape[1]
            else:
                self.user_lengths.append(0)
        
        self.user_lengths = np.array(self.user_lengths)
        self._print(f"Cache preparato. Dim embedding: {self.embedding_dim}, Lunghezza media: {np.mean(self.user_lengths):.1f}")

    def _get_user_batch_embeddings(self, user_ids):
        """Ottiene gli embedding per un batch di utenti in modo efficiente"""
        if self.user_embedding is not None:
            return self.user_embedding(user_ids)
        
        # Prepara i tensor per il batch
        batch_histories = []
        batch_lengths = []
        
        for user_id in user_ids:
            user_id = user_id.item() if isinstance(user_id, torch.Tensor) else user_id
            
            if (user_id < len(self.user_history_embeddings) and 
                self.user_history_embeddings[user_id] is not None):
                history = torch.tensor(self.user_history_embeddings[user_id], 
                                     dtype=torch.float32, device=self.device)
                batch_histories.append(history)
                batch_lengths.append(len(history))
            else:
                # Utente senza cronologia
                empty_history = torch.zeros(1, self.embedding_dim, dtype=torch.float32, device=self.device)
                batch_histories.append(empty_history)
                batch_lengths.append(1)
        
        # Padding del batch
        padded_histories = pad_sequence(batch_histories, batch_first=True, padding_value=0.0)
        lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long, device=self.device)
        
        # Aggregazione
        if self.use_fast_aggregation:
            return self.aggregation_model(padded_histories, lengths_tensor)
        else:
            # Per il Transformer, converti in lista
            histories_list = []
            for i, user_idx in enumerate(user_ids):
                length = batch_lengths[i]
                histories_list.append(padded_histories[i, :length])
            return self.aggregation_model(histories_list)

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
        user_vector = self._get_user_batch_embeddings(user_input)

        # Ottieni il vettore iniziale per l'item
        if self.item_embedding is not None:
            item_vector = self.item_embedding(item_input)
        else:
            item_vector = self.pretrained_item_embeddings[item_input]

        user_tower_output = self.user_tower(user_vector)
        item_tower_output = self.item_tower(item_vector)

        prediction_logits = (user_tower_output * item_tower_output).sum(dim=1, keepdim=True)
        
        return prediction_logits

    def compute_user_embeddings(self, batch_size=2048):  # Batch size ridotto
        """Calcola gli embedding finali per tutti gli utenti"""
        self.eval()
        user_ids_tensor = torch.arange(self.n_users, device=self.device)
        embeddings_list = []
        
        with torch.no_grad():
            for start_pos in tqdm(range(0, self.n_users, batch_size), desc="Computing user embeddings"):
                end_pos = min(start_pos + batch_size, self.n_users)
                batch_ids = user_ids_tensor[start_pos:end_pos]
                
                # Usa il metodo ottimizzato
                initial_vectors = self._get_user_batch_embeddings(batch_ids)
                final_embeddings = self.user_tower(initial_vectors)
                embeddings_list.append(final_embeddings.cpu().numpy())
                
                # Libera memoria cache
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
        self.USER_factors = np.concatenate(embeddings_list, axis=0)
        return self.USER_factors

    def compute_item_embeddings(self, batch_size=4096):
        self.eval()
        item_ids_tensor = torch.arange(self.n_items, device=self.device)
        embeddings_list = []
        with torch.no_grad():
            for start_pos in tqdm(range(0, self.n_items, batch_size), desc="Computing item embeddings"):
                end_pos = min(start_pos + batch_size, self.n_items)
                batch_ids = item_ids_tensor[start_pos:end_pos]
                
                if self.item_embedding is not None:
                    initial_vectors = self.item_embedding(batch_ids)
                else:
                    initial_vectors = self.pretrained_item_embeddings[batch_ids]
                    
                final_embeddings = self.item_tower(initial_vectors)
                embeddings_list.append(final_embeddings.cpu().numpy())
        self.ITEM_factors = np.concatenate(embeddings_list, axis=0)
        return self.ITEM_factors
    
    def compute_all_embeddings(self, batch_size=2048):
        self.compute_user_embeddings(batch_size)
        self.compute_item_embeddings(batch_size)

    def fit(self, epochs=10, batch_size=1024, num_negatives=1, optimizer=None, use_mixed_precision=False):
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

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if self.USER_factors is None or self.ITEM_factors is None:
            self.compute_all_embeddings()
        user_f = self.USER_factors[user_id_array]
        item_f = self.ITEM_factors
        if items_to_compute is not None:
            item_f = item_f[items_to_compute]
        scores = user_f @ item_f.T
        return scores

# Mantieni le classi originali per compatibilità
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