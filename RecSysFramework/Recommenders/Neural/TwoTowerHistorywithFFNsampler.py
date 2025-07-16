import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

class AttentionUserEmbedding(nn.Module):
    """
    Versione che utilizza un blocco simile a un Transformer, con un sotto-layer
    di attention e un sotto-layer di Feed-Forward Network (FFN) per ogni layer.
    """
    def __init__(self, embedding_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # --- Moduli per ogni layer ---
        self.attention_nets = nn.ModuleList(
            [nn.Linear(embedding_dim, 1, bias=False) for _ in range(num_layers)]
        )
        
        # FFN come richiesto, una per ogni layer
        self.ffns = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim * 2, embedding_dim)
            ) for _ in range(num_layers)]
        )
        
        # Due layer di normalizzazione per ogni blocco (uno dopo l'attention, uno dopo la FFN)
        self.layer_norms_1 = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(num_layers)]
        )
        self.layer_norms_2 = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        
        # Layer di attention finale per l'aggregazione
        self.final_attention = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, item_embeddings_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            item_embeddings_list (list[torch.Tensor]): Lista di tensori di cronologie.

        Returns:
            torch.Tensor: Embedding utente aggregato [batch_size, embedding_dim].
        """
        # 1. Padding e creazione della maschera
        lengths = torch.tensor([len(seq) for seq in item_embeddings_list], device=item_embeddings_list[0].device)
        padded_embeddings = pad_sequence(item_embeddings_list, batch_first=True, padding_value=0.0)
        
        max_len = padded_embeddings.size(1)
        mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
        
        current_embeddings = padded_embeddings

        # 2. Ciclo sui layer sequenziali (blocchi tipo Transformer)
        for i in range(self.num_layers):
            # --- Primo sotto-layer: Attention ---
            identity_1 = current_embeddings
            
            attention_scores = self.attention_nets[i](current_embeddings)
            attention_scores.masked_fill_(~mask.unsqueeze(-1), -float('inf'))
            attention_weights = F.softmax(attention_scores, dim=1)
            attention_output = attention_weights * identity_1
            
            # Connessione residua e normalizzazione
            x = self.layer_norms_1[i](identity_1 + self.dropout(attention_output))

            # --- Secondo sotto-layer: Feed-Forward Network (FFN) ---
            identity_2 = x
            ffn_output = self.ffns[i](x)
            
            # Connessione residua e normalizzazione
            current_embeddings = self.layer_norms_2[i](identity_2 + self.dropout(ffn_output))
            
            # Assicura che i valori di padding rimangano a zero
            current_embeddings.masked_fill_(~mask.unsqueeze(-1), 0.0)

        # 3. Aggregazione finale
        final_scores = self.final_attention(current_embeddings)
        final_scores.masked_fill_(~mask.unsqueeze(-1), -float('inf'))
        final_weights = F.softmax(final_scores, dim=1)
        
        user_embeddings = torch.sum(final_weights * current_embeddings, dim=1)
        
        return user_embeddings
    
class TwoTowerRecommender(nn.Module, BaseMatrixFactorizationRecommender):
    """
    Versione modificata che limita la lunghezza della cronologia utente
    campionando casualmente gli embedding se superano una soglia.
    """
    RECOMMENDER_NAME = "TwoTowerRecommender_SampledHistory"
    def __init__(self,
                 URM_train,
                 num_users,
                 num_items,
                 layers=[64],
                 user_embeddings=None,
                 item_embeddings=None,
                 attention_num_layers=2,
                 max_history_length=150, # <-- NUOVO PARAMETRO
                 verbose=True):

        super().__init__()
        BaseMatrixFactorizationRecommender.__init__(self, URM_train, verbose)
        self.DEBUG = False
        self.oneprint_forward = True
        self.oneprint_get_user_emb = True  # <-- Aggiungi questa riga
        self.oneprint_get_item_emb = True  # <-- Aggiungi questa riga
        self.n_users = num_users
        self.n_items = num_items
        self.max_history_length = max_history_length # <-- SALVA IL PARAMETRO

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self._print(f"Device selezionato: {self.device}")
        if user_embeddings is not None:
            # Stampa la nuova impostazione
            self._print(f"La lunghezza massima della cronologia è limitata a: {self.max_history_length}")

        self.aggregation_model = None
        self.user_embedding = None
        self.item_embedding = None

        if user_embeddings is not None:
            self._print(f"Embedding utenti pre-calcolati forniti. La torre utenti userà {attention_num_layers} layer di attenzione.")
            self.user_history_embeddings = user_embeddings
            user_tower_input_dim = self.user_history_embeddings[0].shape[1]
            # NOTA: Qui sto usando l'ultima versione di AttentionUserEmbedding che hai fornito
            self.aggregation_model = AttentionUserEmbedding(
                embedding_dim=user_tower_input_dim,
                num_layers=attention_num_layers
            )
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
        layers_list = []
        current_dim = input_dim
        for layer_dim in hidden_layers:
            layers_list.append(nn.Linear(current_dim, layer_dim))
            layers_list.append(nn.BatchNorm1d(layer_dim))
            layers_list.append(nn.ReLU())
            current_dim = layer_dim
        return nn.Sequential(*layers_list)

    # --- NUOVO METODO HELPER PER IL CAMPIONAMENTO ---
    def _get_sampled_histories(self, user_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        Recupera le cronologie per un batch di utenti, campionando se necessario.
        """
        histories_for_batch = []
        # Cicliamo sugli ID utente del batch
        for user_id in user_ids.cpu().numpy():
            history = self.user_history_embeddings[user_id]
            
            # Se la cronologia è più lunga del massimo consentito, campiona
            if len(history) > self.max_history_length:
                # Scegli 'max_history_length' indici casuali senza ripetizione
                sample_indices = np.random.choice(len(history), self.max_history_length, replace=False)
                sampled_history = history[sample_indices]
            else:
                sampled_history = history
            
            histories_for_batch.append(torch.tensor(sampled_history, dtype=torch.float32).to(self.device))
            
        return histories_for_batch

    def forward(self, user_input, item_input):
        # Ottieni il vettore iniziale per l'utente
        if self.user_embedding is not None:
            user_vector = self.user_embedding(user_input)
        else:
            # --- MODIFICA: Usa il metodo helper per ottenere le cronologie campionate ---
            histories_for_batch = self._get_sampled_histories(user_input)
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

    def compute_user_embeddings(self, batch_size=4096):
        """
        Calcola gli embedding finali per tutti gli utenti, applicando prima il
        modello di aggregazione (con campionamento) e poi la user tower.
        """
        self.eval()
        user_ids_tensor = torch.arange(self.n_users, device=self.device)
        embeddings_list = []

        with torch.no_grad():
            for start_pos in range(0, self.n_users, batch_size):
                end_pos = min(start_pos + batch_size, self.n_users)
                batch_ids = user_ids_tensor[start_pos:end_pos]

                if self.aggregation_model is not None:
                    # --- MODIFICA: Usa il metodo helper anche qui per coerenza ---
                    histories_for_batch = self._get_sampled_histories(batch_ids)
                    initial_vectors = self.aggregation_model(histories_for_batch)
                else:
                    initial_vectors = self.user_embedding(batch_ids)

                final_embeddings = self.user_tower(initial_vectors)
                embeddings_list.append(final_embeddings.cpu().numpy())

        self.USER_factors = np.concatenate(embeddings_list, axis=0)
        return self.USER_factors

    # Il resto dei metodi (fit, compute_embeddings, _compute_item_score, etc.)
    # rimangono invariati rispetto alla versione precedente.
    # Assicurati di includerli quando copi il codice.

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

    def fit(self, epochs=10, batch_size=2048, num_negatives=1, optimizer=None):
        self.train() 
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        num_batches = math.ceil(self.URM_train.nnz / batch_size)
        for i in range(epochs):
            self._print(f"Epoch {i+1}/{epochs} start")
            data_generator = self._data_generator_fixed(batch_size, num_negatives) 
            progress_bar = tqdm(iterable=data_generator, total=num_batches, desc=f"Epoch {i+1}/{epochs}", disable=not self.verbose)
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