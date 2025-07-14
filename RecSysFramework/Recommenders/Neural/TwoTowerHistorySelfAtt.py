import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

class SelfAttentionUserEmbedding(nn.Module):
    """
    Versione che utilizza nn.MultiheadAttention per implementare la self-attention
    e aggregare le cronologie degli utenti di lunghezza variabile.
    """
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embedding_dim (int): La dimensione degli embedding degli item.
            num_heads (int): Il numero di teste di attenzione parallele.
                             Deve essere un divisore di embedding_dim.
            dropout (float): La probabilità di dropout da applicare.
        """
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) deve essere divisibile per num_heads ({num_heads})")
            
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # MHA di PyTorch si aspetta (seq_len, batch, dim)
        )
        # Layer opzionali per processare l'output, se necessario
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )


    def forward(self, item_embeddings_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            item_embeddings_list (list[torch.Tensor]): Una lista di tensori.
                Ogni tensore ha shape [num_items_per_user, embedding_dim].

        Returns:
            torch.Tensor: Un tensore di user embeddings aggregati di shape
                          [batch_size, embedding_dim].
        """
        # 1. Padding e creazione della maschera
        padded_embeddings = pad_sequence(item_embeddings_list, batch_first=True, padding_value=0.0)
        # padded_embeddings ha shape [batch_size, max_len, embedding_dim]

        lengths = torch.tensor([len(seq) for seq in item_embeddings_list], device=padded_embeddings.device)
        max_len = padded_embeddings.size(1)
        
        # La maschera per MHA indica le posizioni da ignorare (True per il padding)
        key_padding_mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]
        # key_padding_mask ha shape [batch_size, max_len]

        # 2. Trasposizione per nn.MultiheadAttention
        # L'input deve essere [max_len, batch_size, embedding_dim]
        padded_embeddings_t = padded_embeddings.transpose(0, 1)

        # 3. Applicazione della Self-Attention
        # Query, Key, e Value sono lo stesso tensore di input
        attn_output, _ = self.attention(
            query=padded_embeddings_t,
            key=padded_embeddings_t,
            value=padded_embeddings_t,
            key_padding_mask=key_padding_mask,
            need_weights=False # Non ci servono i pesi di attenzione per l'output
        )
        # attn_output ha shape [max_len, batch_size, embedding_dim]

        # 4. Ri-trasposizione dell'output a [batch_size, max_len, embedding_dim]
        attn_output = attn_output.transpose(0, 1)

        # Applica la maschera anche all'output per azzerare i valori di padding prima dell'aggregazione
        mask = ~key_padding_mask.unsqueeze(-1)
        attn_output = attn_output.masked_fill(~mask, 0.0)

        # 5. Aggregazione dell'output
        # Calcoliamo la media degli embedding contestualizzati (solo sulle parti non-paddate)
        sum_embeddings = attn_output.sum(dim=1)
        # Dividiamo per la lunghezza effettiva di ogni sequenza per una media corretta
        # Aggiungiamo 1e-9 per evitare divisione per zero se una sequenza fosse vuota
        user_embeddings = sum_embeddings / (lengths.unsqueeze(1) + 1e-9)
        
        # Passaggio opzionale attraverso un feed-forward network
        user_embeddings = self.layer_norm(user_embeddings + self.ffn(user_embeddings))
        
        return user_embeddings
    
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
                 verbose=True):
        
        super().__init__()
        BaseMatrixFactorizationRecommender.__init__(self, URM_train, verbose)

        self.n_users = num_users
        self.n_items = num_items
        self.aggregation_model = SelfAttentionUserEmbedding(embedding_dim=user_embeddings[0].shape[1], num_heads=2)
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
            user_tower_input_dim = self.user_history_embeddings[0].shape[1]
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
            histories_for_batch = [torch.tensor(self.user_history_embeddings[i], dtype=torch.float32).to(self.device) for i in user_input]
            user_vector = self.aggregation_model(histories_for_batch)

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