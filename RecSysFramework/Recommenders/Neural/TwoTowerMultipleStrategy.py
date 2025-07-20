import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F
from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

class TwoTowerRecommender(nn.Module, BaseMatrixFactorizationRecommender):
    """
    Enhanced TwoTowerRecommender with mixed embedding support.
    - Can combine learnable embeddings with pre-computed embeddings
    - Supports three modes: learnable_only, pretrained_only, or mixed
    - Includes fusion strategies for combining different embedding types
    """
    RECOMMENDER_NAME = "TwoTowerRecommender"
    
    def __init__(self, 
                 URM_train, 
                 num_users, 
                 num_items, 
                 layers=[10], 
                 user_embeddings=None,
                 item_embeddings=None,
                 user_embedding_mode='mixed',  # 'learnable_only', 'pretrained_only', 'mixed'
                 item_embedding_mode='mixed',  # 'learnable_only', 'pretrained_only', 'mixed'
                 fusion_strategy='concatenate',  # 'concatenate', 'add', 'weighted_sum', 'gated'
                 learnable_embedding_dim=None,  # If None, uses layers[0]
                 verbose=True,
                 debug=False):
        
        super().__init__()
        BaseMatrixFactorizationRecommender.__init__(self, URM_train, verbose)

        self.n_users = num_users
        self.n_items = num_items
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self._print(f"Device selezionato: {self.device}")

        # --- DEBUG FLAGS ---
        self.DEBUG = debug
        self.oneprint_forward = True
        self.oneprint_get_user_emb = True
        self.oneprint_get_item_emb = True
        
        # --- EMBEDDING CONFIGURATION ---
        self.user_embedding_mode = user_embedding_mode
        self.item_embedding_mode = item_embedding_mode
        self.fusion_strategy = fusion_strategy
        
        # Set default learnable embedding dimension
        if learnable_embedding_dim is None:
            learnable_embedding_dim = layers[0] if len(layers) > 0 else 64
        self.learnable_embedding_dim = learnable_embedding_dim
        
        # --- USER EMBEDDING SETUP ---
        self.user_embedding = None
        user_tower_input_dim = self._setup_user_embeddings(user_embeddings)
        
        # --- ITEM EMBEDDING SETUP ---
        self.item_embedding = None
        item_tower_input_dim = self._setup_item_embeddings(item_embeddings)
        
        # --- DEBUG PRINTS ---
        if self.DEBUG:
            print(f"DEBUG: User embedding mode: {self.user_embedding_mode}, Item embedding mode: {self.item_embedding_mode}")
            print(f"DEBUG: Fusion strategy: {self.fusion_strategy}")
            print(f"DEBUG: Learnable User/Item embedding dimension: {self.learnable_embedding_dim}")
            print(f"DEBUG: User tower input dimension: {user_tower_input_dim}, Item tower input dimension: {item_tower_input_dim}")
        
        # --- TOWER CREATION ---
        self.user_tower = self._create_tower(user_tower_input_dim, layers)
        self.item_tower = self._create_tower(item_tower_input_dim, layers)
        
        self.to(self.device)
        self.USER_factors = None
        self.ITEM_factors = None

    def _setup_user_embeddings(self, user_embeddings):
        """Setup user embedding configuration based on mode"""
        if self.user_embedding_mode == 'learnable_only':
            self._print("Modalità utenti: solo embedding addestrabili")
            self.user_embedding = nn.Embedding(self.n_users, self.learnable_embedding_dim)
            return self.learnable_embedding_dim
            
        elif self.user_embedding_mode == 'pretrained_only':
            if user_embeddings is None:
                raise ValueError("user_embeddings deve essere fornito per la modalità 'pretrained_only'")
            self._print("Modalità utenti: solo embedding pre-calcolati")
            if not hasattr(self, 'pretrained_user_embeddings'):
                self.register_buffer('pretrained_user_embeddings', torch.tensor(user_embeddings, dtype=torch.float32))
            return self.pretrained_user_embeddings.shape[1]
            
        elif self.user_embedding_mode == 'mixed':
            if user_embeddings is None:
                raise ValueError("user_embeddings deve essere fornito per la modalità 'mixed'")
            self._print("Modalità utenti: embedding misti (addestrabili + pre-calcolati)")
            self.user_embedding = nn.Embedding(self.n_users, self.learnable_embedding_dim)
            if not hasattr(self, 'pretrained_user_embeddings'):
                self.register_buffer('pretrained_user_embeddings', torch.tensor(user_embeddings, dtype=torch.float32))
            
            # Setup fusion components
            pretrained_dim = self.pretrained_user_embeddings.shape[1]
            return self._setup_fusion_components('user', self.learnable_embedding_dim, pretrained_dim)
            
        else:
            raise ValueError(f"Modalità utente non supportata: {self.user_embedding_mode}")

    def _setup_item_embeddings(self, item_embeddings):
        """Setup item embedding configuration based on mode"""
        if self.item_embedding_mode == 'learnable_only':
            self._print("Modalità item: solo embedding addestrabili")
            self.item_embedding = nn.Embedding(self.n_items, self.learnable_embedding_dim)
            return self.learnable_embedding_dim
            
        elif self.item_embedding_mode == 'pretrained_only':
            if item_embeddings is None:
                raise ValueError("item_embeddings deve essere fornito per la modalità 'pretrained_only'")
            self._print("Modalità item: solo embedding pre-calcolati")
            if not hasattr(self, 'pretrained_item_embeddings'):
                self.register_buffer('pretrained_item_embeddings', torch.tensor(item_embeddings, dtype=torch.float32))
            return self.pretrained_item_embeddings.shape[1]
            
        elif self.item_embedding_mode == 'mixed':
            if item_embeddings is None:
                raise ValueError("item_embeddings deve essere fornito per la modalità 'mixed'")
            self._print("Modalità item: embedding misti (addestrabili + pre-calcolati)")
            self.item_embedding = nn.Embedding(self.n_items, self.learnable_embedding_dim)
            if not hasattr(self, 'pretrained_item_embeddings'):
                self.register_buffer('pretrained_item_embeddings', torch.tensor(item_embeddings, dtype=torch.float32))
            
            # Setup fusion components
            pretrained_dim = self.pretrained_item_embeddings.shape[1]
            return self._setup_fusion_components('item', self.learnable_embedding_dim, pretrained_dim)
            
        else:
            raise ValueError(f"Modalità item non supportata: {self.item_embedding_mode}")

    # def _setup_fusion_components(self, entity_type, learnable_dim, pretrained_dim):
    #     """Setup fusion components for mixed embedding mode"""
    #     if self.fusion_strategy == 'concatenate':
    #         self._print(f"Strategia di fusione {entity_type}: concatenazione")
    #         return learnable_dim + pretrained_dim
            
    #     elif self.fusion_strategy == 'add':
    #         if learnable_dim != pretrained_dim:
    #             self._print(f"Strategia di fusione {entity_type}: somma con proiezione")
    #             # Create projection layer to match dimensions
    #             projection_layer = nn.Linear(pretrained_dim, learnable_dim)
    #             setattr(self, f'{entity_type}_pretrained_projection', projection_layer)
    #         else:
    #             self._print(f"Strategia di fusione {entity_type}: somma diretta")
    #         return learnable_dim
            
    #     elif self.fusion_strategy == 'weighted_sum':
    #         self._print(f"Strategia di fusione {entity_type}: somma pesata")
    #         # Create learnable weights
    #         weight_layer = nn.Parameter(torch.tensor([0.5, 0.5]))
    #         setattr(self, f'{entity_type}_fusion_weights', weight_layer)
    #         if learnable_dim != pretrained_dim:
    #             projection_layer = nn.Linear(pretrained_dim, learnable_dim)
    #             setattr(self, f'{entity_type}_pretrained_projection', projection_layer)
    #         return learnable_dim
            
    #     elif self.fusion_strategy == 'gated':
    #         self._print(f"Strategia di fusione {entity_type}: gating")
    #         # Create gating mechanism
    #         gate_dim = learnable_dim + pretrained_dim
    #         gate_layer = nn.Sequential(
    #             nn.Linear(gate_dim, gate_dim // 2),
    #             nn.ReLU(),
    #             nn.Linear(gate_dim // 2, learnable_dim),
    #             nn.Sigmoid()
    #         )
    #         setattr(self, f'{entity_type}_gate', gate_layer)
    #         return learnable_dim
            
    #     else:
    #         raise ValueError(f"Strategia di fusione non supportata: {self.fusion_strategy}")
        
    def _setup_fusion_components(self, entity_type, learnable_dim, pretrained_dim):
        """Setup fusion components for mixed embedding mode"""
        if self.fusion_strategy == 'concatenate':
            self._print(f"Strategia di fusione {entity_type}: concatenazione")
            return learnable_dim + pretrained_dim
            
        elif self.fusion_strategy == 'add':
            if learnable_dim != pretrained_dim:
                self._print(f"Strategia di fusione {entity_type}: somma con proiezione")
                # Create projection layer to match dimensions
                projection_layer = nn.Linear(pretrained_dim, learnable_dim)
                setattr(self, f'{entity_type}_pretrained_projection', projection_layer)
            else:
                self._print(f"Strategia di fusione {entity_type}: somma diretta")
            return learnable_dim
            
        elif self.fusion_strategy == 'weighted_sum':
            self._print(f"Strategia di fusione {entity_type}: somma pesata")
            # Create learnable weights
            weight_layer = nn.Parameter(torch.tensor([0.5, 0.5]))
            setattr(self, f'{entity_type}_fusion_weights', weight_layer)
            if learnable_dim != pretrained_dim:
                projection_layer = nn.Linear(pretrained_dim, learnable_dim)
                setattr(self, f'{entity_type}_pretrained_projection', projection_layer)
            return learnable_dim
            
        elif self.fusion_strategy == 'gated':
            self._print(f"Strategia di fusione {entity_type}: gating")
            # Create gating mechanism
            # The gate input dimension should account for both embeddings
            # After projection, both embeddings will have learnable_dim size
            gate_input_dim = learnable_dim + learnable_dim  # Both will be projected to learnable_dim
            gate_layer = nn.Sequential(
                nn.Linear(gate_input_dim, gate_input_dim // 2),
                nn.ReLU(),
                nn.Linear(gate_input_dim // 2, learnable_dim),
                nn.Sigmoid()
            )
            setattr(self, f'{entity_type}_gate', gate_layer)
            
            # Pre-create projection layer if dimensions don't match
            if learnable_dim != pretrained_dim:
                projection_layer = nn.Linear(pretrained_dim, learnable_dim)
                setattr(self, f'{entity_type}_pretrained_projection_gated', projection_layer)
            
            return learnable_dim
        
        else:
            raise ValueError(f"Strategia di fusione non supportata: {self.fusion_strategy}")


    def _fuse_embeddings(self, learnable_emb, pretrained_emb, entity_type):
        """Fuse learnable and pretrained embeddings based on fusion strategy"""
        if self.fusion_strategy == 'concatenate':
            return torch.cat([learnable_emb, pretrained_emb], dim=-1)
            
        elif self.fusion_strategy == 'add':
            projection_layer = getattr(self, f'{entity_type}_pretrained_projection', None)
            if projection_layer is not None:
                pretrained_emb = projection_layer(pretrained_emb)
            return learnable_emb + pretrained_emb
            
        elif self.fusion_strategy == 'weighted_sum':
            weights = getattr(self, f'{entity_type}_fusion_weights')
            projection_layer = getattr(self, f'{entity_type}_pretrained_projection', None)
            if projection_layer is not None:
                pretrained_emb = projection_layer(pretrained_emb)
            weights_norm = F.softmax(weights, dim=0)
            return weights_norm[0] * learnable_emb + weights_norm[1] * pretrained_emb
            
        elif self.fusion_strategy == 'gated':
            gate = getattr(self, f'{entity_type}_gate')
            combined = torch.cat([learnable_emb, pretrained_emb], dim=-1)
            gate_weights = gate(combined)
            return gate_weights * learnable_emb + (1 - gate_weights) * pretrained_emb

    def _get_user_vector(self, user_input):
        """Get user vector based on embedding mode"""
        if self.user_embedding_mode == 'learnable_only':
            return self.user_embedding(user_input)
            
        elif self.user_embedding_mode == 'pretrained_only':
            return self.pretrained_user_embeddings[user_input]
            
        elif self.user_embedding_mode == 'mixed':
            learnable_emb = self.user_embedding(user_input)
            pretrained_emb = self.pretrained_user_embeddings[user_input]
            return self._fuse_embeddings(learnable_emb, pretrained_emb, 'user')

    def _get_item_vector(self, item_input):
        """Get item vector based on embedding mode"""
        if self.item_embedding_mode == 'learnable_only':
            return self.item_embedding(item_input)
            
        elif self.item_embedding_mode == 'pretrained_only':
            return self.pretrained_item_embeddings[item_input]
            
        elif self.item_embedding_mode == 'mixed':
            learnable_emb = self.item_embedding(item_input)
            pretrained_emb = self.pretrained_item_embeddings[item_input]
            return self._fuse_embeddings(learnable_emb, pretrained_emb, 'item')

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
        # Get user and item vectors based on their respective modes
        user_vector = self._get_user_vector(user_input)
        item_vector = self._get_item_vector(item_input)
        
        # Debug prints
        if self.DEBUG and self.oneprint_forward:
            print(f"DEBUG: Forward pass - User mode: {self.user_embedding_mode}, Item mode: {self.item_embedding_mode}")
            print(f"DEBUG: Fusion strategy: {self.fusion_strategy}")
            print(f"Shape of user_vector: {user_vector.shape}, Shape of item_vector: {item_vector.shape}")
            #self.oneprint_forward = False

        # Pass through towers
        user_tower_output = self.user_tower(user_vector)
        item_tower_output = self.item_tower(item_vector)
        
        if self.DEBUG and self.oneprint_forward:
            print(f"Shape of user_tower_output after tower: {user_tower_output.shape}")
            print(f"Shape of item_tower_output after tower: {item_tower_output.shape}")
            print(f"DEBUG: Forward pass - User mode: {self.user_embedding_mode}, Item mode: {self.item_embedding_mode}")
            print(f"DEBUG: Fusion strategy: {self.fusion_strategy}")
            self.oneprint_forward = False

        # Compute prediction
        prediction_logits = (user_tower_output * item_tower_output).sum(dim=1, keepdim=True)
        
        return prediction_logits

    def _get_initial_embeddings(self, user_ids=None, item_ids=None):
        """Helper per ottenere i vettori iniziali prima delle torri, con stampe di DEBUG."""
        
        if user_ids is not None:
            if self.DEBUG and self.oneprint_get_user_emb:
                print(f"DEBUG: Getting user embeddings with mode: {self.user_embedding_mode}")
                self.oneprint_get_user_emb = False
            return self._get_user_vector(user_ids)
        
        if item_ids is not None:
            if self.DEBUG and self.oneprint_get_item_emb:
                print(f"DEBUG: Getting item embeddings with mode: {self.item_embedding_mode}")
                self.oneprint_get_item_emb = False
            return self._get_item_vector(item_ids)

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