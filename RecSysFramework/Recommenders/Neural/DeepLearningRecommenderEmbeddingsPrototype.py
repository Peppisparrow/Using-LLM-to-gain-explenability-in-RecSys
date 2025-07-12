from RecSysFramework.Recommenders.BaseRecommender import BaseRecommender
import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
# import torch.profiler # <-- 1. Importa il profiler

class DeepLearningRecommender(nn.Module, BaseRecommender):

    def __init__(self, URM_train, verbose=True):
        super().__init__()
        BaseRecommender.__init__(self, URM_train, verbose)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Questo serve se sposti il codice su un Mac
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def _data_generator_fixed(self, batch_size, num_negatives=1, num_items=None):
        """
        Generatore di dati OTTIMIZZATO. 
        La conversione .todok() viene eseguita una sola volta.
        """
        rows, cols = self.URM_train.nonzero()
        positive_pairs = list(zip(rows, cols))
        dok_train = self.URM_train.todok()
        n_positive_samples = len(positive_pairs)
        n_collisions = 0
        if num_items is None:
            num_items = self.URM_train.shape[1]
        
        np.random.shuffle(positive_pairs)

        for start_pos in range(0, n_positive_samples, batch_size):
            user_input, item_input, labels = [], [], []
            
            batch_positive_pairs = positive_pairs[start_pos : start_pos + batch_size]

            for u, i in batch_positive_pairs:
                user_input.append(u)
                item_input.append(i)
                labels.append(1)

                for _ in range(num_negatives):
                    j = np.random.randint(num_items)
                    while (u, j) in dok_train: 
                        j = np.random.randint(num_items)
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)
            
            user_tensor = torch.tensor(user_input, dtype=torch.long, device=self.device)
            item_tensor = torch.tensor(item_input, dtype=torch.long, device=self.device)
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
            labels_tensor = labels_tensor.reshape((-1, 1))
            yield user_tensor, item_tensor, labels_tensor
        
    def forward(self, user_input, item_input=None, user_embedding=None, item_embedding=None):
        raise NotImplementedError("Forward function not implemented.")


    def fit(self, epochs=30, batch_size=1024, optimizer=None, user_embeddings=None, item_embeddings=None):
        # If pre-computed embeddings are passed, ensure they are tensors on the correct device
        if user_embeddings is not None:
            user_embeddings = torch.tensor(user_embeddings, dtype=torch.float32, device=self.device)
        if item_embeddings is not None:
            item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32, device=self.device)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    
        num_batches = math.ceil(self.URM_train.nnz / batch_size)

        for i in range(epochs):
            self._print(f"Epoch {i+1}/{epochs} start")
            
            data_generator = self._data_generator_fixed(batch_size) 
            
            progress_bar = tqdm(
                iterable=data_generator, 
                total=num_batches,
                desc=f"Epoch {i+1}/{epochs}"
            )
            
            for user_input_ids, item_input_ids, labels in progress_bar:
                optimizer.zero_grad()

                # --- MODIFIED LOGIC: Prepare arguments for the forward pass ---
                forward_args = {}
                if hasattr(self, 'mlp_embedding_user') and self.mlp_embedding_user is not None:
                    forward_args['user_input'] = user_input_ids
                else:
                    forward_args['user_embedding'] = user_embeddings[user_input_ids]

                if hasattr(self, 'mlp_embedding_item') and self.mlp_embedding_item is not None:
                    forward_args['item_input'] = item_input_ids
                else:
                    forward_args['item_embedding'] = item_embeddings[item_input_ids]
                # --- END MODIFIED LOGIC ---

                predictions = self.forward(**forward_args)
                
                loss_fn = torch.nn.BCELoss()
                loss = loss_fn(predictions, labels)
                
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            self._print(f"Epoch {i+1}/{epochs} finished. Last batch Loss: {loss.item():.4f}\n")

    def _compute_item_score(self, user_id_array, items_to_compute=None, batch_size=4096, user_embeddings=None, item_embeddings=None):
        
        # If pre-computed embeddings are passed, ensure they are tensors on the correct device
        if user_embeddings is not None:
            user_embeddings = torch.tensor(user_embeddings, dtype=torch.float32, device=self.device)
        if item_embeddings is not None:
            item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32, device=self.device)

        if items_to_compute is None:
            items_to_compute = np.arange(self.URM_train.shape[1])

        num_users = len(user_id_array)
        num_items = len(items_to_compute)
        
        predictions_list = []
        
        # Prepare user-side input once
        if hasattr(self, 'mlp_embedding_user') and self.mlp_embedding_user is not None:
            user_tensor_gpu = torch.tensor(user_id_array, dtype=torch.long, device=self.device)
        else:
            # If using pre-computed embeddings, look them up
            user_tensor_gpu = user_embeddings[user_id_array]

        items_to_compute_gpu_ids = torch.tensor(items_to_compute, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for start_pos in range(0, num_items, batch_size):
                end_pos = min(start_pos + batch_size, num_items)
                
                item_batch_ids_gpu = items_to_compute_gpu_ids[start_pos:end_pos]
                num_item_batch = len(item_batch_ids_gpu)
                
                # Expand user tensor to match the number of items in the batch
                user_input_batch_gpu = user_tensor_gpu.repeat_interleave(num_item_batch, dim=0)

                # --- MODIFIED LOGIC: Prepare item-side arguments ---
                forward_args = {}
                if hasattr(self, 'mlp_embedding_user') and self.mlp_embedding_user is not None:
                    forward_args['user_input'] = user_input_batch_gpu
                else:
                    forward_args['user_embedding'] = user_input_batch_gpu

                if hasattr(self, 'mlp_embedding_item') and self.mlp_embedding_item is not None:
                    # Tile item IDs for each user
                    forward_args['item_input'] = item_batch_ids_gpu.tile(num_users)
                else:
                    # If using pre-computed embeddings, look them up and then tile
                    item_embeddings_batch = item_embeddings[item_batch_ids_gpu]
                    forward_args['item_embedding'] = item_embeddings_batch.repeat(num_users, 1)
                # --- END MODIFIED LOGIC ---

                predictions_batch = self.forward(**forward_args)

                predictions_list.append(predictions_batch.cpu().numpy().reshape(num_users, -1))

        final_predictions = np.concatenate(predictions_list, axis=1)

        return final_predictions









































    def fit(self, epochs=30, batch_size=1024, optimizer=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    
        # 1. Calcola il numero totale di batch per la barra di avanzamento
        #    self.URM_train.nnz è il numero totale di interazioni positive
        num_batches = math.ceil(self.URM_train.nnz / batch_size)

        for i in range(epochs):
            self._print(f"Epoch {i+1}/{epochs} start")
            
            # Chiama il generatore (assumendo che usi la versione corretta _data_generator_fixed)
            data_generator = self._data_generator_fixed(batch_size) 
            
            # 2. Crea la barra di avanzamento con tqdm
            progress_bar = tqdm(
                iterable=data_generator, 
                total=num_batches,
                desc=f"Epoch {i+1}/{epochs}"
            )
            
            # 3. Itera usando la progress_bar
            for user_input, item_input, labels in progress_bar:
                optimizer.zero_grad()
                predictions = self.forward(user_input, item_input)
                
                loss_fn = torch.nn.BCELoss()
                loss = loss_fn(predictions, labels)
                
                loss.backward()
                optimizer.step()

                # 4. (Opzionale ma utile) Aggiorna la barra con la loss corrente
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            self._print(f"Epoch {i+1}/{epochs} finished. Last batch Loss: {loss.item():.4f}\n")

    def _compute_item_score(self, user_id_array, items_to_compute=None, batch_size=4096):
        """
        Versione ultra-ottimizzata che esegue il pairing utente-item direttamente su GPU.
        """
        if items_to_compute is None:
            items_to_compute = np.arange(self.URM_train.shape[1])

        num_users = len(user_id_array)
        num_items = len(items_to_compute)
        
        predictions_list = []

        # 1. Sposta i tensori di base sulla GPU UNA SOLA VOLTA all'inizio
        user_tensor_gpu = torch.tensor(user_id_array, dtype=torch.long, device=self.device)
        items_to_compute_gpu = torch.tensor(items_to_compute, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for start_pos in range(0, num_items, batch_size):
                end_pos = min(start_pos + batch_size, num_items)
                
                # Seleziona il batch di item (già sulla GPU)
                item_batch_gpu = items_to_compute_gpu[start_pos:end_pos]
                num_item_batch = len(item_batch_gpu)

                # 2. Esegui il pairing (repeat/tile) direttamente su GPU
                #    Usa le funzioni PyTorch equivalenti a quelle NumPy
                user_input_batch_gpu = user_tensor_gpu.repeat_interleave(num_item_batch)
                item_input_batch_gpu = item_batch_gpu.tile(num_users)

                # 3. I tensori sono già sulla GPU, esegui il forward pass
                predictions_batch = self.forward(user_input_batch_gpu, item_input_batch_gpu)

                # 4. Sposta il risultato sulla CPU solo alla fine
                predictions_list.append(predictions_batch.cpu().numpy().reshape(num_users, -1))

        # 5. Concatena i risultati
        final_predictions = np.concatenate(predictions_list, axis=1)

        return final_predictions
    