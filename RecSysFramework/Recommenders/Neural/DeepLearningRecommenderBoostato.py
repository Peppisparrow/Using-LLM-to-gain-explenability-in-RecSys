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
        # 1. Prepara i dati una sola volta, inclusa la conversione a DOK
        # .nonzero() restituisce due array: uno per gli indici di riga (utenti) e uno per le colonne (item)
        rows, cols = self.URM_train.nonzero()
        # zip li unisce per creare le coppie (utente, item)
        positive_pairs = list(zip(rows, cols))
        dok_train = self.URM_train.todok()
        n_positive_samples = len(positive_pairs)
        n_collisions = 0
        if num_items is None:
            num_items = self.URM_train.shape[1]
        
        # 2. Mischia i dati per l'epoca corrente
        np.random.shuffle(positive_pairs)

        # 3. Itera sul dataset in batch
        for start_pos in range(0, n_positive_samples, batch_size):
            user_input, item_input, labels = [], [], []
            
            batch_positive_pairs = positive_pairs[start_pos : start_pos + batch_size]

            for u, i in batch_positive_pairs:
                user_input.append(u)
                item_input.append(i)
                labels.append(1)

                for _ in range(num_negatives):
                    j = np.random.randint(num_items)
                    # Usa l'oggetto DOK già creato. Questa operazione ora è velocissima.
                    while (u, j) in dok_train: # <-- PROBLEMA RISOLTO
                        j = np.random.randint(num_items)
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)
            
            user_tensor = torch.tensor(user_input, dtype=torch.long, device=self.device)
            item_tensor = torch.tensor(item_input, dtype=torch.long, device=self.device)
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
            labels_tensor = labels_tensor.reshape((-1, 1))
            yield user_tensor, item_tensor, labels_tensor
    # Sostituisci la vecchia funzione con questa per la massima velocità
    def _data_generator_extreme(self, batch_size, num_negatives=1, num_items=None):
        """
        Generatore di dati ESTREMAMENTE OTTIMIZZATO con NumPy.
        Rimuove il controllo delle collisioni per la massima velocità.
        """
        # 1. Prepara i dati una sola volta
        rows, cols = self.URM_train.nonzero()
        positive_pairs = np.array(list(zip(rows, cols)), dtype=np.int64)
        n_positive_samples = len(positive_pairs)
        
        if num_items is None:
            num_items = self.URM_train.shape[1]
        
        # 2. Mischia i dati
        np.random.shuffle(positive_pairs)

        # 3. Itera sul dataset in batch
        for start_pos in range(0, n_positive_samples, batch_size):
            batch_positive_pairs = positive_pairs[start_pos : start_pos + batch_size]
            n_batch_pos = len(batch_positive_pairs)
            
            # --- Inizio della Magia NumPy ---

            # 4. Pre-alloca gli array per l'intero batch (positivi + negativi)
            batch_total_size = n_batch_pos * (1 + num_negatives)
            user_input = np.empty(batch_total_size, dtype=np.int64)
            item_input = np.empty(batch_total_size, dtype=np.int64)
            labels = np.empty(batch_total_size, dtype=np.float32)

            # 5. Riempi la parte POSITIVA del batch
            user_input[:n_batch_pos] = batch_positive_pairs[:, 0]
            item_input[:n_batch_pos] = batch_positive_pairs[:, 1]
            labels[:n_batch_pos] = 1.0

            # 6. Genera la parte NEGATIVA del batch in modo vettorizzato
            # Ripeti ogni utente 'num_negatives' volte
            user_neg_samples = np.repeat(batch_positive_pairs[:, 0], num_negatives)
            
            # Genera TUTTI i campioni negativi in un colpo solo
            item_neg_samples = np.random.randint(0, num_items, size=n_batch_pos * num_negatives, dtype=np.int64)
            
            # Riempi la parte negativa degli array
            user_input[n_batch_pos:] = user_neg_samples
            item_input[n_batch_pos:] = item_neg_samples
            labels[n_batch_pos:] = 0.0

            # --- Fine della Magia NumPy ---

            # 7. Converti in tensori e restituisci
            user_tensor = torch.from_numpy(user_input).to(self.device)
            item_tensor = torch.from_numpy(item_input).to(self.device)
            labels_tensor = torch.from_numpy(labels).to(self.device).reshape((-1, 1))

            yield user_tensor, item_tensor, labels_tensor
        
    def forward(self, user_input, item_input=None):
        raise NotImplementedError("Forward function not implemented.")


    # def fit(self, epochs=30, batch_size=1024, optimizer=None):
    #     if optimizer is None:
    #         optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    #     # Definisci le attività da profilare
    #     activities = [torch.profiler.ProfilerActivity.CPU]
    #     if self.device.type == 'cuda':
    #         activities.append(torch.profiler.ProfilerActivity.CUDA)
        
    #     # Usa il context manager del profiler
    #     with torch.profiler.profile(
    #         activities=activities,
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #         record_shapes=True # Utile per l'analisi
    #     ) as prof: # Rimosso 'on_trace_ready'
    #         batch_counter = 0
    #         # Esegui un'epoca per raccogliere i dati di profiling
    #         for i in range(1): # Eseguiamo solo un'epoca per il test
    #             data_generator = self._data_generator_fixed(batch_size)
    #             num_batches = math.ceil(self.URM_train.nnz / batch_size)
    #             progress_bar = tqdm(iterable=data_generator, total=num_batches, desc=f"Epoch {i+1}")
                
    #             for user_input, item_input, labels in progress_bar:
    #                 # ... (il tuo ciclo di training non cambia)
    #                 optimizer.zero_grad()
    #                 predictions = self.forward(user_input, item_input)
    #                 loss_fn = torch.nn.BCELoss()
    #                 loss = loss_fn(predictions, labels)
    #                 loss.backward()
    #                 optimizer.step()
                    
    #                 # Notifica al profiler la fine di un passo
    #                 prof.step()
    #                 batch_counter += 1
    #                 if batch_counter >= 50:
    #                     break  # Limita il profiling a 50 batch per velocità

    #     # --- Stampa dei Risultati ---
    #     # Stampa le 100 operazioni che hanno speso più tempo sulla CPU
    #     print("\n--- Top 100 Funzioni per Tempo CPU (Totale) ---")
    #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

    #     # Se hai una GPU NVIDIA, stampa anche le statistiche relative
    #     if self.device.type == 'cuda':
    #         print("\n--- Top 100 Funzioni per Tempo GPU (Totale) ---")
    #         print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))



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

    # def _compute_item_score(self, user_id_array, items_to_compute=None):
    #     step = user_id_array.shape[0]
        
    #     if items_to_compute is None:
    #         items_to_compute = np.arange(self.URM_train.shape[1], dtype=np.int32)
        
    #     predictions = np.empty((step,items_to_compute.shape[0]))
    #     for item in items_to_compute:
    #         with torch.no_grad():
    #             predictions[:, item] = self.forward(
    #                 torch.tensor(user_id_array),
    #                 torch.tensor(
    #                     np.ones(step, dtype=np.int32) * item)
    #                 ).cpu().detach().numpy().ravel()
    #     return predictions

    # def _compute_item_score(self, user_id_array, items_to_compute=None, batch_size=128):
    #     """
    #     Calcola i punteggi in batch per evitare di iterare su un singolo item alla volta.
    #     """
    #     if items_to_compute is None:
    #         items_to_compute = np.arange(self.URM_train.shape[1], dtype=np.int32)

    #     num_users = len(user_id_array)
    #     num_items = len(items_to_compute)
        
    #     # Lista per raccogliere i punteggi dei vari batch
    #     predictions_list = []

    #     # Disattiva il calcolo del gradiente per velocizzare e risparmiare memoria
    #     with torch.no_grad():
    #         # Itera sugli item in "batch" (blocchi)
    #         for start_pos in range(0, num_items, batch_size):
    #             end_pos = min(start_pos + batch_size, num_items)
                
    #             # Seleziona il batch di item corrente
    #             item_batch = items_to_compute[start_pos:end_pos]
    #             num_item_batch = len(item_batch)

    #             # 1. Crea tutte le coppie (utente, item) per il batch corrente
    #             # Ripeti ogni utente per il numero di item nel batch
    #             user_input_batch = np.repeat(user_id_array, num_item_batch)
    #             # Ripeti la lista di item del batch per ogni utente
    #             item_input_batch = np.tile(item_batch, num_users)

    #             # 2. Converti in tensori
    #             user_tensor = torch.tensor(user_input_batch, dtype=torch.long, device=self.device)
    #             item_tensor = torch.tensor(item_input_batch, dtype=torch.long, device=self.device)

    #             # 3. Esegui UNA SOLA chiamata `forward` per l'intero batch di coppie
    #             predictions_batch = self.forward(user_tensor, item_tensor).cpu().numpy().ravel()

    #             # 4. Riformatta l'output e aggiungilo alla lista
    #             predictions_list.append(predictions_batch.reshape(num_users, -1))

    #     # 5. Concatena i risultati di tutti i batch
    #     final_predictions = np.concatenate(predictions_list, axis=1)

    #     return final_predictions
    # Inserisci/Sostituisci questi due metodi nella tua classe base DeepLearningRecommender

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None, return_scores=False, remove_top_pop_flag=False, remove_custom_items_flag=False, custom_items_to_remove=None):
        """
        Versione finale con rimozione degli item visti completamente vettorizzata su GPU.
        """
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # 1. Calcola i punteggi, ottenendo un tensore su GPU
        scores_tensor = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)

        # 2. VETTORIZZA la rimozione degli item già visti
        if remove_seen_flag:
            # Estrai la sottomatrice delle interazioni per gli utenti nel batch
            seen_items_matrix = self.URM_train[user_id_array]
            
            # Trova le coordinate (utente_nel_batch, item_id) degli item visti
            user_indices_in_batch, item_indices_seen = seen_items_matrix.nonzero()

            # Imposta i punteggi a -inf per tutte le coppie in un'unica operazione
            scores_tensor[user_indices_in_batch, item_indices_seen] = -torch.inf

        # 3. Usa torch.topk per l'ordinamento veloce su GPU
        _, top_k_indices = torch.topk(scores_tensor, k=cutoff, dim=1)

        # 4. Sposta solo il risultato finale sulla CPU
        ranking_list = top_k_indices.cpu().numpy().tolist()

        if single_user:
            ranking_list = ranking_list[0]

        if return_scores:
            return ranking_list, scores_tensor.cpu().numpy()
        else:
            return ranking_list

    def _compute_item_score(self, user_id_array, items_to_compute=None, batch_size=32):
        """
        Versione modificata per restituire il tensore dei punteggi direttamente su GPU.
        """
        if items_to_compute is None:
            items_to_compute = np.arange(self.URM_train.shape[1])

        num_users = len(user_id_array)
        num_items = len(items_to_compute)
        
        # Pre-alloca il tensore dei punteggi finali direttamente su GPU
        final_predictions_gpu = torch.empty(num_users, num_items, device=self.device)

        user_tensor_gpu = torch.tensor(user_id_array, dtype=torch.long, device=self.device)
        items_to_compute_gpu = torch.tensor(items_to_compute, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for start_pos in range(0, num_items, batch_size):
                end_pos = min(start_pos + batch_size, num_items)
                item_batch_gpu = items_to_compute_gpu[start_pos:end_pos]
                num_item_batch = len(item_batch_gpu)

                user_input_batch_gpu = user_tensor_gpu.repeat_interleave(num_item_batch)
                item_input_batch_gpu = item_batch_gpu.tile(num_users)

                predictions_batch = self.forward(user_input_batch_gpu, item_input_batch_gpu)
                
                # Riempi la porzione corretta del tensore finale
                final_predictions_gpu[:, start_pos:end_pos] = predictions_batch.reshape(num_users, -1)

        return final_predictions_gpu

    # Dovrai aggiungere anche questo piccolo metodo helper alla classe BaseRecommender
    # per recuperare gli item visti
    def get_user_seen_items(self, user_id):
        # Assumendo che URM_train sia in formato CSR
        return self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id+1]]
    