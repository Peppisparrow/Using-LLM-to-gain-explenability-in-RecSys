from RecSysFramework.Recommenders.BaseRecommender import BaseRecommender
import torch
import torch.nn as nn
import numpy as np
class DeepLearningRecommender(nn.Module, BaseRecommender):

    def __init__(self, URM_train, verbose=True):
        super().__init__()
        BaseRecommender.__init__(self, URM_train, verbose)
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def _data_generator(self, batch_size, num_negatives=1, num_items=None):
        user_input, item_input, labels = [], [], []
        dok_train = self.URM_train.todok() # <- Dictionary representation of a sparse matrix: allows us to check existing interactions as key-value pairs
        if num_items is None : num_items = self.URM_train.shape[1]

        self.batch_counter = 0
        start = self.batch_counter
        stop = min(self.batch_counter + batch_size, len(dok_train.keys()))
        for (u,i) in dok_train[start:stop].keys():
            # positive interaction
            user_input.append(u)
            item_input.append(i)
            labels.append(1) # <- (Implicit ratings)
            # negative interactions
            for t in range(num_negatives): # <- num_negatives is a hyperparameter
                # randomly select an interaction; check if negative
                j = np.random.randint(num_items)
                while (u,j) in dok_train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        self.batch_counter += 1
        
        user_input = torch.tensor(user_input, dtype=torch.int32, device=self.device)
        item_input = torch.tensor(item_input, dtype=torch.int32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.int32, device=self.device)
        labels = labels.reshape((labels.shape[0],1))
        yield user_input, item_input, labels
    
    def forward(self, user_input, item_input=None):
        raise NotImplementedError("Forward function not implemented.")

    def fit(self, epochs=30, batch_size=1024, optimizer=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001) # <- The optimizer can be (additionally) considered as a hyperparameter
        for i in range(epochs):
            for user_input, item_input, labels in self._data_generator(batch_size):
                self._print("Epoch start")
                optimizer.zero_grad()
                predictions = self.forward(user_input, item_input)
                loss = torch.nn.BCELoss().to(self.device) # <- The loss function can be (additionally) considered as a hyperparameter
                loss = loss(predictions, labels.float())
                loss.backward()
                optimizer.step()
            self._print("Epoch {} finished. Loss: {}".format(i, loss.item()))

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

    def _compute_item_score(self, user_id_array, items_to_compute=None, batch_size=2048):
        """
        Calcola i punteggi in batch per evitare di iterare su un singolo item alla volta.
        """
        if items_to_compute is None:
            items_to_compute = np.arange(self.URM_train.shape[1], dtype=np.int32)

        num_users = len(user_id_array)
        num_items = len(items_to_compute)
        
        # Lista per raccogliere i punteggi dei vari batch
        predictions_list = []

        # Disattiva il calcolo del gradiente per velocizzare e risparmiare memoria
        with torch.no_grad():
            # Itera sugli item in "batch" (blocchi)
            for start_pos in range(0, num_items, batch_size):
                end_pos = min(start_pos + batch_size, num_items)
                
                # Seleziona il batch di item corrente
                item_batch = items_to_compute[start_pos:end_pos]
                num_item_batch = len(item_batch)

                # 1. Crea tutte le coppie (utente, item) per il batch corrente
                # Ripeti ogni utente per il numero di item nel batch
                user_input_batch = np.repeat(user_id_array, num_item_batch)
                # Ripeti la lista di item del batch per ogni utente
                item_input_batch = np.tile(item_batch, num_users)

                # 2. Converti in tensori
                user_tensor = torch.tensor(user_input_batch, dtype=torch.long, device=self.device)
                item_tensor = torch.tensor(item_input_batch, dtype=torch.long, device=self.device)

                # 3. Esegui UNA SOLA chiamata `forward` per l'intero batch di coppie
                predictions_batch = self.forward(user_tensor, item_tensor).cpu().numpy().ravel()

                # 4. Riformatta l'output e aggiungilo alla lista
                predictions_list.append(predictions_batch.reshape(num_users, -1))

        # 5. Concatena i risultati di tutti i batch
        final_predictions = np.concatenate(predictions_list, axis=1)

        return final_predictions
        