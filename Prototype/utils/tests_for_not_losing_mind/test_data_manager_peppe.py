# File: test_data_manager.py
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from itertools import product

# Importa la classe dal nuovo file 'data_manager.py'
from data_manager_peppe import DataManger

class TestDataManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_path = self.temp_dir
        self.data_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_dummy_files(self, users, items, users_npz=None, items_npz=None, embed_dim=5, interaction_pattern='one_to_one'):
        if interaction_pattern == 'one_to_one':
            assert len(users) == len(items), "Per 'one_to_one', le liste devono avere la stessa lunghezza."
            user_ids_list, item_ids_list = users, items
        elif interaction_pattern == 'all_to_all':
            interactions = list(product(users, items))
            user_ids_list = [i[0] for i in interactions]
            item_ids_list = [i[1] for i in interactions]
        else:
            raise ValueError("interaction_pattern non valido.")

        train_df = pd.DataFrame({'user_id': user_ids_list, 'app_id': item_ids_list})
        test_df = pd.DataFrame({'user_id': [users[0]], 'app_id': [items[0]]})
        train_df.to_csv(self.data_path / 'train_recommendations.csv', index=False)
        test_df.to_csv(self.data_path / 'test_recommendations.csv', index=False)

        # La creazione degli embedding è ora opzionale
        user_path = None
        if users_npz:
            user_vectors = [np.full(embed_dim, uid, dtype=np.float32) for uid in users_npz]
            user_path = self.temp_dir / 'user_embeddings.npz'
            np.savez(user_path, embeddings=np.array(user_vectors), user_ids=np.array(users_npz, dtype=str))

        item_path = None
        if items_npz:
            item_vectors = [np.full(embed_dim, iid, dtype=np.float32) for iid in items_npz]
            item_path = self.temp_dir / 'item_embeddings.npz'
            np.savez(item_path, embeddings=np.array(item_vectors), app_id=np.array(items_npz, dtype=str))

        return user_path, item_path

    def test_successful_loading_and_alignment(self):
        users = [10, 20]; items = [101, 102]; embed_dim = 5
        user_path, item_path = self._create_dummy_files(
            users, items, users_npz=users, items_npz=items, embed_dim=embed_dim, interaction_pattern='all_to_all'
        )
        dm = DataManger(user_embedding_path=user_path, data_path=self.data_path, item_embeddings_path=item_path)
        self.assertEqual(dm.get_user_embeddings().shape, (2, embed_dim))
        idx_user_20 = dm.get_user_mapping()[20]
        expected_user_vec = np.full(embed_dim, 20, dtype=np.float32)
        np.testing.assert_array_equal(dm.get_user_embeddings()[idx_user_20], expected_user_vec)

    def test_extra_users_in_embeddings_are_filtered(self):
        users_in_npz = [10, 20, 40, 80, 100] 
        users_in_csv = [80, 20, 40]
        items_in_csv = [101, 102, 103]
        
        user_path, item_path = self._create_dummy_files(
            users_in_csv, items_in_csv, users_npz=users_in_npz, items_npz=items_in_csv
        )
        dm = DataManger(user_embedding_path=user_path, data_path=self.data_path, item_embeddings_path=item_path)
        self.assertEqual(dm.get_user_embeddings().shape[0], 3)
        self.assertEqual(len(dm.get_user_mapping()), 3)
        idx_user_40 = dm.get_user_mapping()[40]
        self.assertEqual(idx_user_40, 1)
        loaded_embedding = dm.get_user_embeddings()[idx_user_40]
        expected_embedding = np.full(5, 40, dtype=np.float32)
        np.testing.assert_array_equal(loaded_embedding, expected_embedding)

    def test_extra_users_in_embeddings_are_filtered2(self):
        users_in_npz = [10, 20, 40, 80, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400] 
        users_in_csv = [80, 20, 40, 200, 350]
        items_in_csv = [101, 102, 103, 104, 105]
        
        user_path, item_path = self._create_dummy_files(
            users_in_csv, items_in_csv, users_npz=users_in_npz, items_npz=items_in_csv
        )
        dm = DataManger(user_embedding_path=user_path, data_path=self.data_path, item_embeddings_path=item_path)
        self.assertEqual(dm.get_user_embeddings().shape[0], 5)
        self.assertEqual(len(dm.get_user_mapping()), 5)
        idx_user_40 = dm.get_user_mapping()[40]
        self.assertEqual(idx_user_40, 1)
        loaded_embedding = dm.get_user_embeddings()[idx_user_40]
        expected_embedding = np.full(5, 40, dtype=np.float32)
        np.testing.assert_array_equal(loaded_embedding, expected_embedding)

        idx_user_200 = dm.get_user_mapping()[200]
        loaded_embedding = dm.get_user_embeddings()[idx_user_200]
        expected_embedding = np.full(5, 200, dtype=np.float32)
        np.testing.assert_array_equal(loaded_embedding, expected_embedding)
    def test_extra_users_and_items_in_embeddings_are_filtered(self):
        """
        🧪 Test con utenti e item extra negli embedding rispetto alla URM.
        Verifica che gli elementi extra vengano ignorati e che il mapping sia corretto per entrambi.
        """
        # --- SETUP DEI DATI ---
        # Definiamo un universo di embedding più grande...
        users_in_npz = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        items_in_npz = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        
        # ...di cui solo un sottoinsieme ha interazioni nei file CSV.
        # Gli ID sono volutamente non in ordine per testare il sorting.
        users_in_csv = [80, 20, 100, 100]
        items_in_csv = [108, 101, 105, 102]
        
        embed_dim = 2 # La dimensione degli embedding deterministici

        # Creiamo i file fittizi
        user_path, item_path = self._create_dummy_files(
            users_in_csv, items_in_csv, 
            users_npz=users_in_npz, items_npz=items_in_npz,
            embed_dim=embed_dim
        )
        
        # --- ESECUZIONE ---
        # Il caricamento deve avere successo
        dm = DataManger(user_embedding_path=user_path, data_path=self.data_path, item_embeddings_path=item_path)
        # --- VERIFICHE PER GLI UTENTI ---
        print("\nVerifying Users...")
        # 1. Le matrici finali devono contenere solo i 3 utenti dei CSV
        self.assertEqual(dm.get_URM_train().shape[0], 3, "Il numero di utenti nella URM non è corretto.")
        self.assertEqual(dm.get_user_embeddings().shape[0], 3, "Il numero di user embeddings non è corretto.")
        self.assertEqual(len(dm.get_user_mapping()), 3, "La mappa degli utenti non ha la dimensione corretta.")

        # 2. Verifichiamo che il mapping e l'allineamento siano corretti
        #    Gli utenti finali sono ordinati: [20, 80, 100]
        user_map = dm.get_user_mapping()
        self.assertEqual(user_map[20], 0)
        self.assertEqual(user_map[80], 1)
        self.assertEqual(user_map[100], 2)
        
        # Controlliamo il vettore per l'utente 80 (indice 1)
        loaded_user_embedding = dm.get_user_embeddings()[user_map[80]]
        expected_user_embedding = np.full(embed_dim, 80, dtype=np.float32)
        np.testing.assert_array_equal(loaded_user_embedding, expected_user_embedding, "L'embedding per l'utente 80 non è allineato.")
        print("User verification successful.")
        
        # --- VERIFICHE PER GLI ITEM (NUOVA PARTE) ---
        print("\nVerifying Items...")
        # 1. Le matrici finali devono contenere solo i 3 item dei CSV
        self.assertEqual(dm.get_URM_train().shape[1], 4, "Il numero di item nella URM non è corretto.")
        self.assertEqual(dm.get_item_embeddings().shape[0], 4, "Il numero di item embeddings non è corretto.")
        self.assertEqual(len(dm.get_item_mapping()), 4, "La mappa degli item non ha la dimensione corretta.")
        
        # 2. Verifichiamo che il mapping e l'allineamento siano corretti
        #    Gli item finali sono ordinati: [101, 105, 108]
        item_map = dm.get_item_mapping()
        self.assertEqual(item_map[101], 0)
        self.assertEqual(item_map[102], 1)
        self.assertEqual(item_map[105], 2)
        self.assertEqual(item_map[108], 3)
        
        # Controlliamo il vettore per l'item 105 (indice 1)
        loaded_item_embedding = dm.get_item_embeddings()[item_map[105]]
        expected_item_embedding = np.full(embed_dim, 105, dtype=np.float32)
        np.testing.assert_array_equal(loaded_item_embedding, expected_item_embedding, "L'embedding per l'item 105 non è allineato.")
        print("Item verification successful.")

        # --- STAMPA FINALE DELLE DIMENSIONI ---
        print(f'\nFinal URM_train shape: {dm.get_URM_train().shape}')
        print(f'Final User embeddings shape: {dm.get_user_embeddings().shape}')
        print(f'Final Item embeddings shape: {dm.get_item_embeddings().shape}')

    def test_raises_error_on_missing_user(self):
        users_csv = [10, 99]; items_csv = [101, 102]; users_npz = [10]
        user_path, item_path = self._create_dummy_files(
            users_csv, items_csv, users_npz=users_npz, items_npz=items_csv
        )
        with self.assertRaisesRegex(ValueError, "Missing embeddings for 1 users"):
            DataManger(user_embedding_path=user_path, data_path=self.data_path, item_embeddings_path=item_path)

    def test_no_user_embeddings_with_item_embeddings(self):
        """🧪 Test di funzionamento senza user_embeddings ma con item_embeddings."""
        users_csv = [10, 20]
        items_csv = [101, 102]
        # Non creiamo un file per gli user (users_npz=None)
        _, item_path = self._create_dummy_files(
            users_csv, items_csv, users_npz=None, items_npz=items_csv
        )
        
        # Inizializziamo il DataManager senza passare lo user_embedding_path
        dm = DataManger(data_path=self.data_path, item_embeddings_path=item_path)

        # Verifichiamo che gli user embeddings siano None e gli altri dati corretti
        self.assertIsNone(dm.get_user_embeddings(), "User embeddings dovrebbe essere None.")
        self.assertIsNotNone(dm.get_item_embeddings(), "Item embeddings non dovrebbe essere None.")
        self.assertEqual(dm.get_URM_train().shape, (2, 2), "La forma della URM non è corretta.")
        self.assertEqual(dm.get_item_embeddings().shape[0], 2, "Il numero di item embeddings non è corretto.")
        self.assertEqual(len(dm.get_user_mapping()), 2, "La mappa degli utenti non è corretta.")

    def test_no_embeddings_at_all(self):
        """🧪 Test di funzionamento senza nessun embedding fornito."""
        users_csv = [55, 66, 77]
        items_csv = [1, 2, 3]
        
        # Non creiamo nessun file di embedding (users_npz=None, items_npz=None)
        _, _ = self._create_dummy_files(
            users_csv, items_csv, users_npz=None, items_npz=None
        )
        
        # Inizializziamo il DataManager senza passare nessun path di embedding
        dm = DataManger(data_path=self.data_path)
        
        # Verifichiamo che entrambi gli embedding siano None e che la URM sia corretta
        self.assertIsNone(dm.get_user_embeddings(), "User embeddings dovrebbe essere None.")
        self.assertIsNone(dm.get_item_embeddings(), "Item embeddings dovrebbe essere None.")
        self.assertEqual(dm.get_URM_train().shape, (3, 3), "La forma della URM non è corretta.")
        self.assertEqual(len(dm.get_user_mapping()), 3, "La mappa degli utenti non è corretta.")
        self.assertEqual(len(dm.get_item_mapping()), 3, "La mappa degli item non è corretta.")

if __name__ == '__main__':
    unittest.main()