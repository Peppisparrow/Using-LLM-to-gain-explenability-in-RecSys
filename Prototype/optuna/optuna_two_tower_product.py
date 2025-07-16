from argparse import ArgumentParser
import os
from functools import partial
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import torch

from implicit.evaluation import ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares

from RecSysFramework.Recommenders.Neural.TwoTower import TwoTowerRecommender
from Prototype.data_manager import DataManger
from Prototype.utils.optuna_utils import SaveResults

# ---------- CONSTANTS ----------
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
# STUDY_NAME = "Provailnuovobimbo"
# DATA_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small')
# USER_EMBEDDING_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small/user_embeddings_compressed_t5.npz')

def calculate_map_at_k_manual(model, urm_train, urm_test, k):
        """
        Calcola manualmente la MAP@K, replicando la logica di un valutatore standard.
        """
        # Ottieni gli utenti che hanno interazioni nel set di test
        users_in_test = np.unique(urm_test.nonzero()[0])
        
        average_precisions = []
        
        for user_id in users_in_test:
            # Ground truth: gli item con cui l'utente ha interagito nel test set
            ground_truth_items = urm_test[user_id].indices
            if not len(ground_truth_items):
                continue

            # Ottieni le raccomandazioni dal modello
            # `filter_already_liked_items=True` è FONDAMENTALE e corrisponde a passare
            # URM_train alla funzione `ranking_metrics_at_k`.
            recommended_items_ids, _ = model.recommend(
                userid=user_id,
                user_items=urm_train[user_id],
                N=k,
                filter_already_liked_items=True
            )
            
            # Calcola l'Average Precision (AP@k) per questo utente
            hits = 0
            precision_sum = 0.0
            
            for i, recommended_id in enumerate(recommended_items_ids):
                if recommended_id in ground_truth_items:
                    hits += 1
                    precision_at_i = hits / (i + 1)
                    precision_sum += precision_at_i
            
            # Il denominatore per AP è min(k, |ground_truth|)
            denominator = min(k, len(ground_truth_items))
            ap_at_k = precision_sum / denominator if denominator > 0 else 0.0
            average_precisions.append(ap_at_k)

        # La MAP@k è la media delle AP@k su tutti gli utenti del test set
        return np.mean(average_precisions) if average_precisions else 0.0
# ---------- /CONSTANTS ----------

def objective_function(trial, URM_train, URM_test):

    epochs = trial.suggest_int("epochs", 5, 40)
    batch_size = trial.suggest_int("batch_size", 512, 4096)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    input = 2 ** trial.suggest_int("input", 4, 7)
    n_layers= trial.suggest_int("n_layers", 2, 5)
    # output = 1
    output = 1
    input = 1
    n_layers = 2
    batch_size = 4048
    layers = np.linspace(input, output, n_layers, dtype=np.int16)
    layers = layers.astype(int)
    print(f"Current layers: {layers}")
    recommender = TwoTowerRecommender(URM_train, URM_train.shape[0], URM_train.shape[1], layers=layers, verbose=True)
    print(f"Current parameters: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}, layers={layers}")
    optimizer = torch.optim.AdamW(params=recommender.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    print("Optimizer initialized.")
    recommender = torch.compile(recommender)
    recommender.fit(epochs=epochs, batch_size=batch_size, optimizer=optimizer)
    print("Recommender fitted.")
    recommender.compute_all_embeddings(batch_size=batch_size)


    fake_model = AlternatingLeastSquares(
        regularization=0.1,
        factors=recommender.ITEM_factors.shape[1],
        iterations=1,  # We don't need to train the model, just need the structure
        num_threads=1,  # Number of threads can be adjusted based on the environment
        calculate_training_loss=False,  # We don't need training loss for this task
        use_gpu=False,  # Use GPU for training

    )
    fake_model.item_factors = recommender.ITEM_factors
    fake_model.user_factors = recommender.USER_factors
    
    fake_model = fake_model.to_gpu()

    result_imp = ranking_metrics_at_k(
        fake_model,
        URM_train,
        URM_test,
        K=METRIC_K,
    )[METRIC]
    
    print(f"{METRIC}@{METRIC_K} from implicit: {result_imp:.6f}")
    
    
def main():
    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=USER_EMBEDDING_PATH)
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage="sqlite:///Prototype/optuna/optuna_study.db", sampler=optuna.samplers.TPESampler(seed=43))
            
    save_results = SaveResults(csv_path=BASE_OPTUNA_FOLDER / f"logs/{STUDY_NAME}/trials_results.csv")

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 1)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run Optuna study for ImplicitUserFactorLearner")
    parser.add_argument("--study_name", type=str, help="Name of the Optuna study")
    parser.add_argument("--data_path", type=str, help="Path to the dataset with train and test csv files")
    parser.add_argument("--user_embedding_path", type=str, help="Path to the user embeddings file")
    parser.add_argument("--item_embedding_path", type=str, help="Path to the item embeddings file")
    parser.add_argument("--db_path", type=str, help="Path to the database file", default="Prototype/optuna/optuna_study.db")
    parser.add_argument("--metric", type=str, help="Metric to optimize", default='map')
    parser.add_argument("--metric_k", type=int, help="K value for the metric", default=10)
    args = parser.parse_args()
    
    
    STUDY_NAME = args.study_name
    DATA_PATH = Path(args.data_path)
    USER_EMBEDDING_PATH = Path(args.user_embedding_path)
    ITEM_EMBEDDING_PATH = Path(args.item_embedding_path)
    DB_PATH = args.db_path

    METRIC = args.metric
    METRIC_K = args.metric_k

    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}")
    print(f"Item embedding path: {ITEM_EMBEDDING_PATH} and database path: {DB_PATH}")
    print(f"Evaluation with implicit backend using metric: {METRIC} at K: {METRIC_K}")
    main()