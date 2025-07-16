import os
from functools import partial
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import torch
from argparse import ArgumentParser

from implicit.evaluation import ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares

from RecSysFramework.Recommenders.Neural.TwoTowerHistoryTrans import TwoTowerRecommender
from Prototype.data_manager_history import DataManger
from Prototype.utils.optuna_utils import SaveResults
# import cProfile
# import pstats
# import io
# Current layers: [256 192 128  64   1]
# Current parameters: epochs=10, batch_size=2048, learning_rate=0.001, weight_decay=1e-05, layers=[256 192 128  64   1]
# ---------- CONSTANTS ----------
# METRIC = 'MAP_MIN_DEN'
# METRIC_K = 10
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
# STUDY_NAME = "Provailnuovobimbo2"
# DATA_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small')
# USER_EMBEDDING_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small/game_embeddings/game_embeddings_t5_MATRIX_USER.pkl')

# ---------- /CONSTANTS ----------

def objective_function(trial, URM_train, URM_test, user_embeddings=None):

    epochs = trial.suggest_int("epochs", 5, 40)
    batch_size = trial.suggest_int("batch_size", 512, 4096)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    input = 2 ** trial.suggest_int("input", 4, 7)
    n_layers= trial.suggest_int("n_layers", 2, 5)
    # output = 1
    output = 32
    input = 512
    n_layers = 6
    batch_size = 256
    layers = np.linspace(input, output, n_layers, dtype=np.int16)
    layers = layers.astype(int)
    print(f"Current layers: {layers}")
    recommender = TwoTowerRecommender(URM_train, URM_train.shape[0], URM_train.shape[1], user_embeddings=user_embeddings,layers=layers, verbose=True)
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
    
    

    result_imp = ranking_metrics_at_k(
        fake_model,
        URM_train,
        URM_test,
        K=METRIC_K,
    )[METRIC] # like map
    
    print(f"{METRIC}@{METRIC_K} from implicit: {result_imp:.6f}")
    return result_imp

def main():
    data_manager = DataManger(data_path=DATA_PATH, user_history_path=USER_EMBEDDING_PATH)
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    user_embeddings = data_manager.get_user_history_embeddings()
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test,
        user_embeddings=user_embeddings
    )

    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage=f"sqlite:///{DB_PATH}", sampler=optuna.samplers.TPESampler(seed=43))

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