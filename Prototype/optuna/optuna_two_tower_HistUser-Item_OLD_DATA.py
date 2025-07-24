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
# Defining Recommender
from RecSysFramework.Recommenders.Neural.TwoTowerE import TwoTowerRecommender
from Prototype.data_manager_TEMPORANEO import DataManger
from Prototype.utils.optuna_utils import SaveResults
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
# ---------- CONSTANTS ----------
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
# ---------- /CONSTANTS ----------

def objective_function(trial, URM_train, URM_test, item_embeddings=None, user_embeddings=None):

    epochs = trial.suggest_int("epochs", 1, 40)
    batch_size = trial.suggest_int("batch_size", 512, 4096)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    output = 2 ** trial.suggest_int("input", 4, 10)
    moltiplier = 2 ** trial.suggest_int("multiplier", 0, 4)
    n_layers= trial.suggest_int("n_layers", 2, 8)
    input = output * moltiplier
    layers = np.linspace(input, output, n_layers, dtype=np.int16)
    layers = layers.astype(int)
    print(f"Current layers: {layers}")
    recommender = TwoTowerRecommender(URM_train, URM_train.shape[0], URM_train.shape[1], user_embeddings=user_embeddings, item_embeddings=item_embeddings, layers=layers, verbose=True)
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
    return result_imp

def main():
    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=USER_EMBEDDING_PATH, item_embeddings_path=ITEM_EMBEDDING_PATH)
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    user_embeddings = data_manager.get_user_embeddings() if USER_EMBEDDING_PATH else None
    item_embeddings = data_manager.get_item_embeddings() if ITEM_EMBEDDING_PATH else None
    
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test,
        item_embeddings=item_embeddings,
        user_embeddings=user_embeddings
    )

    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage=f"sqlite:///{DB_PATH}", sampler=optuna.samplers.TPESampler(seed=43))

    save_results = SaveResults(csv_path=BASE_OPTUNA_FOLDER / f"logs/{STUDY_NAME}/trials_results.csv")

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 100)
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Run Optuna study for IALS with fixed alpha")
    parser.add_argument("--study_name", type=str, help="Name of the Optuna study")
    parser.add_argument("--data_path", type=str, help="Path to the dataset with train and test csv files")
    parser.add_argument("--user_embedding_path", type=str, help="Path to the user embeddings file", default=None)
    parser.add_argument("--item_embedding_path", type=str, help="Path to the item embeddings file", default=None)
    parser.add_argument("--db_path", type=str, help="Path to the database file", default="Prototype/optuna/optuna_study.db")
    parser.add_argument("--metric", type=str, help="Metric to optimize", default='map')
    parser.add_argument("--metric_k", type=int, help="K value for the metric", default=10)
    args = parser.parse_args()
    
    
    STUDY_NAME = args.study_name
    DATA_PATH = Path(args.data_path)
    USER_EMBEDDING_PATH = Path(args.user_embedding_path) if args.user_embedding_path else None
    ITEM_EMBEDDING_PATH = Path(args.item_embedding_path) if args.item_embedding_path else None
    
    DB_PATH = args.db_path
    METRIC = args.metric
    METRIC_K = args.metric_k
    
    
    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}")
    print(f"Item embedding path: {ITEM_EMBEDDING_PATH} and database path: {DB_PATH}")
    print(f"Metric: {METRIC} with K={METRIC_K}")
    
    main()