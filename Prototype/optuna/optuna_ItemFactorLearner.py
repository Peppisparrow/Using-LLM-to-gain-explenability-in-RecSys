import os
from functools import partial
from pathlib import Path
from argparse import ArgumentParser

import optuna
import pandas as pd
from implicit.evaluation import ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares

# Defining Recommender
from Prototype.Decoder.ItemFactorLearner_implicit import ImplicitItemFactorLearner
from Prototype.data_manager import DataManger
from Prototype.utils.optuna_utils import SaveResults


# ---------- CONSTANTS ----------
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
# ---------- /CONSTANTS ----------

def objective_function(trial, user_embeddings, URM_train, URM_test):
    
    params = {
        "user_factors": user_embeddings,
        "alpha": trial.suggest_float("alpha", 0.0, 50.0),
        "reg": trial.suggest_float("reg", 1e-5, 1e-1, log=True),
    }

    recommender = ImplicitItemFactorLearner(URM_train)
    recommender.fit(**params)

    fake_model = AlternatingLeastSquares(
        regularization=params["reg"],
        factors=params["user_factors"].shape[1],
        iterations=1,  # We don't need to train the model, just need the structure
        num_threads=1,  # Number of threads can be adjusted based on the environment
        calculate_training_loss=False,  # We don't need training loss for this task
        use_gpu=False,  # Use GPU for training

    )
    fake_model.item_factors = recommender.ITEM_factors
    fake_model.user_factors = recommender.USER_factors
    
    fake_model = fake_model.to_gpu()

    result = ranking_metrics_at_k(
        fake_model,
        URM_train,
        URM_test,
        K=METRIC_K,
    )[METRIC]
    
    print("Current {} = {:.4f} with parameters {}".format(METRIC, result, params))
    
    return result


def main():
    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=USER_EMBEDDING_PATH)
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    user_embeddings = data_manager.get_user_embeddings()
    
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test,
        user_embeddings=user_embeddings
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage=f"sqlite:///{DB_PATH}")
            
    save_results = SaveResults(csv_path=BASE_OPTUNA_FOLDER / f"logs/{STUDY_NAME}/trials_results.csv")

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 100)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run Optuna study for ItemFactorLearner")
    parser.add_argument("--study_name", type=str, help="Name of the Optuna study")
    parser.add_argument("--data_path", type=str, help="Path to the dataset with train and test csv files")
    parser.add_argument("--user_embedding_path", type=str, help="Path to the user embeddings file")
    parser.add_argument("--db_path", type=str, help="Path to the database file", default="Prototype/optuna/optuna_study.db")
    parser.add_argument("--metric", type=str, help="Metric to optimize", default='map')
    parser.add_argument("--metric_k", type=int, help="K value for the metric", default=10)
    args = parser.parse_args()
    
    
    STUDY_NAME = args.study_name
    DATA_PATH = Path(args.data_path)
    USER_EMBEDDING_PATH = Path(args.user_embedding_path)
    DB_PATH = args.db_path
    
    METRIC = args.metric
    METRIC_K = args.metric_k
    
    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}")
    print(f"Database path: {DB_PATH}")
    print(f"Evaluation with implicit backend using metric: {METRIC} at K: {METRIC_K}")
    
    main()