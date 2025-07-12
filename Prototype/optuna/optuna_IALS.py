import os
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
from implicit.evaluation import ranking_metrics_at_k

# Defining Recommender
from RecSysFramework.Recommenders.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from Prototype.data_manager import DataManger
from Prototype.utils.optuna_utils import SaveResults


# ---------- CONSTANTS ----------
METRIC = 'map'
METRIC_K = 10
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
STUDY_NAME = "IALS_STUDY_MAP"
DATA_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small')
USER_EMBEDDING_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small/user_embeddings_compressed.npz')
# ---------- /CONSTANTS ----------

def objective_function(trial, URM_train, URM_test):
    
    params = {
        "iterations": trial.suggest_int("iterations", 1, 120),
        "factors": trial.suggest_int("num_factors", 10, 5200),
        "regularization": trial.suggest_float("regularization", 1e-5, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 0.0, 50.0),
        "use_gpu": True
    }

    recommender = ImplicitALSRecommender(URM_train)
    
    recommender.fit(**params)

    result = ranking_metrics_at_k(
        recommender.model,
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
    
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage="sqlite:///Prototype/optuna/optuna_study.db")
            
    save_results = SaveResults(csv_path=BASE_OPTUNA_FOLDER / f"logs/{STUDY_NAME}/trials_results.csv")

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 100)

if __name__ == "__main__":
    main()