import os
from functools import partial
from pathlib import Path
from argparse import ArgumentParser

import optuna
import pandas as pd
import numpy as np
import scipy.sparse as sps

# Defining Recommender
from RecSysFramework.Recommenders.FactorizationMachines.LightFMRecommenderBoosted import LightFMUserItemHybridRecommender
from Prototype.data_manager import DataManger
from Prototype.utils.optuna_utils import SaveResults
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout


# ---------- CONSTANTS ----------
METRIC = 'MAP_MIN_DEN'
METRIC_K = 10
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
# ---------- /CONSTANTS ----------

def objective_function(trial, user_embeddings, item_embeddings, URM_train, URM_test):

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[METRIC_K])
    recommender = LightFMUserItemHybridRecommender(URM_train, ICM_train=item_embeddings, UCM_train=user_embeddings)
    recommender.vectorized_mode = True # Enable vectorized mode for faster computation
    recommender.num_threads = 6
    
    params = {
        "num_threads": recommender.num_threads,
        "epochs": trial.suggest_int("epochs", 1, 100, step=5),
        "num_components": trial.suggest_int("num_components", 10, 600, step=15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "loss": 'bpr',  # explicitly set loss function to 'bpr' (by default is the same)
        "item_alpha": trial.suggest_float("item_alpha", 0.0, 1.0, log=True),
        "user_alpha": trial.suggest_float("user_alpha", 0.0, 1.0, log=True),
        # Now early stopping kwargs
        'stop_on_validation': True,
        'validation_every_n': 5,
        'evaluator_object': evaluator_test,
        'lower_validations_allowed':3, # 3*validation_every_n = 15 epochs
        'validation_metric': METRIC,
    }
    
    print("Optuna for LightFM: Current parameters: {}".format(params))
    
    recommender.fit(**params)

    print(f"Training completed... evaluating with {METRIC} at cutoff {METRIC_K}")
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)
    result = result_dict.loc[METRIC_K][METRIC]
    
    print("Current {} = {:.4f} with parameters {}".format(METRIC, result, params))
    
    return result

def main():
    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=USER_EMBEDDING_PATH, item_embeddings_path=ITEM_EMBEDDING_PATH)
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    item_embeddings = data_manager.get_item_embeddings()
    user_embeddings = data_manager.get_user_embeddings()
    
    user_embeddings = sps.csr_matrix(user_embeddings) if isinstance(user_embeddings, np.ndarray) else user_embeddings
    item_embeddings = sps.csr_matrix(item_embeddings) if isinstance(item_embeddings, np.ndarray) else item_embeddings
    
    
    objective_function_with_data = partial(
        objective_function,
        user_embeddings=data_manager,
        item_embeddings=item_embeddings,
        URM_train=URM_train,
        URM_test=URM_test,   
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage=f"sqlite:///{DB_PATH}")
            
    save_results = SaveResults(csv_path=BASE_OPTUNA_FOLDER / f"logs/{STUDY_NAME}/trials_results.csv")

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 100)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run Optuna study for ImplicitUserFactorLearner")
    parser.add_argument("--study_name", type=str, help="Name of the Optuna study")
    parser.add_argument("--data_path", type=str, help="Path to the dataset with train and test csv files")
    parser.add_argument("--user_embedding_path", type=str, help="Path to the user embeddings file")
    parser.add_argument("--item_embedding_path", type=str, help="Path to the item embeddings file")
    parser.add_argument("--db_path", type=str, help="Path to the database file", default="Prototype/optuna/optuna_study.db")
    args = parser.parse_args()
    
    
    STUDY_NAME = args.study_name
    DATA_PATH = Path(args.data_path)
    USER_EMBEDDING_PATH = Path(args.user_embedding_path)
    ITEM_EMBEDDING_PATH = Path(args.item_embedding_path)
    DB_PATH = args.db_path
    
    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}")
    print(f"Item embedding path: {ITEM_EMBEDDING_PATH} and database path: {DB_PATH}")
    
    main()