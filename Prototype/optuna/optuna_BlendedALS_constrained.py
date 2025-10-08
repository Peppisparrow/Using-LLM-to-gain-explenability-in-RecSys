import os
from functools import partial
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

import optuna
import pandas as pd
from implicit.evaluation import ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares

# Defining Recommender
from Prototype.Decoder.BlendedALSModelsUserRecommender import BlendedALSModelsUserRecommender
from Prototype.data_manager2 import DataManger
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
import sys

# ---------- CONSTANTS ----------
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
# ---------- /CONSTANTS ----------

def objective_function(trial, URM_train, URM_test, user_embeddings=None):

    init_model_params = {
        "iterations": trial.suggest_int("iterations", 1, 500),
        "regularization": trial.suggest_float("reg_init", 1e-5, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha_init", 0.0, 50.0),
        "confidence_scaling": trial.suggest_categorical("confidence_scaling", ['linear', 'log']),
    }
    
    #blending_factor = trial.suggest_float("blending_factor", 0.1, 0.9)
    blending_factor = trial.suggest_float("blending_factor", 0.1, 0.5)

    fixed_model_params = {
        "reg": trial.suggest_float("reg_fixed", 1e-5, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha_fixed", 0.0, 50.0),
    }

    recommender = BlendedALSModelsUserRecommender(URM_train)

    recommender.fit(
        user_factors=user_embeddings,
        blending_factor=blending_factor,
        init_model_params=init_model_params,
        fixed_model_params=fixed_model_params
    )


    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[METRIC_K], verbose=False, exclude_seen=True)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)
    result_from_evaluator = result_dict.loc[METRIC_K][METRIC]
    #print("IMPLICIT Current {} = {:.4f}".format(METRIC, result))
    print("PROF Current {} = {:.4f}".format(METRIC, result_from_evaluator))

    for metric_name, metric_value in result_dict.items():
        try:
            trial.set_user_attr(metric_name, float(metric_value))
        except (TypeError, ValueError):
            print(f"Attenzione: l'attributo '{metric_name}' con valore '{metric_value}' non è stato salvato perché non convertibile in float.")
    print("Current {} = {:.4f} with parameters {}".format(METRIC, result_from_evaluator, trial.params))

    return result_from_evaluator

def main():
    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=USER_EMBEDDING_PATH, user_key='user_id', item_key='item_id')
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    user_embeddings = data_manager.get_user_embeddings() if USER_EMBEDDING_PATH else None
    
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test,
        user_embeddings=user_embeddings,

    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage=f"sqlite:///{DB_PATH}")

    optuna_study.optimize(objective_function_with_data,
                        n_trials = N_TRIALS)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run Optuna study")
    parser.add_argument("--study_name", type=str, help="Name of the Optuna study")
    parser.add_argument("--data_path", type=str, help="Path to the dataset with train and test csv files")
    parser.add_argument("--user_embedding_path", type=str, help="Path to the user embeddings file", default=None)
    parser.add_argument("--db_path", type=str, help="Path to the database file", default="Prototype/optuna/optuna_study.db")
    parser.add_argument("--metric", type=str, help="Metric to optimize", default='NDCG') # Options: 'MAP_MIN_DEN', 'NDCG'
    parser.add_argument("--metric_k", type=int, help="K value for the metric", default=10)
    parser.add_argument("--n_trials", type=int, help="Number of optuna trials", default=100)
    args = parser.parse_args()
    
    
    STUDY_NAME = args.study_name
    DATA_PATH = Path(args.data_path)
    USER_EMBEDDING_PATH = Path(args.user_embedding_path) if args.user_embedding_path else None
    DB_PATH = args.db_path

    METRIC = args.metric
    METRIC_K = args.metric_k
    N_TRIALS = args.n_trials

    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}")
    print(f"Evaluation with implicit backend using metric: {METRIC} at K: {METRIC_K}")
    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}", file=sys.stderr)
    main()