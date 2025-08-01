import os
from functools import partial
from pathlib import Path
from argparse import ArgumentParser

import optuna
import pandas as pd
from implicit.evaluation import ranking_metrics_at_k
import sys
# Defining Recommender
from RecSysFramework.Recommenders.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from Prototype.data_manager2 import DataManger
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from Prototype.utils.optuna_utils import SaveResultsWithUserAttrs


# ---------- CONSTANTS ----------

BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
#STUDY_NAME = "IALS_STUDY_FIXED_ALPHA_MAP"
#DATA_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small')
#USER_EMBEDDING_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small/user_embeddings_compressed.npz')
# ---------- /CONSTANTS ----------

def objective_function(trial, URM_train, URM_test):
    
    params = {
        "iterations": trial.suggest_int("iterations", 1, 500),
        "factors": trial.suggest_int("num_factors", 10, 5200),
        "regularization": trial.suggest_float("regularization", 1e-5, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 0.0, 50.0),
        "confidence_scaling": trial.suggest_categorical("confidence_scaling", ['linear', 'log']),
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
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[METRIC_K], verbose=False, exclude_seen=True)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)
    map_from_evaluator = result_dict.loc[METRIC_K]['MAP_MIN_DEN']
    print("IMPLICIT Current {} = {:.4f}".format(METRIC, result))
    print("PROF Current {} = {:.4f}".format(METRIC, map_from_evaluator))

    for metric_name, metric_value in result_dict.items():
        # --- MODIFICA CHIAVE QUI ---
        # Converti esplicitamente il valore in un float nativo di Python.
        # Questo risolve i problemi di serializzazione con tipi come numpy.float64.
        try:
            trial.set_user_attr(metric_name, float(metric_value))
        except (TypeError, ValueError):
            # Questo blocco gestisce il caso in cui un valore non sia convertibile in float,
            # evitando che il trial fallisca. Utile per la robustezza.
            print(f"Attenzione: l'attributo '{metric_name}' con valore '{metric_value}' non è stato salvato perché non convertibile in float.")
    print("Current {} = {:.4f} with parameters {}".format(METRIC, result, params))
    print("Current {} = {:.4f} with parameters {}".format(METRIC, result, params))
    
    return result

def main():
    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=USER_EMBEDDING_PATH, user_key='user_id', item_key='item_id')
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test
    )

    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage=f"sqlite:///{DB_PATH}")

    save_results = SaveResultsWithUserAttrs(csv_path=BASE_OPTUNA_FOLDER / f"logs/{STUDY_NAME}/trials_results.csv")

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = N_TRIALS)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run Optuna study for IALS with fixed alpha")
    parser.add_argument("--study_name", type=str, help="Name of the Optuna study")
    parser.add_argument("--data_path", type=str, help="Path to the dataset with train and test csv files")
    parser.add_argument("--user_embedding_path", type=str, help="Path to the user embeddings file")
    parser.add_argument("--db_path", type=str, help="Path to the database file", default="Prototype/optuna/optuna_study.db")
    parser.add_argument("--metric", type=str, help="Metric to optimize", default='map')
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
    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}", file=sys.stderr)
    main()