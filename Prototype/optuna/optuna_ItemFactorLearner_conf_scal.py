import os
from functools import partial
from pathlib import Path
from argparse import ArgumentParser

import optuna
import pandas as pd
from implicit.evaluation import ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares
import sys
# Defining Recommender
from Prototype.Decoder.ItemFactorLearner_implicit_with_conf_scaling import ImplicitItemFactorLearner
from Prototype.data_manager2 import DataManger
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from Prototype.utils.optuna_utils import SaveResultsWithUserAttrs


# ---------- CONSTANTS ----------
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
# ---------- /CONSTANTS ----------

def objective_function(trial, user_embeddings, URM_train, URM_test):
    
    params = {
        "user_factors": user_embeddings,
        "alpha": trial.suggest_float("alpha", 0.0, 50.0),
        "reg": trial.suggest_float("reg", 1e-5, 1e-1, log=True),
        "confidence_scaling": trial.suggest_categorical("confidence_scaling", ['linear', 'log']),
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
    user_embeddings = data_manager.get_user_embeddings()
    
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test,
        user_embeddings=user_embeddings
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage=f"sqlite:///{DB_PATH}")
            
    #save_results = SaveResultsWithUserAttrs(csv_path=BASE_OPTUNA_FOLDER / f"logs/{STUDY_NAME}/trials_results.csv")

    optuna_study.optimize(objective_function_with_data,
                        #callbacks=[save_results],
                        n_trials = N_TRIALS)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run Optuna study for ItemFactorLearner")
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
    USER_EMBEDDING_PATH = Path(args.user_embedding_path)
    DB_PATH = args.db_path
    
    METRIC = args.metric
    METRIC_K = args.metric_k
    N_TRIALS = args.n_trials
    
    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}")
    print(f"Database path: {DB_PATH}")
    print(f"Evaluation with implicit backend using metric: {METRIC} at K: {METRIC_K}")
    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}", file=sys.stderr)
    
    main()