import os
from functools import partial
from pathlib import Path
from argparse import ArgumentParser

import optuna
import pandas as pd
import numpy as np
from implicit.evaluation import ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares

# Defining Recommender
from Prototype.Decoder.ImplicitUserFactorLearner import ImplicitUserFactorLearner
from Prototype.data_manager import DataManger
from Prototype.utils.optuna_utils import SaveResults


# ---------- CONSTANTS ----------
METRIC = 'map'
METRIC_K = 10
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
# ---------- /CONSTANTS ----------

def objective_function(trial, item_embeddings, URM_train, URM_test):
    
    params = {
        "item_factors": item_embeddings,
        "alpha": trial.suggest_float("alpha", 0.0, 50.0),
        "reg": trial.suggest_float("reg", 1e-5, 1e-1, log=True),
    }

    recommender = ImplicitUserFactorLearner(URM_train)
    
    recommender.fit(**params)

    fake_model = AlternatingLeastSquares(
        regularization=params["reg"],
        factors=params["item_factors"].shape[1],
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

def get_item_embeddings(item2index):
    """
    Load item embeddings from a file and map them to the indices in item2index.
    """
    item_embeddings_DICT = np.load(ITEM_EMBEDDING_PATH)
    item_embeddings = item_embeddings_DICT['embeddings']
    item_ids = item_embeddings_DICT['app_id']
    
    # Convert item_ids to integers
    item_ids = [int(item_id) for item_id in item_ids]
    
    # Assert that all items in item2index are in item_ids
    assert all(item in item_ids for item in item2index.keys()), "Not all items in item2index are present in item_ids"
    
    # Then we map the item_ids to their indices
    item_embeddings = item_embeddings[[item2index[item_id] for item_id in item_ids if item_id in item2index]]
    
    return item_embeddings

def main():
    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=USER_EMBEDDING_PATH)
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    item_embeddings = get_item_embeddings(data_manager.get_item_mapping())
    
    
    objective_function_with_data = partial(
        objective_function,
        item_embeddings=item_embeddings,
        URM_train=URM_train,
        URM_test=URM_test
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage="sqlite:///Prototype/optuna/optuna_study.db")
            
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
    args = parser.parse_args()
    
    
    STUDY_NAME = args.study_name
    DATA_PATH = Path(args.data_path)
    USER_EMBEDDING_PATH = Path(args.user_embedding_path)
    ITEM_EMBEDDING_PATH = Path(args.item_embedding_path)
    
    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}")
    
    main()