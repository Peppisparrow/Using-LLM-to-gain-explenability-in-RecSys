import os
from functools import partial
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import torch
from argparse import ArgumentParser
# Defining Recommender
from RecSysFramework.Recommenders.Neural.TwoTowerRecommenderEmbeddingsPrototype import TwoTowerRecProductNorm
from Prototype.data_manager_peppe import DataManger
from Prototype.utils.optuna_utils import SaveResults
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
# ---------- CONSTANTS ----------
METRIC = 'MAP'
METRIC_K = 10
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
# STUDY_NAME = "2Tower_product_norm_prototype"
# DATA_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10')
# USER_EMBEDDING_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/user_embeddings_compressed_t5.npz')
# ITEM_EMBEDDING_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/game_embeddings_t5.npz')
# ---------- /CONSTANTS ----------

def objective_function(trial, URM_train, URM_test, item_embeddings=None, user_embeddings=None):

    if user_embeddings is not None:
        print("Using user embeddings from the data manager.")
        user_embeddings_dim = user_embeddings.shape[1]
    else:
        user_embeddings_dim = None
    if item_embeddings is not None:
        item_embeddings_dim = item_embeddings.shape[1]
    else:
        item_embeddings_dim = None

    epochs = trial.suggest_int("epochs", 5, 10)
    batch_size = trial.suggest_int("batch_size", 512, 4096)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    input = 2 ** trial.suggest_int("input", 4, 7)
    n_layers= trial.suggest_int("n_layers", 2, 5)
    # output = 1
    output = 4
    input = 32
    layers = np.linspace(input, output, n_layers, dtype=np.int16)
    layers = layers.astype(int)
    print(f"Current layers: {layers}")
    recommender = TwoTowerRecProductNorm(URM_train, URM_train.shape[0], URM_train.shape[1], user_embeddings_dim=user_embeddings_dim, item_embeddings_dim=item_embeddings_dim, layers=layers, verbose=True, first_dim_layer=32)
    print(f"Current parameters: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}, layers={layers}")
    optimizer = torch.optim.AdamW(params=recommender.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    print("Optimizer initialized.")
    recommender = torch.compile(recommender)
    recommender.fit(epochs=1, batch_size=batch_size, optimizer=optimizer, user_embeddings=user_embeddings, item_embeddings=item_embeddings)
    print("Recommender fitted.")
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[METRIC_K], verbose=True, exclude_seen=True)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)
    result = result_dict.loc[METRIC_K][METRIC] 
    
    print("Current {} = {:.16f} with parameters ".format(METRIC, result))
    return result

def main():
    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=USER_EMBEDDING_PATH, item_embeddings_path=ITEM_EMBEDDING_PATH)
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    if USER_EMBEDDING_PATH is not None:
        print("Loading user embeddings from path:", USER_EMBEDDING_PATH)
        user_embeddings = data_manager.get_user_embeddings()
    else:
        user_embeddings = None
    if ITEM_EMBEDDING_PATH is not None:
        item_embeddings = data_manager.get_item_embeddings()
    else:
        item_embeddings = None
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test,
        item_embeddings=item_embeddings,
        user_embeddings=user_embeddings
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage="sqlite:///Prototype/optuna/optuna_study.db", sampler=optuna.samplers.TPESampler(seed=43))
            
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
    args = parser.parse_args()
    
    
    STUDY_NAME = args.study_name
    DATA_PATH = Path(args.data_path)
    if args.user_embedding_path is not None:
        USER_EMBEDDING_PATH = Path(args.user_embedding_path)
    else:
        USER_EMBEDDING_PATH = None
    if args.item_embedding_path is not None:
        ITEM_EMBEDDING_PATH = Path(args.item_embedding_path)
    else:
        ITEM_EMBEDDING_PATH = None
    
    print(f"Running study: {STUDY_NAME} with data path: {DATA_PATH} and user embedding path: {USER_EMBEDDING_PATH}")
    
    main()