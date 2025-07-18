import os
from functools import partial
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import torch
# Defining Recommender
from RecSysFramework.Recommenders.Neural.TwoTowerRecommender import TwoTowerRecommenderConcat
from RecSysFramework.Recommenders.Neural.TwoTowerRecommender import TwoTowerRecConcatNorm
from Prototype.data_manager import DataManger
from Prototype.utils.optuna_utils import SaveResults
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout


# ---------- CONSTANTS ----------
METRIC = 'MAP_MIN_DEN'
METRIC_K = 10
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
STUDY_NAME = "2Tower_prova"
DATA_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10')
USER_EMBEDDING_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/user_embeddings_compressed.npz')
# ---------- /CONSTANTS ----------

def objective_function(trial, URM_train, URM_test):

    epochs = trial.suggest_int("epochs", 5, 40)
    batch_size = trial.suggest_int("batch_size", 512, 4096)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    output= 2 ** trial.suggest_int("input", 4, 8)
    moltiplication= 2 ** trial.suggest_int("output", 1, 8)
    input = output * moltiplication
    n_layers= trial.suggest_int("n_layers", 2, 5)
    # output = 1
    layers = np.linspace(input, output, n_layers + 2, dtype=np.int16)
    layers = layers.astype(int)
    print(f"Current layers: {layers}")
    recommender = TwoTowerRecConcatNorm(URM_train, URM_train.shape[0], URM_train.shape[1], layers=layers, verbose=True)
    print(f"Current parameters: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}, layers={layers}")
    optimizer = torch.optim.AdamW(params=recommender.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    print("Optimizer initialized.")
    
    recommender.fit(epochs=epochs, batch_size=batch_size, optimizer=optimizer)
    print("Recommender fitted.")
    
    print("Current {} = {:.4f} with parameters ".format(METRIC, 0))
    return 0

def main():
    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=USER_EMBEDDING_PATH)
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage="sqlite:///Prototype/optuna/optuna_study.db", sampler=optuna.samplers.TPESampler(seed=43))
            
    save_results = SaveResults(csv_path=BASE_OPTUNA_FOLDER / f"logs/{STUDY_NAME}/trials_results.csv")

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 1)

if __name__ == "__main__":
    main()