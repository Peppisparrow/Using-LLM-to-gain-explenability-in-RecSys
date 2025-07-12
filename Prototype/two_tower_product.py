import os
from functools import partial
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import torch
# Defining Recommender
from RecSysFramework.Recommenders.Neural.TwoTowerRecommender import TwoTowerRecommenderProduct
from RecSysFramework.Recommenders.Neural.TwoTowerRecommender import TwoTowerRecProductNorm
from RecSysFramework.Recommenders.Neural.TwoTowerRecommender import TwoTowerRecProductNormSeq
from Prototype.data_manager import DataManger
from Prototype.utils.optuna_utils import SaveResults
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
import cProfile
import pstats
import io

# ---------- CONSTANTS ----------
METRIC = 'MAP_MIN_DEN'
METRIC_K = 10
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
STUDY_NAME = "2Tower_prova"
DATA_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/mid')
USER_EMBEDDING_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/mid/user_embeddings_compressed_t5.npz')
# ---------- /CONSTANTS ----------

def objective_function(trial, URM_train, URM_test):
        
    
    # epochs = trial.suggest_int("epochs", 5, 50)
    # batch_size = trial.suggest_int("batch_size", 32, 512)
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    # input= 2 ** trial.suggest_int("input", 4, 13)
    # division= trial.suggest_int("output", 1, input)
    # output = 2 ** (input // division)
    # n_layers= trial.suggest_int("n_layers", 2, 5)
    
    # layers = np.linspace(input, output, n_layers + 2, dtype=np.int16)
    # layers = layers.astype(int)

    epochs = trial.suggest_int("epochs", 5, 10)
    batch_size = trial.suggest_int("batch_size", 512, 4096)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    output= 2 ** trial.suggest_int("input", 4, 8)
    moltiplication= 2 ** trial.suggest_int("output", 1, 3)
    input = output * moltiplication
    n_layers= trial.suggest_int("n_layers", 2, 5)
    # output = 1
    learning_rate = 1e-3
    weight_decay = 1e-5
    batch_size = 2048
    output = 64
    input = 1024
    n_layers = 5
    layers = np.linspace(input, output, n_layers + 2, dtype=np.int16)
    layers = layers.astype(int)
    print(f"Current layers: {layers}")
    recommender = TwoTowerRecProductNorm(URM_train, URM_train.shape[0], URM_train.shape[1], layers=layers, verbose=True)
    print(f"Current parameters: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}, layers={layers}")
    optimizer = torch.optim.AdamW(params=recommender.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    print("Optimizer initialized.")
    recommender = torch.compile(recommender)
    recommender.fit(epochs=1, batch_size=batch_size, optimizer=optimizer)
    print("Recommender fitted.")
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[METRIC_K], verbose=True, exclude_seen=True)
    profiler = cProfile.Profile()
    profiler.enable()

    print("--- INIZIO PROFILAZIONE EVALUATION ---")
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)
    print("--- FINE PROFILAZIONE EVALUATION ---")
    profiler.disable()

    # 5. Stampa i risultati del profiler
    s = io.StringIO()
    # Ordina le statistiche per 'cumulative time' per vedere dove è stato speso più tempo in totale
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(40)  # Stampa le 40 funzioni più lente

    print("\n\n--- RISULTATI DEL PROFILER (Top 40 funzioni per tempo cumulativo) ---")
    print(s.getvalue())
    result = result_dict.loc[METRIC_K][METRIC]
    
    print("Current {} = {:.4f} with parameters ".format(METRIC, result))
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
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage="sqlite:///Prototype/optuna/optuna_study.db", sampler=optuna.samplers.TPESampler(seed=43))
            
    save_results = SaveResults(csv_path=BASE_OPTUNA_FOLDER / f"logs/{STUDY_NAME}/trials_results.csv")

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 1)

if __name__ == "__main__":
    main()