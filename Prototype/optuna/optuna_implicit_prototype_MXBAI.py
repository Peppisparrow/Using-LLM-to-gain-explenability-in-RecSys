import os
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
from implicit.evaluation import ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares

# Defining Recommender
from Prototype.Decoder.ItemFactorLearner_implicit import ImplicitItemFactorLearner
from Prototype.data_manager import DataManger
from Prototype.utils.optuna_utils import SaveResults


# ---------- CONSTANTS ----------
METRIC = 'map'
METRIC_K = 10
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
STUDY_NAME = "implicit_prototype_MAP_MXBAI"
DATA_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small')
USER_EMBEDDING_PATH = Path('/leonardo_work/IscrC_DMG4RS/embednbreakfast/Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/user_embeddings_compressed_mxbai.npz')
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
        
    optuna_study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True, storage="sqlite:///Prototype/optuna/optuna_study.db")
            
    save_results = SaveResults(csv_path=BASE_OPTUNA_FOLDER / f"logs/{STUDY_NAME}/trials_results.csv")

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 100)

if __name__ == "__main__":
    main()