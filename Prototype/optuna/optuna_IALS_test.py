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

def objective_function(URM_train, URM_test):

    params = {
        "iterations": 11,
        "factors": 841,
        "regularization": 0.022686701389472864,
        "alpha": 31.345846885311925,
        "confidence_scaling": "linear",
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

    print(f"Result dict: {result_dict}")

def main():
    data_manager = DataManger(data_path=DATA_PATH, user_key='user_id', item_key='item_id')
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    
    objective_function(
        URM_train=URM_train,
        URM_test=URM_test
    )

    

if __name__ == "__main__":

    DATA_PATH = Path("Prototype/Dataset/ml_small/final")

    METRIC = "map"
    METRIC_K = 10
    main()