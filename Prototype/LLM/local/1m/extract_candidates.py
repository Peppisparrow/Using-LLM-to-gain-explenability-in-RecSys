import os
from functools import partial
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
from implicit.evaluation import ranking_metrics_at_k
import sys
# Defining Recommender
from RecSysFramework.Recommenders.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from Prototype.data_manager2 import DataManger
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from Prototype.utils.optuna_utils import SaveResultsWithUserAttrs


# ---------- CONSTANTS ----------
NUM_CANDIDATES = 30
output_csv_path = f'Prototype/Dataset/ml/ml-1m/final/candidate_items_{NUM_CANDIDATES}.csv'
#STUDY_NAME = "IALS_STUDY_FIXED_ALPHA_MAP"
DATA_PATH = Path('Prototype/Dataset/ml/ml-1m/final/')
#USER_EMBEDDING_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small/user_embeddings_compressed.npz')


def main():
    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=None, user_key='user_id', item_key='item_id')
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    METRIC_K = 10
    METRIC = 'ndcg'
    params = {
        "iterations": 54,
        "factors": 165,
        "regularization": 0.022686701389472864,
        "alpha": 18.90196682289257,
        "confidence_scaling": 'linear',
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
    map_from_evaluator = result_dict.loc[METRIC_K]['NDCG']
    print("IMPLICIT Current {} = {:.4f}".format(METRIC, result))
    print("PROF Current {} = {:.4f}".format(METRIC, map_from_evaluator))
     
    index_to_user_map = data_manager.get_index_to_user_mapping()
    index_to_item_map = data_manager.get_index_to_item_mapping()

    recommendations_to_write = []
    # 2. Itera su tutti gli indici degli utenti nella matrice di training
    for user_index in range(URM_train.shape[0]):
        # Ottieni le raccomandazioni (che sono indici di item)
        recommended_item_indices = recommender.recommend(user_index, cutoff=NUM_CANDIDATES, remove_seen_flag=True)
        
        # Se ci sono raccomandazioni per questo utente
        if len(recommended_item_indices) > 0:
            # 3. Converti l'indice dell'utente nel suo ID originale
            original_user_id = index_to_user_map[user_index]
            
            # 4. Per ogni item raccomandato, crea una coppia (user_id, item_id)
            for item_index in recommended_item_indices:
                original_item_id = index_to_item_map[item_index]
                recommendations_to_write.append([original_user_id, original_item_id])

    # 5. Crea un DataFrame pandas e salvalo in CSV (modo efficiente)
    recommendations_df = pd.DataFrame(recommendations_to_write, columns=['user_id', 'item_id'])
    recommendations_df.to_csv(output_csv_path, index=False)
    return result

if __name__ == "__main__":
    main()