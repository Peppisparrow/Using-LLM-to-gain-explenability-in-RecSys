import os
from functools import partial
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import torch

from implicit.evaluation import ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares

from RecSysFramework.Recommenders.Neural.TwoTower import TwoTowerRecommender
from Prototype.data_manager_peppe import DataManger
from Prototype.utils.optuna_utils import SaveResults
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from Prototype.Decoder.ItemFactorLearner_implicit import ImplicitItemFactorLearner
# import cProfile
# import pstats
# import io
# Current layers: [256 192 128  64   1]
# Current parameters: epochs=10, batch_size=2048, learning_rate=0.001, weight_decay=1e-05, layers=[256 192 128  64   1]
# ---------- CONSTANTS ----------
METRIC = 'MAP_MIN_DEN'
METRIC_K = 10
BASE_OPTUNA_FOLDER = Path("Prototype/optuna/")
STUDY_NAME = "Provailnuovobimbo"
DATA_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small')
USER_EMBEDDING_PATH = Path('Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/small/user_embeddings_compressed_t5.npz')
def calculate_map_at_k_manual(model, urm_train, urm_test, k):
        """
        Calcola manualmente la MAP@K, replicando la logica di un valutatore standard.
        """
        # Ottieni gli utenti che hanno interazioni nel set di test
        users_in_test = np.unique(urm_test.nonzero()[0])
        
        average_precisions = []
        
        for user_id in users_in_test:
            # Ground truth: gli item con cui l'utente ha interagito nel test set
            ground_truth_items = urm_test[user_id].indices
            if not len(ground_truth_items):
                continue

            # Ottieni le raccomandazioni dal modello
            # `filter_already_liked_items=True` è FONDAMENTALE e corrisponde a passare
            # URM_train alla funzione `ranking_metrics_at_k`.
            recommended_items_ids, _ = model.recommend(
                userid=user_id,
                user_items=urm_train[user_id],
                N=k,
                filter_already_liked_items=True
            )
            
            # Calcola l'Average Precision (AP@k) per questo utente
            hits = 0
            precision_sum = 0.0
            
            for i, recommended_id in enumerate(recommended_items_ids):
                if recommended_id in ground_truth_items:
                    hits += 1
                    precision_at_i = hits / (i + 1)
                    precision_sum += precision_at_i
            
            # Il denominatore per AP è min(k, |ground_truth|)
            denominator = min(k, len(ground_truth_items))
            ap_at_k = precision_sum / denominator if denominator > 0 else 0.0
            average_precisions.append(ap_at_k)

        # La MAP@k è la media delle AP@k su tutti gli utenti del test set
        return np.mean(average_precisions) if average_precisions else 0.0
# ---------- /CONSTANTS ----------

def objective_function(trial, URM_train, URM_test):

    epochs = trial.suggest_int("epochs", 5, 10)
    batch_size = trial.suggest_int("batch_size", 512, 4096)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    input = 2 ** trial.suggest_int("input", 4, 7)
    n_layers= trial.suggest_int("n_layers", 2, 5)
    # output = 1
    output = 1
    input = 1
    n_layers = 2
    batch_size = 4048
    layers = np.linspace(input, output, n_layers, dtype=np.int16)
    layers = layers.astype(int)
    print(f"Current layers: {layers}")
    recommender = TwoTowerRecommender(URM_train, URM_train.shape[0], URM_train.shape[1], layers=layers, verbose=True)
    print(f"Current parameters: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}, layers={layers}")
    optimizer = torch.optim.AdamW(params=recommender.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    print("Optimizer initialized.")
    recommender = torch.compile(recommender)
    recommender.fit(epochs=1, batch_size=batch_size, optimizer=optimizer)
    print("Recommender fitted.")
    recommender.compute_all_embeddings(batch_size=batch_size)


    fake_model = AlternatingLeastSquares(
        regularization=0.1,
        factors=recommender.ITEM_factors.shape[1],
        iterations=1,  # We don't need to train the model, just need the structure
        num_threads=1,  # Number of threads can be adjusted based on the environment
        calculate_training_loss=False,  # We don't need training loss for this task
        use_gpu=False,  # Use GPU for training

    )
    fake_model.item_factors = recommender.ITEM_factors
    fake_model.user_factors = recommender.USER_factors
    
    fake_model = fake_model.to_gpu()

    result_imp = ranking_metrics_at_k(
        fake_model,
        URM_train,
        URM_test,
        K=METRIC_K,
    )['map']
    
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[METRIC_K], verbose=True, exclude_seen=True)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)
    result = result_dict.loc[METRIC_K][METRIC]

    rec = ImplicitItemFactorLearner(URM_train)
    rec.USER_factors = recommender.USER_factors
    rec.ITEM_factors = recommender.ITEM_factors

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[METRIC_K], verbose=False, exclude_seen=True)
    result_dict, _ = evaluator_test.evaluateRecommender(rec)
    map_from_evaluator = result_dict.loc[METRIC_K]['MAP_MIN_DEN']
    print(f"MAP@{METRIC_K} from EvaluatorHoldout on ials: {map_from_evaluator:.6f}")
    print(f"MAP@{METRIC_K} from EvaluatorHoldout on twotower: {result:.6f}")
    print(f"MAP@{METRIC_K} from implicit: {result_imp:.6f}")



    if URM_test.nnz > 0:
        user_id_test = URM_test.nonzero()[0][0]
        print(f"Utente selezionato per il test: user_id = {user_id_test}\n")

        # 1. Raccomandazioni dal modello ORIGINALE 'implicit'
        # Questo è il nostro riferimento corretto.
        original_recs, original_scores = fake_model.recommend(
            userid=user_id_test,
            user_items=URM_train[user_id_test],
            N=METRIC_K,
            filter_already_liked_items=True
        )

        print(f"--- Raccomandazioni da modello 'implicit' (corretto) ---")
        for i, (item, score) in enumerate(zip(original_recs, original_scores)):
            print(f"  {i+1}. Item: {item:<5} Score: {score:.4f}")
        print("-" * 25)

        # 2. Raccomandazioni dal tuo recommender custom 'ImplicitItemFactorLearner'
        # Assumiamo che abbia un metodo .recommend() con una firma simile
        
        custom_recs, custom_score = recommender.recommend(
            user_id_array=user_id_test, 
            cutoff=10,    
            remove_seen_flag=True, # Assicurati che il tuo metodo supporti questo
            return_scores=True  # Assicurati che il tuo metodo supporti questo
        )
        # Ora custom_recs e custom_scores sono semplici array, non matrici
        for i, item_id in enumerate(custom_recs):
            score = custom_score[0, item_id]
            
            print(f"  {i+1}. Item: {item_id:<5} Score: {score}")

        print("-" * 25)

        # 3. Confronto
        are_recs_identical = np.array_equal(original_recs, custom_recs)
        if are_recs_identical:
            print("✅ RISULTATO: Le liste di raccomandazioni sono IDENTICHE.")
            print("🐛 Il bug è quasi sicuramente nella classe 'EvaluatorHoldout'.")
        else:
            print("❌ RISULTATO: Le liste di raccomandazioni sono DIVERSE.")
            print("🐛 Il bug è quasi sicuramente nel metodo .recommend() della tua classe 'ImplicitItemFactorLearner'.")
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