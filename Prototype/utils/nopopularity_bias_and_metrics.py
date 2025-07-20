from pathlib import Path
import json

import numpy as np
from scipy.sparse import csr_matrix
import optuna
import torch


from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from RecSysFramework.Recommenders.Neural.TwoTowerE import TwoTowerRecommender
from Prototype.data_manager2 import DataManger


def remove_popularity_bias(urm_test: csr_matrix, fraction_to_remove: float = 1/3):
    """
    Identifica gli item più popolari da ignorare per ridurre il popularity bias,
    fino a coprire una certa frazione delle interazioni totali.
    La URM di test NON viene modificata nella sua forma, ma viene restituita
    una lista di ID di item da ignorare.

    Args:
        urm_test (csr_matrix): La URM di test (matrice sparse in formato CSR).
        fraction_to_remove (float): La frazione delle interazioni totali da "rimuovere"
                                     (cioè, gli item le cui interazioni sommate superano questa frazione
                                      saranno considerati da ignorare).

    Returns:
        list: Una lista di ID originali degli item più popolari da ignorare.
    """

    # 1. Calcolare la popolarità degli item (numero di interazioni per item)
    item_popularity = np.array(urm_test.sum(axis=0)).flatten()

    # Creare una lista di tuple (indice_item, popolarità)
    item_indices = np.arange(urm_test.shape[1])
    indexed_item_popularity = list(zip(item_indices, item_popularity))

    # 2. Ordinare gli item per popolarità in ordine decrescente
    sorted_items = sorted(indexed_item_popularity, key=lambda x: x[1], reverse=True)

    # Calcolare il numero totale di interazioni nella URM di test
    total_interactions = urm_test.nnz
    interactions_to_remove = int(total_interactions * fraction_to_remove)
    
    print(f"Interazioni totali nella URM di test: {total_interactions}")
    print(f"Interazioni target da ignorare (somma delle popolarità): {interactions_to_remove}")

    current_interactions_removed_sum = 0
    items_to_ignore_ids = []

    # 3. Identificare gli item (colonne) da ignorare
    for item_idx, popularity in sorted_items:
        if current_interactions_removed_sum >= interactions_to_remove:
            break
        items_to_ignore_ids.append(item_idx)
        current_interactions_removed_sum += popularity
        
    print(f"Numero di items da ignorare: {len(items_to_ignore_ids)}")
    print(f"Somma delle popolarità degli item da ignorare: {current_interactions_removed_sum}")

    return items_to_ignore_ids


def main():
    if USER_EMBEDDING_PATH is not None:
        print("Using user embeddings from:", USER_EMBEDDING_PATH)
    if ITEM_EMBEDDING_PATH is not None:
        print("Using item embeddings from:", ITEM_EMBEDDING_PATH)

    data_manager = DataManger(data_path=DATA_PATH, user_embedding_path=USER_EMBEDDING_PATH, item_embeddings_path=ITEM_EMBEDDING_PATH, user_key='user_id', item_key='item_id')
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()

    # Get the list of popular items to ignore
    items_to_ignore_for_bias_reduction = remove_popularity_bias(URM_test, fraction_to_remove=1/3)
    
    # Initialize evaluator for original URM_test (no ignored items)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    
    # Initialize evaluator for "filtered" context by passing the items to ignore.
    # The URM_test passed to it still has the original shape.
    evaluator_test_filtered = EvaluatorHoldout(URM_test, cutoff_list=[10],
                                               ignore_items=items_to_ignore_for_bias_reduction)
    
    # Load optuna study
    optuna_study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{OPTUNA_PATH}")
    
    # Get best parameters
    print(f"Original value for study {STUDY_NAME}: {optuna_study.best_value}")
    
    best_params = optuna_study.best_params
    print(f"Best parameters for {STUDY_NAME}: {best_params}")
    
    # TWO TOWER SECTION
    epochs = best_params['epochs']
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    output = best_params['output']
    input = best_params['input']
    n_layers = best_params['n_layers']
    
    input = 2 ** input
    output = 2 ** output
    layers = np.linspace(input, output, n_layers, dtype=np.int16).astype(int)
    
    user_embeddings = data_manager.get_user_embeddings() if USER_EMBEDDING_PATH else None
    item_embeddings = data_manager.get_item_embeddings() if ITEM_EMBEDDING_PATH else None
    print('-'* 50)
    print(f"Using user embeddings: {user_embeddings is not None}")
    print(f"Using item embeddings: {item_embeddings is not None}")
    print('-'* 50)
    # END
    recommender = TwoTowerRecommender(URM_train, user_embeddings=user_embeddings, item_embeddings=item_embeddings, num_items=URM_train.shape[1], num_users=URM_train.shape[0], layers=layers, verbose=True)
    optimizer = torch.optim.AdamW(params=recommender.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    recommender.fit(
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
    )
    
    model_results = {}
    print('-'* 50)
    print("\n--- Evaluating on Original URM_test ---")
    results, _ = evaluator_test.evaluateRecommender(recommender)
    
    print("\n--- Evaluating on Popularity-Reduced URM_test ---")
    # Temporarily set items to ignore in the recommender for this specific evaluation
    recommender.set_items_to_ignore(items_to_ignore_for_bias_reduction)
    results_filtered, _ = evaluator_test_filtered.evaluateRecommender(recommender)
    # Reset items to ignore after evaluation for good practice
    recommender.reset_items_to_ignore()

    for metric in METRICS:
        print(f"\nEvaluating metric: {metric}")
        
        # Evaluate on original URM test
        result = results.loc[METRIC_K][metric]
        model_results[metric] = result
        print(f"Results on original URM test (K={METRIC_K}): {result:.6f}")

        # Evaluate on filtered URM test
        result_filt = results_filtered.loc[METRIC_K][metric]
        model_results[f"{metric}_filtered"] = result_filt
        print(f"Results on filtered URM test (K={METRIC_K}): {result_filt:.6f}")

    results['STUDY_NAME'] = STUDY_NAME
    results['OPTUNA_PATH'] = OPTUNA_PATH
    results['METRIC_K'] = METRIC_K
    # Save results
    SAVING_PATH.mkdir(parents=True, exist_ok=True)
    with open(SAVING_PATH / f"{STUDY_NAME}_results.json", 'w') as f:
        json.dump(model_results, f, indent=4)
        print(f"\nResults saved to: {SAVING_PATH / f'{STUDY_NAME}_results.json'}")

# Esempio di utilizzo:
if __name__ == "__main__":
    
    METRICS = ['MAP_MIN_DEN','DIVERSITY_MEAN_INTER_LIST', "COVERAGE_ITEM", 'NOVELTY']
    OPTUNA_PATH = "Prototype/optuna/optuna_study_ML_small.db" # Make sure this is the correct path to your .db file
    STUDY_NAME = "New2TP_ITEMUSERMean_map"
    SAVING_PATH = Path("Prototype/optuna/popularity_filter_and_metrics")
    DATA_PATH = Path("/leonardo_work/IscrC_DMG4RS/embednbreakfast/Prototype/Dataset/ml-latest-small")
    METRIC_K = 10
    
    # Placeholder for embedding paths if not used
    #USER_EMBEDDING_PATH = None
    USER_EMBEDDING_PATH=Path('/leonardo_work/IscrC_DMG4RS/embednbreakfast/Prototype/Dataset/ml-latest-small/mean_user_embeddings_mxbai.npz')
    ITEM_EMBEDDING_PATH=Path('/leonardo_work/IscrC_DMG4RS/embednbreakfast/Prototype/Dataset/ml-latest-small/item_embeddings_mxbai.npz')
    #ITEM_EMBEDDING_PATH = None
    
    main()