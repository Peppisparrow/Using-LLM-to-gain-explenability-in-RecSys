from pathlib import Path
from argparse import ArgumentParser
import pandas as pd # Added for CSV handling

import numpy as np
from scipy.sparse import csr_matrix
import optuna
import torch

from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from RecSysFramework.Recommenders.Neural.TwoTowerE import TwoTowerRecommender
from RecSysFramework.Recommenders.Neural.TwoTowerMultipleStrategy import TwoTowerRecommender as TwoTowerRecommenderMultipleStrategy
from RecSysFramework.Recommenders.MatrixFactorization.ImplicitALSEmbeddingsInitialization import ImplicitALSRecommender
from Prototype.Decoder.ItemFactorLearner_implicit import ImplicitItemFactorLearner
from Prototype.Decoder.ImplicitUserFactorLearner import ImplicitUserFactorLearner
from Prototype.data_manager2 import DataManger

import threadpoolctl
threadpoolctl.threadpool_limits(1, "blas")


def remove_popularity_bias(urm_test: csr_matrix, fraction_to_remove: float = 1/3):
    """
    Identifies the most popular items to ignore to reduce popularity bias,
    up to a certain fraction of total interactions.
    The test URM is not modified in shape, but a list of item IDs to ignore is returned.

    Args:
        urm_test (csr_matrix): The test URM (sparse matrix in CSR format).
        fraction_to_remove (float): The fraction of total interactions to "remove"
                                     (i.e., items whose summed interactions exceed this fraction
                                      will be considered for ignoring).

    Returns:
        list: A list of original IDs of the most popular items to ignore.
    """

    # 1. Calculate item popularity (number of interactions per item)
    item_popularity = np.array(urm_test.sum(axis=0)).flatten()

    # Create a list of tuples (item_index, popularity)
    item_indices = np.arange(urm_test.shape[1])
    indexed_item_popularity = list(zip(item_indices, item_popularity))

    # 2. Sort items by popularity in descending order
    sorted_items = sorted(indexed_item_popularity, key=lambda x: x[1], reverse=True)

    # Calculate the total number of interactions in the test URM
    total_interactions = urm_test.nnz
    interactions_to_remove = int(total_interactions * fraction_to_remove)
    
    print(f"Total interactions in test URM: {total_interactions}")
    print(f"Target interactions to ignore (sum of popularities): {interactions_to_remove}")

    current_interactions_removed_sum = 0
    items_to_ignore_ids = []

    # 3. Identify items (columns) to ignore
    for item_idx, popularity in sorted_items:
        if current_interactions_removed_sum >= interactions_to_remove:
            break
        items_to_ignore_ids.append(item_idx)
        current_interactions_removed_sum += popularity
        
    print(f"Number of items to ignore: {len(items_to_ignore_ids)}")
    print(f"Sum of popularities of items to ignore: {current_interactions_removed_sum}")

    return items_to_ignore_ids

def get_two_towers_results(best_params, data_manager: DataManger, URM_train: csr_matrix):
    """
    Trains and returns a TwoTowerRecommender model with the given parameters.
    """
    epochs = best_params['epochs']
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    output = best_params['output']
    input = best_params['input']
    n_layers = best_params['n_layers']
    
    input_dim = 2 ** input
    output_dim = 2 ** output
    layers = np.linspace(input_dim, output_dim, n_layers, dtype=np.int16).astype(int)
    
    user_embeddings = data_manager.get_user_embeddings() if USER_EMBEDDING_PATH else None
    item_embeddings = data_manager.get_item_embeddings() if ITEM_EMBEDDING_PATH else None
    print('-'* 50)
    print(f"Using user embeddings: {user_embeddings is not None}")
    print(f"Using item embeddings: {item_embeddings is not None}")
    print('-'* 50)
    
    print("Layers configuration:", layers)

    recommender = TwoTowerRecommender(URM_train, user_embeddings=user_embeddings, item_embeddings=item_embeddings, num_items=URM_train.shape[1], num_users=URM_train.shape[0], layers=layers, verbose=True)
    optimizer = torch.optim.AdamW(params=recommender.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    recommender.fit(
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
    )
    
    return recommender

def get_IALS_results(best_params, data_manager: DataManger, URM_train: csr_matrix):
    """
    Trains and returns an ImplicitALSRecommender model with the given parameters.
    """
    best_params['factors'] = best_params.pop('num_factors')  # Ensure 'factors' is used correctly
    best_params['use_gpu'] = True  # Ensure GPU usage is enabled if available
    recommender = ImplicitALSRecommender(URM_train)
    recommender.fit(
        **best_params,
    )
    return recommender

def get_IALS_results_with_embeddings(best_params, data_manager: DataManger, URM_train: csr_matrix):
    """
    Trains and returns an ImplicitALSRecommender model initialized with pre-trained embeddings.
    """
    best_params['factors'] = 1024 # Example dimension, adjust as needed
    user_embeddings = data_manager.get_user_embeddings() if USER_EMBEDDING_PATH else None
    item_embeddings = data_manager.get_item_embeddings() if ITEM_EMBEDDING_PATH else None
    
    best_params['user_factors'] = user_embeddings
    best_params['item_factors'] = item_embeddings
    best_params['use_gpu'] = True
    
    recommender = ImplicitALSRecommender(URM_train)
    recommender.fit(
        **best_params,
    )
    return recommender

def get_ItemFactorLearner_results(best_params, data_manager: DataManger, URM_train: csr_matrix):
    """
    Trains and returns an ImplicitALSRecommender model initialized with pre-trained embeddings.
    """
    best_params['user_factors'] = data_manager.get_user_embeddings()

    recommender = ImplicitItemFactorLearner(URM_train)
    recommender.fit(
        **best_params,
    )
    return recommender

def get_UserFactorLearner_results(best_params, data_manager: DataManger, URM_train: csr_matrix):
    """
    Trains and returns an ImplicitALSRecommender model initialized with pre-trained embeddings.
    """
    best_params['item_factors'] = data_manager.get_item_embeddings()

    recommender = ImplicitUserFactorLearner(URM_train)
    recommender.fit(
        **best_params,
    )
    return recommender

def get_2T_mixed_strategy_results(best_params, data_manager: DataManger, URM_train: csr_matrix, strategy='concatenate'):
    """
    Trains and returns a TwoTowerRecommender model with a mixed strategy of user and item embeddings.
    """
    user_embeddings = data_manager.get_user_embeddings() if USER_EMBEDDING_PATH else None
    item_embeddings = data_manager.get_item_embeddings() if ITEM_EMBEDDING_PATH else None
    
    epochs = best_params['epochs']
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    output = 2 ** best_params['input']
    multiplier = 2 ** best_params['multiplier']
    n_layers = best_params['n_layers']
    input = output * multiplier
    layers = np.linspace(input, output, n_layers, dtype=np.int16)
    layers = layers.astype(int)

    recommender = TwoTowerRecommenderMultipleStrategy(URM_train,
                                      URM_train.shape[0],
                                      URM_train.shape[1],
                                      user_embeddings=user_embeddings,
                                      item_embeddings=item_embeddings,
                                      layers=layers,
                                      user_embedding_mode='mixed',
                                      item_embedding_mode='mixed',
                                      fusion_strategy=strategy,
                                      verbose=True)
    
    optimizer = torch.optim.AdamW(params=recommender.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    recommender = torch.compile(recommender)
    recommender.fit(epochs=epochs, batch_size=batch_size, optimizer=optimizer)
    
    return recommender

def main():
    """
    Main execution function to run the experiment, evaluate the model, and save results to a CSV.
    """
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
    evaluator_test_filtered = EvaluatorHoldout(URM_test, cutoff_list=[10],
                                               ignore_items=items_to_ignore_for_bias_reduction)
    
    # Load optuna study
    optuna_study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{OPTUNA_PATH}")
    
    # Get best parameters
    print(f"Original value for study {STUDY_NAME}: {optuna_study.best_value}")
    
    best_params = optuna_study.best_params
    print(f"Best parameters for {STUDY_NAME}: {best_params}")

    if MODEL_TYPE == 'TwoTower':
        recommender = get_two_towers_results(best_params, data_manager, URM_train)
    elif MODEL_TYPE == 'IALS':
        recommender = get_IALS_results(best_params, data_manager, URM_train)
    elif MODEL_TYPE == 'IALS_with_embeddings':
        recommender = get_IALS_results_with_embeddings(best_params, data_manager, URM_train)
    elif MODEL_TYPE == 'ItemFactorLearner':
        recommender = get_ItemFactorLearner_results(best_params, data_manager, URM_train)
    elif MODEL_TYPE == 'UserFactorLearner':
        recommender = get_UserFactorLearner_results(best_params, data_manager, URM_train)
    elif MODEL_TYPE == '2T_mixed_strategy':
        recommender = get_2T_mixed_strategy_results(best_params, data_manager, URM_train, strategy='concatenate')
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    model_results = {}

    # --- MODIFIED SAVING LOGIC ---
    # Add metadata to the results dictionary for a complete record
    model_results['STUDY_NAME'] = STUDY_NAME
    model_results['OBJECTIVE_METRIC'] = OBJECTIVE_METRIC
    model_results['METRIC_K'] = METRIC_K
    model_results['STUDY_BEST_VALUE'] = optuna_study.best_value
    model_results['Has_User_Embedding'] = True if USER_EMBEDDING_PATH else False
    model_results['Has_Item_Embedding'] = True if ITEM_EMBEDDING_PATH else False
    
    print('-'* 50)
    print("\n--- Evaluating on Original URM_test ---")
    results, _ = evaluator_test.evaluateRecommender(recommender)
    
    print("\n--- Evaluating on Popularity-Reduced URM_test ---")
    recommender.set_items_to_ignore(items_to_ignore_for_bias_reduction)
    results_filtered, _ = evaluator_test_filtered.evaluateRecommender(recommender)
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

    # Convert the flat dictionary of results to a pandas DataFrame
    df_results = pd.DataFrame([model_results])
    # Order columns to ensure consistent order
    
    df_results = df_results[['STUDY_NAME', 'OBJECTIVE_METRIC', 'METRIC_K', 'STUDY_BEST_VALUE',
                             'Has_User_Embedding', 'Has_Item_Embedding', OBJECTIVE_METRIC, f'{OBJECTIVE_METRIC}_filtered',
                             'DIVERSITY_MEAN_INTER_LIST', 'DIVERSITY_MEAN_INTER_LIST_filtered',
                             'COVERAGE_ITEM', 'COVERAGE_ITEM_filtered', 'NOVELTY', 'NOVELTY_filtered']]

    # Ensure the target directory exists
    SAVING_PATH.mkdir(parents=True, exist_ok=True)
    csv_file_path = SAVING_PATH / "all_model_results.csv"

    # Append DataFrame to the CSV file
    df_results.to_csv(
        csv_file_path,
        mode='a',  # 'a' for append
        header=not csv_file_path.exists(),  # Write header only if file doesn't exist
        index=False,
        sep=';',
        decimal=','
    )
    
    print(f"\nResults appended to: {csv_file_path}")


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Run recommender system experiment with popularity bias reduction and metrics evaluation.")
    parser.add_argument('--use_user_embedding', action='store_true', help="Use user embeddings")
    parser.add_argument('--use_item_embedding', action='store_true', help="Use item embeddings")
    parser.add_argument('--objective_metric', type=str, default='MAP_MIN_DEN', help="Objective metric for evaluation (MAP_MIN_DEN, NDGC)")
    parser.add_argument('--data_path', type=str, help="Path to the dataset")
    parser.add_argument('--user_embedding_path', type=str, help="Path to user embeddings")
    parser.add_argument('--item_embedding_path', type=str, help="Path to item embeddings")
    parser.add_argument('--study_name', type=str, help="Optuna study name")
    parser.add_argument('--optuna_path', type=str, help="Path to the Optuna database")
    parser.add_argument('--saving_path', type=str, help="Path to save results")
    
    parser.add_argument('--model_type', type=str, choices=['TwoTower', 'IALS', 'IALS_with_embeddings', 'ItemFactorLearner', 'UserFactorLearner', '2T_mixed_strategy'], help="Type of model to train")

    args = parser.parse_args()

    USE_USER_EMBEDDING = args.use_user_embedding
    USE_ITEM_EMBEDDING = args.use_item_embedding

    OBJECTIVE_METRIC = args.objective_metric
    METRICS = [OBJECTIVE_METRIC,'DIVERSITY_MEAN_INTER_LIST', "COVERAGE_ITEM", 'NOVELTY']
    OPTUNA_PATH = args.optuna_path
    STUDY_NAME = args.study_name
    SAVING_PATH = Path(args.saving_path)
    DATA_PATH = Path(args.data_path)
    METRIC_K = 10
    

    USER_EMBEDDING_PATH = Path(args.user_embedding_path) if args.user_embedding_path is not None else None
    ITEM_EMBEDDING_PATH = Path(args.item_embedding_path) if args.item_embedding_path is not None else None
    
    MODEL_TYPE = args.model_type

    main()