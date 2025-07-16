import os
from functools import partial

import optuna
import pandas as pd

# Defining Recommender
from Prototype.Decoder.ItemFactorLearner_implicit import ImplicitItemFactorLearner
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from Prototype.data_manager import DataManger
from implicit.evaluation import ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares

# CONSTANTS
METRIC = 'MAP_MIN_DEN'
METRIC_K = 10

class SaveResults(object):
    
    def __init__(self, csv_path ="Prototype/prova.csv"):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.results_df = pd.DataFrame(columns = ["result"])
    
    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]
        
        self.results_df = pd.concat([self.results_df, pd.DataFrame([hyperparam_dict])], ignore_index=True)
        self.results_df.to_csv(self.csv_path, index = False)

def objective_function(trial, user_embeddings, URM_train, URM_test):
#    alpha=20.0, reg=1e-2, use_gpu=False, n_jobs=-1):    
    params = {
        "user_factors": user_embeddings,
        "alpha": trial.suggest_float("alpha", 0.0, 50.0),
        "reg": trial.suggest_float("reg", 1e-5, 1e-1, log=True),
    }

    recommender = ImplicitItemFactorLearner(URM_train)
    recommender.fit(**params)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[METRIC_K], verbose=False, exclude_seen=True)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)
    result = result_dict.loc[METRIC_K][METRIC]
    # print("Current {} = {:.4f} with parameters {}".format(METRIC, result, params))

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
    
    result_implicit = ranking_metrics_at_k(
        fake_model,
        URM_train,
        URM_test,
        K=METRIC_K,
        show_progress=False
        
    )['map']
    print("RESULT WITH IMPLICIT MODEL: ", result_implicit)
    print("RESULT WITH PROF EVALUATION: ", result)
    return result

def main():
    data_manager = DataManger(data_path="Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10")
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    user_embeddings = data_manager.get_user_embeddings()
    
    objective_function_with_data = partial(
        objective_function,
        user_embeddings=user_embeddings,
        URM_train=URM_train,
        URM_test=URM_test
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name="prova", load_if_exists=True, storage="sqlite:///Prototype/optuna_study.db")
            
    save_results = SaveResults()

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 10)



if __name__ == "__main__":
    main()