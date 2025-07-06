import os
from functools import partial

import optuna
import pandas as pd

# Defining Recommender
from Prototype.Decoder.ItemFactorLearnerGPU import ItemFactorLearner
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from Prototype.data_manager import DataManger


# CONSTANTS
METRIC = 'NDCG'
METRIC_K = 10

class SaveResults(object):
    
    def __init__(self, csv_path ="Prototype/logs/RecommenderDecoder/trials_results.csv"):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.results_df = pd.DataFrame(columns = ["result"])
    
    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]
        
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True, parents=True)    
        self.results_df = pd.concat([self.results_df, pd.DataFrame([hyperparam_dict])], ignore_index=True)
        self.results_df.to_csv(self.csv_path, index = False)

def objective_function(trial, user_embeddings, URM_train, URM_test):
    
    params = {
        "user_factors": user_embeddings,
        "use_gpu": True,  # Use GPU for training
    }

    recommender = ItemFactorLearner(URM_train)
    recommender.fit(**params)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[METRIC_K], verbose=False)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)

    result = result_dict.loc[METRIC_K][METRIC]
    print("Current {} = {:.4f} with parameters {}".format(METRIC, result, params))
    
    return result

def main():
    data_manager = DataManger(data_path="Prototype/data")
    URM_train = data_manager.get_URM_train()
    URM_test = data_manager.get_URM_test()
    user_embeddings = data_manager.get_user_embeddings()
    
    objective_function_with_data = partial(
        objective_function,
        user_embeddings=user_embeddings,
        URM_train=URM_train,
        URM_test=URM_test
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name="RecommenderDecoder_Study", load_if_exists=True, storage="sqlite:///Prototype/optuna_study.db")
            
    save_results = SaveResults()

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 100)



if __name__ == "__main__":
    main()