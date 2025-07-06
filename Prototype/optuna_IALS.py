import os
from functools import partial

import optuna
import pandas as pd

# Defining Recommender
from RecSysFramework.Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from Prototype.data_manager import DataManger


# CONSTANTS
METRIC = 'NDCG'
METRIC_K = 10

class SaveResults(object):
    
    def __init__(self, csv_path ="Prototype/logs/IALS_optuna/trials_results.csv"):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.results_df = pd.DataFrame(columns = ["result"])
    
    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]
        
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True, parents=True)    
        self.results_df = pd.concat([self.results_df, pd.DataFrame([hyperparam_dict])], ignore_index=True)
        self.results_df.to_csv(self.csv_path, index = False)

def objective_function(trial, URM_train, URM_test):
    
    params = {
        "epochs": trial.suggest_int("epochs", 1, 10),
        "num_factors": trial.suggest_int("num_factors", 10, 1200),
        "reg": trial.suggest_float("regularization", 1e-5, 1e-1, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-5, 1.0, log=True),
        "confidence_scaling": trial.suggest_categorical("confidence_scaling", ["linear", "log"]),
        "alpha": trial.suggest_float("alpha", 0.0, 50.0),
    }

    recommender = IALSRecommender(URM_train)
    
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
    
    objective_function_with_data = partial(
        objective_function,
        URM_train=URM_train,
        URM_test=URM_test
    )
        
    optuna_study = optuna.create_study(direction="maximize", study_name="IALS_STUDY", load_if_exists=True, storage="sqlite:///Prototype/optuna_study.db")
            
    save_results = SaveResults()

    optuna_study.optimize(objective_function_with_data,
                        callbacks=[save_results],
                        n_trials = 100)



if __name__ == "__main__":
    main()