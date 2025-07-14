import optuna
import pandas as pd
import numpy as np
import os
from datetime import timedelta

# --- Configuration ---
# 1. Define the path to your Optuna database
db_path = "/leonardo_work/IscrC_DMG4RS/embednbreakfast/Using-LLM-to-gain-explenability-in-RecSys/Prototype/optuna/DBs/optuna_big_steam.db"

# 2. Define the name for your output CSV file
output_csv_path = "optuna_studies_analysis_report_with_ranges_BIG.csv"

# --- Helper Function ---
def get_parameter_ranges(trials, param_names):
    """
    Calculates the min/max range for numerical parameters and unique values for categorical ones.
    
    Args:
        trials (list): A list of Optuna Trial objects.
        param_names (list): The names of the parameters to analyze.

    Returns:
        dict: A dictionary with parameter names as keys and their range/values as string values.
    """
    ranges = {}
    if not trials:
        return {}

    for p_name in param_names:
        # Collect all values for the parameter, skipping trials where it might be missing
        values = [t.params[p_name] for t in trials if p_name in t.params]
        if not values:
            continue

        # Check if the parameter is numerical or categorical
        if isinstance(values[0], (int, float)):
            min_val, max_val = min(values), max(values)
            if min_val == max_val:
                ranges[p_name] = f"{min_val:.4g}" # Use .4g for concise float formatting
            else:
                ranges[p_name] = f"[{min_val:.4g} to {max_val:.4g}]"
        else:  # Assumed to be categorical
            unique_values = sorted(list(set(map(str, values))))
            ranges[p_name] = f"{{{', '.join(unique_values)}}}"
            
    return ranges

# --- Main Analysis Logic ---
def analyze_studies(db_url, csv_path):
    """
    Analyzes all studies in an Optuna database and saves the results to a CSV file.
    """
    # Check if the database file exists
    if not os.path.exists(db_path):
        print(f"❌ Error: Database file not found at '{db_path}'")
        return

    print(f"🔗 Connecting to database: {db_url}")
    try:
        study_summaries = optuna.get_all_study_summaries(storage=db_url)
    except Exception as e:
        print(f"❌ Could not connect to the database or read studies. Error: {e}")
        return

    if not study_summaries:
        print("⚠️ No studies found in the database.")
        return

    print(f"Found {len(study_summaries)} studies. Analyzing each one...")
    all_studies_data = []

    for summary in study_summaries:
        study_name = summary.study_name
        try:
            study = optuna.load_study(study_name=study_name, storage=db_url)
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

            if not completed_trials:
                print(f"Skipping study '{study_name}' as it has no completed trials.")
                continue

            study_data = {"study_name": study_name}
            param_names = study.best_params.keys()

            is_minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
            sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=not is_minimize)
            top_10_trials = sorted_trials[:10]

            # --- 1. Value range of the top 10 trials ---
            if len(top_10_trials) > 1:
                top_10_values = [t.value for t in top_10_trials]
                study_data["value_range_top10"] = f"{max(top_10_values):.5f}-{min(top_10_values):5f}"
            else:
                study_data["value_range_top10"] = 0.0

            # --- 2. Parameter importance in percentage ---
            try:
                param_importance = optuna.importance.get_param_importances(study)
                total_importance = sum(param_importance.values())
                if total_importance > 0:
                    importance_percent = {p: (v / total_importance) * 100 for p, v in param_importance.items()}
                    study_data["param_importance_percent"] = "; ".join([f"{p}: {v:.1f}%" for p, v in sorted(importance_percent.items(), key=lambda item: item[1], reverse=True)])
                else:
                    study_data["param_importance_percent"] = "N/A"
            except Exception:
                study_data["param_importance_percent"] = "Calculation Error"

            # --- 3. Parameter Ranges (NEW) ---
            study_ranges = get_parameter_ranges(completed_trials, param_names)
            study_data["study_param_ranges"] = "; ".join([f"{p}: {r}" for p, r in study_ranges.items()])
            
            top10_ranges = get_parameter_ranges(top_10_trials, param_names)
            study_data["top10_param_ranges"] = "; ".join([f"{p}: {r}" for p, r in top10_ranges.items()])

            # --- 4. Sbizzarrisciti (Go Wild!) ---
            study_data["direction"] = study.direction.name
            study_data["best_value"] = summary.best_trial.value if summary.best_trial else None
            study_data["n_trials"] = summary.n_trials
            study_data["n_completed_trials"] = len(completed_trials)
            study_data["best_params"] = "; ".join([f"{p}: {v}" for p, v in study.best_params.items()])
            
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            study_data["pruned_trials_percent"] = (len(pruned_trials) / summary.n_trials) * 100 if summary.n_trials > 0 else 0

            if param_importance:
                dominant_param = max(param_importance, key=param_importance.get)
                study_data["dominant_parameter"] = f"{dominant_param} ({importance_percent.get(dominant_param, 0):.1f}%)"
            else:
                study_data["dominant_parameter"] = "N/A"

            all_studies_data.append(study_data)
        except Exception as e:
            print(f"❌ Failed to process study '{study_name}'. Error: {e}")

    if not all_studies_data:
        print("Could not generate report as no studies were successfully analyzed.")
        return
        
    df = pd.DataFrame(all_studies_data)
    
    # Reorder columns for better readability
    column_order = [
        "study_name",
        "best_value",
        "value_range_top10",
        "direction",
        "dominant_parameter",
        "param_importance_percent",
        "top10_param_ranges", # New
        "study_param_ranges", # New
        "best_params",
        "n_trials",
        "n_completed_trials",
        "pruned_trials_percent",
    ]
    df = df[[col for col in column_order if col in df.columns]]

    df.to_csv(csv_path, index=False, sep=';', decimal=',', float_format="%.6f")
    print(f"\n✨ Analysis complete! Report saved to:\n{os.path.abspath(csv_path)}")


if __name__ == "__main__":
    storage_url = f"sqlite:///{db_path}"
    analyze_studies(storage_url, output_csv_path)
