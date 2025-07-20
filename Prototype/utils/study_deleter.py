import optuna
import os

# --- Configuration ---
# The name of the study to delete
study_name = "ItemFactorLearner_map10"
# The path to your Optuna database file
#db_path = "/leonardo_work/IscrC_DMG4RS/embednbreakfast/Using-LLM-to-gain-explenability-in-RecSys/Prototype/optuna/optuna_study.db"
db_path = "/leonardo_work/IscrC_DMG4RS/embednbreakfast/Using-LLM-to-gain-explenability-in-RecSys/Prototype/optuna/optuna_study_ML_small.db"
# --- Deletion Logic ---

# First, check if the database file actually exists
if not os.path.exists(db_path):
    print(f"❌ Error: Database file not found at '{db_path}'")
else:
    # Create the connection string for Optuna
    storage_url = f"sqlite:///{db_path}"

    try:
        # Use Optuna's function to delete the study
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"✅ Successfully deleted study '{study_name}' from '{db_path}'")
    except ValueError:
        # Optuna raises a ValueError if the study_name doesn't exist in the database
        print(f"⚠️ Study '{study_name}' not found in the database. No action taken.")
    except Exception as e:
        # Catch other potential issues (e.g., database permissions, corruption)
        print(f"❌ An unexpected error occurred: {e}")