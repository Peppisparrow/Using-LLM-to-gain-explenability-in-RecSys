import optuna
import os
import sys

# --- Configuration ---
# 1. Source Database: Where the original study is located.
source_db_path = "/leonardo_work/IscrC_DMG4RS/embednbreakfast/Using-LLM-to-gain-explenability-in-RecSys/Prototype/optuna/optuna_study.db"

# 2. Destination Database: Where the study will be moved. CAN BE THE SAME as the source.
dest_db_path = "/leonardo_work/IscrC_DMG4RS/embednbreakfast/Using-LLM-to-gain-explenability-in-RecSys/Prototype/optuna/DBs/optuna_big_steam.db"

# 3. Study Names: The original name and the new name for the destination.
#    If new_study_name is the same as old_study_name, it will be a simple move/copy.
old_study_name = "TwoTowerProd_BIG_steam_HIST_USER_ITEM_embs_MXBAI"
new_study_name = "TwoTowerProd_HistUser-Item_embs_MXBAI_MAP10" # Change this to rename the study

# 4. Delete Original: Set to True to delete the study from the source DB after a successful copy.
#    Set to False to simply copy the study.
DELETE_ORIGINAL_STUDY = False

# --- Main Logic ---

def move_rename_study(source_db, dest_db, old_name, new_name, delete_original=False):
    """
    Moves or copies a study from a source database to a destination database,
    with an option to rename it.

    Args:
        source_db (str): Path to the source .db file.
        dest_db (str): Path to the destination .db file.
        old_name (str): The name of the study to move.
        new_name (str): The new name for the study in the destination.
        delete_original (bool): If True, deletes the study from the source after copying.
    """
    # --- 1. Validate Paths and Create Connections ---
    if not os.path.exists(source_db):
        print(f"❌ FATAL ERROR: Source database not found at '{source_db}'")
        sys.exit(1)

    # Ensure the destination directory exists
    dest_dir = os.path.dirname(dest_db)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    source_storage_url = f"sqlite:///{source_db}"
    dest_storage_url = f"sqlite:///{dest_db}"

    print(f"Source DB: {source_storage_url}")
    print(f"Destination DB: {dest_storage_url}")
    print("-" * 30)

    # --- 2. Load Source Study ---
    try:
        print(f"Attempting to load study '{old_name}' from source...")
        source_study = optuna.load_study(study_name=old_name, storage=source_storage_url)
        print(f"✅ Successfully loaded '{old_name}' with {len(source_study.trials)} trials.")
    except (ValueError, KeyError):
        print(f"❌ ERROR: Study '{old_name}' not found in the source database. Aborting.")
        return

    # --- 3. Check for Conflicts in Destination ---
    try:
        # Create if it doesn't exist, otherwise load it to check for conflicts
        print(f"Checking if study '{new_name}' already exists in the destination...")
        optuna.load_study(study_name=new_name, storage=dest_storage_url)
        print(f"❌ ERROR: A study named '{new_name}' already exists in the destination database. Aborting to prevent overwrite.")
        return
    except (ValueError, KeyError):
        # This is the desired outcome: the name is available in the destination.
        # A ValueError or KeyError is raised if the study does not exist.
        print(f"✅ The name '{new_name}' is available in the destination database.")
        pass

    # --- 4. Create and Populate New Study ---
    print(f"Creating new study '{new_name}' in the destination...")
    dest_study = optuna.create_study(
        study_name=new_name,
        storage=dest_storage_url,
        direction=source_study.direction,
        sampler=source_study.sampler, # Preserve the sampler
        pruner=source_study.pruner    # Preserve the pruner
    )

    # Copy all trials from the source to the destination
    dest_study.add_trials(source_study.trials)
    print(f"✅ Successfully copied {len(source_study.trials)} trials to '{new_name}'.")
    
    # --- CORRECTED: Copy user and system attributes one by one ---
    for key, value in source_study.user_attrs.items():
        dest_study.set_user_attr(key, value)
    
    for key, value in source_study.system_attrs.items():
        dest_study.set_system_attr(key, value)

    print("✅ Copied user and system attributes.")

    # --- 5. Delete Original Study (if configured) ---
    if delete_original:
        print(f"Attempting to delete original study '{old_name}' from source...")
        try:
            optuna.delete_study(study_name=old_name, storage=source_storage_url)
            print(f"✅ Successfully deleted original study.")
            print(f"\n✨ Study '{old_name}' was successfully MOVED to '{dest_db}' as '{new_name}'.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to delete original study '{old_name}'. It now exists in both databases. Error: {e}")
    else:
        print(f"\n✨ Study '{old_name}' was successfully COPIED to '{dest_db}' as '{new_name}'.")


if __name__ == "__main__":
    move_rename_study(
        source_db=source_db_path,
        dest_db=dest_db_path,
        old_name=old_study_name,
        new_name=new_study_name,
        delete_original=DELETE_ORIGINAL_STUDY
    )
