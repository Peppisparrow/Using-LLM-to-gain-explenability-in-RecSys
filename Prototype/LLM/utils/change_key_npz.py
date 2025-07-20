import numpy as np
import os

def rename_user_ids_in_npz(input_path, output_path):
    """
    Loads an NPZ file, renames the 'user_ids' key to 'new_user_ids',
    and saves the modified data to a new NPZ file.

    Args:
        input_path (str): The path to the input NPZ file.
        output_path (str): The path where the modified NPZ file will be saved.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    try:
        # Load the npz file
        data = np.load(input_path)

        # Create a new dictionary to hold the modified data
        new_data = {}

        # Flag to check if 'user_ids' was found and renamed
        user_ids_found_and_renamed = False

        # Iterate through the items in the loaded npz file
        for key, value in data.items():
            if key == 'user_ids':
                new_data['user_id'] = value
                user_ids_found_and_renamed = True
            else:
                new_data[key] = value

        # Close the loaded npz file to free up resources
        data.close()

        # Save the modified data to the specified output path
        np.savez(output_path, **new_data)

        if user_ids_found_and_renamed:
            print(f"Successfully renamed 'user_ids' to 'new_user_ids' from '{input_path}' to '{output_path}'")
        else:
            print(f"Warning: 'user_ids' key not found in '{input_path}'. Data saved to '{output_path}' without renaming.")

    except Exception as e:
        print(f"An error occurred: {e}")

input_path = '/leonardo_work/IscrC_DMG4RS/embednbreakfast/Prototype/Dataset/steam/filtering_no_desc_giappo_corean_k10/big/history_embeddings/user_embeddings_compressed.npz'
output_path = '/leonardo_work/IscrC_DMG4RS/embednbreakfast/Using-LLM-to-gain-explenability-in-RecSys/hist_user_embeddings_MXBAI.npz'
rename_user_ids_in_npz(input_path, output_path)