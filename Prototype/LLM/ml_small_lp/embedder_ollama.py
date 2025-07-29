import pandas as pd
from pathlib import Path
import numpy as np
import ollama
from tqdm import tqdm

def generate_and_save_embeddings(csv_path, output_path, model_name='mxbai-embed-large'):
    """
    Reads user descriptions from a CSV, generates embeddings using the Ollama
    backend, and saves them to a .npz file.

    Args:
        csv_path (str or Path): The path to the input CSV file.
                                The file must contain 'user_id' and 'description' columns.
        output_path (str or Path): The path where the output .npz file will be saved.
        model_name (str): The name of the embedding model to use with Ollama.
                          Ensure this model is available in your Ollama setup.
    """
    print(f"Reading data from {csv_path}...")
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    # Ensure the required columns exist
    if 'user_id' not in df.columns or 'description' not in df.columns:
        print("Error: CSV file must contain 'user_id' and 'description' columns.")
        return

    # Drop rows where the description is missing to avoid errors
    print("Processing data...")
    print(f"Length of DataFrame before dropping NaNs: {len(df)}")
    df.dropna(subset=['description'], inplace=True)
    print(f"Length of DataFrame after dropping NaNs: {len(df)}")
    
    # Convert descriptions to a list of strings
    descriptions = df['description'].tolist()
    user_ids = df['user_id'].tolist()

    # Check if there is any data to process
    if not descriptions:
        print("No descriptions found to embed. Exiting.")
        return
        
    # Generate embeddings for the descriptions using Ollama
    print(f"Generating embeddings using Ollama model: {model_name}... (This may take a while)")
    
    embeddings_list = []
    try:
        # Use tqdm for a progress bar as we process descriptions one by one
        for desc in tqdm(descriptions, desc="Generating Embeddings"):
            response = ollama.embeddings(model=model_name, prompt=desc)
            embeddings_list.append(response['embedding'])
    except Exception as e:
        print(f"\nAn error occurred while communicating with Ollama: {e}")
        print(f"Please ensure the Ollama service is running and the model '{model_name}' is pulled.")
        return

    # Convert the list of embeddings to a single NumPy array
    embeddings = np.array(embeddings_list)
    print("Embeddings generated successfully.")

    # Save the user_ids and embeddings to a compressed .npz file
    # This format is efficient for storing numpy arrays.
    print(f"Saving user IDs and embeddings to {output_path}...")
    np.savez_compressed(output_path, user_id=user_ids, embeddings=embeddings)
    print("File saved successfully.")

    # Optional: Verify the saved file
    print("\nVerification:")
    with np.load(output_path) as data:
        print(f"Number of user IDs saved: {len(data['user_id'])}")
        print(f"Shape of embeddings array: {data['embeddings'].shape}")


if __name__ == '__main__':
    # Define the input and output file paths using pathlib
    # The .expanduser() method is used to resolve the '~' to the user's home directory.
    DATA_PATH = Path('~/Downloads/DATA/ML_small/').expanduser()
    INPUT_CSV_FILE = DATA_PATH / 'user_descriptions_from_llm_FIXED_MISS.csv'
    
    # Define the Ollama model to use for embeddings
    #OLLAMA_MODEL = 'mxbai-embed-large' 
    #OLLAMA_MODEL = 'jeffh/intfloat-multilingual-e5-large-instruct:f32'
    OLLAMA_MODEL = 'hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF'

    #OUTPUT_NPZ_FILE = DATA_PATH / f'user_embeddings_mxbai.npz'
    #OUTPUT_NPZ_FILE = DATA_PATH / f'user_embeddings_e5_multilingual.npz'
    OUTPUT_NPZ_FILE = DATA_PATH / f'user_embeddings_qwen_06.npz'

    # Run the function
    generate_and_save_embeddings(INPUT_CSV_FILE, OUTPUT_NPZ_FILE, model_name=OLLAMA_MODEL)

    # Example of how to load the data back
    # print("\n--- How to load the data back ---")
    # with np.load(OUTPUT_NPZ_FILE) as data:
    #     loaded_user_ids = data['user_id']
    #     loaded_embeddings = data['embeddings']
    #     print("First 5 User IDs:", loaded_user_ids[:5])
    #     print("Shape of loaded embeddings:", loaded_embeddings.shape)