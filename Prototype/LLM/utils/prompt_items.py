import pandas as pd
import json
from tqdm import tqdm

# --- FILE PATHS ---
# Adjust these paths to match your file locations
data_dir = 'Dataset/steam/filtering_no_desc_giappo_corean_k10/mid'
METADATA_PATH = 'Dataset/steam/games_metadata.json'
GAMES_PATH = f'{data_dir}/games.csv'
OUTPUT_PATH = f'{data_dir}/game_prompts.json' # Output file for individual game prompts

# --- Helper Functions ---

def _load_game_metadata(path):
    """
    Loads game descriptions from the games_metadata.json file into a dictionary.
    
    Args:
        path (str): The path to the games_metadata.json file.

    Returns:
        dict: A dictionary mapping app_id to its description. Returns None on error.
    """
    metadata_dict = {}
    print("Loading game metadata...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading metadata"):
                try:
                    data = json.loads(line)
                    # Store description if both app_id and description exist
                    if 'app_id' in data and 'description' in data:
                        metadata_dict[data['app_id']] = data['description']
                except json.JSONDecodeError:
                    # Skip lines that are not valid JSON
                    continue
        return metadata_dict
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at '{path}'")
        return None

def _create_single_game_prompt(title, description):
    """
    Creates a detailed prompt for a single game to be analyzed by an LLM.

    Args:
        title (str): The title of the game.
        description (str): The description of the game.

    Returns:
        str: A formatted prompt string.
    """
    instruction = (
        "You are an expert video game analyst. Your task is to create a concise "
        "summary of the following video game. Focus on identifying its key features, "
        "genre, gameplay mechanics, themes, and overall setting based on its "
        "title and description."
    )
    
    # Clean up the description by removing extra whitespace
    clean_description = " ".join(description.split())
    
    # Assemble the final prompt
    prompt = (
        f"Title: {title}\n"
        f"Description: {clean_description}\n\n"
    )
    return prompt

# --- Main Function ---

def generate_prompts_for_all_games(games_path, metadata_path):
    """
    Orchestrates the process of generating a prompt for every game in games.csv.
    """
    # 1. Load the game metadata (descriptions) once.
    game_metadata_db = _load_game_metadata(metadata_path)
    if game_metadata_db is None:
        print("Aborting process due to failure in loading metadata.")
        return None

    # 2. Load the primary list of games.
    try:
        print("Loading the list of games to process...")
        games_df = pd.read_csv(games_path, usecols=['app_id', 'title'])
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: Could not load or process '{games_path}': {e}")
        return None

    # 3. Iterate through each game and generate its specific prompt.
    all_game_prompts = {}
    print("\nGenerating a prompt for each game...")
    
    # Use tqdm to show a progress bar over the DataFrame rows
    for _, row in tqdm(games_df.iterrows(), total=len(games_df), desc="Creating Game Prompts"):
        app_id = row['app_id']
        title = row['title']
        
        # Retrieve the description, providing a default if not found
        description = game_metadata_db.get(app_id, "No description available.")
        
        # Create the prompt for the current game
        prompt = _create_single_game_prompt(title, description)
        
        # Store the prompt in the dictionary with app_id as the key (as a string for JSON compatibility)
        all_game_prompts[str(app_id)] = prompt

    return all_game_prompts

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting: Individual Game Prompt Generation ---")
    
    # Execute the main function to generate all prompts
    generated_prompts = generate_prompts_for_all_games(
        games_path=GAMES_PATH,
        metadata_path=METADATA_PATH
    )
    
    # 4. If prompts were generated, save them to the output file.
    if generated_prompts:
        try:
            print(f"\nSaving {len(generated_prompts)} prompts to JSON file...")
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                # Use indent for a readable, pretty-printed JSON file
                json.dump(generated_prompts, f, indent=4)
            print(f"--- PROCESS COMPLETED ---")
            print(f"Prompts have been successfully saved to: '{OUTPUT_PATH}'")
        except IOError as e:
            print(f"ERROR: Failed to write to output file at '{OUTPUT_PATH}': {e}")
    else:
        print("\nNo prompts were generated. Please check the logs for any errors.")