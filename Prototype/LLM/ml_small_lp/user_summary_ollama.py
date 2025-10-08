import ollama
import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

# --- CONFIGURATION ---

# 1. Define your file paths
INPUT_PATH = Path('Prototype/Dataset/ml_small/tuning/history_desc_embedding/local_summaries/user_prompts_DESC_TESTED.parquet')

# 2. Inference settings
API_CALL_DELAY_SECONDS = 0.1  # Delay between calls to the local Ollama server
SAVE_INTERVAL = 1  # Save after every 1 user to reduce disk I/O
CONTEXT_LENGTH = 128000  # Context length for the model   


#OLLAMA_MODEL_NAME = "gemma3:12b-it-qat"
#OUTPUT_PATH = Path('Prototype/Dataset/ml_small/tuning/history_desc_embedding/qwen_summaries/user_desc_from_llm_Gemma3_12B_128K.csv')
#OLLAMA_MODEL_NAME = "myaniu/qwen2.5-1m:7b-instruct-q4_K_M"
#OLLAMA_MODEL_NAME = "hf.co/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF:Q6_K"
#OLLAMA_MODEL_NAME = "hf.co/bartowski/Qwen2.5-14B-Instruct-1M-GGUF:Q4_K_M"
#OUTPUT_PATH = Path('Prototype/Dataset/ml_small/tuning/history_desc_embedding/qwen_summaries_tested/user_desc_from_llm_QWEN_1M_OLLAMA_PROMPT_TESTAGAIN_14B_Q4.csv')
#OLLAMA_MODEL_NAME = "hf.co/bartowski/Qwen2.5-14B-Instruct-1M-GGUF:Q5_K_S"
#OUTPUT_PATH = Path('Prototype/Dataset/ml_small/tuning/history_desc_embedding/qwen_summaries_tested/user_desc_from_llm_QWEN_1M_OLLAMA_PROMPT_TESTAGAIN_14B_Q5.csv')
OLLAMA_MODEL_NAME = "hf.co/bartowski/Qwen2.5-14B-Instruct-1M-GGUF:Q6_K"
OUTPUT_PATH = Path('Prototype/Dataset/ml_small/tuning/history_desc_embedding/local_summaries/user_desc_from_llm_QWEN_1M_14B_Q6_128K.csv')

# --- EXECUTION BLOCK ---

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate user descriptions using Ollama LLM backend.")
    parser.add_argument('--input', type=str, default=str(INPUT_PATH), help='Path to the input parquet file with user prompts.')
    parser.add_argument('--output', type=str, default=str(OUTPUT_PATH), help='Path to the output CSV file for user descriptions.')
    parser.add_argument('--model', type=str, default=OLLAMA_MODEL_NAME, help='Ollama model name to use for generation.')
    parser.add_argument('--delay', type=float, default=API_CALL_DELAY_SECONDS, help='Delay between API calls in seconds.')
    parser.add_argument('--context_length', type=int, default=CONTEXT_LENGTH, help='Context length for the model.')

    args = parser.parse_args()
    
    INPUT_PATH = Path(args.input)
    OUTPUT_PATH = Path(args.output)
    OLLAMA_MODEL_NAME = args.model
    API_CALL_DELAY_SECONDS = args.delay
    CONTEXT_LENGTH = args.context_length
    
            
    OLLAMA_OPTIONS = {
        'num_ctx': CONTEXT_LENGTH,
    }
    
    print(f"--- Configuration ---")
    print(f"Input Path: {INPUT_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Ollama Model: {OLLAMA_MODEL_NAME}")
    print(f"API Call Delay (s): {API_CALL_DELAY_SECONDS}")
    print(f"Context Length: {CONTEXT_LENGTH}")
    print(f"---------------------\n")
    
    print("--- Starting User Description Generation Process (Ollama Backend) ---")

    # 1. Check if the input file exists
    if not INPUT_PATH.exists():
        print(f"🔴 FATAL: Input file not found at '{INPUT_PATH}'")
        exit()

    # 2. Load existing results for caching
    user_descriptions = []
    processed_user_ids = set()
    if OUTPUT_PATH.exists():
        print(f"🔎 Found existing output file at '{OUTPUT_PATH}'.")
        try:
            existing_df = pd.read_csv(OUTPUT_PATH, dtype={'user_id': int})
            if not existing_df.empty:
                user_descriptions = existing_df.to_dict('records')
                processed_user_ids = set(existing_df['user_id'])
                print(f"✅ Loaded {len(processed_user_ids)} previously generated descriptions. Resuming...")
            else:
                print(f"⚠️ Warning: Output file '{OUTPUT_PATH}' is empty. Starting from scratch.")
        except Exception as e:
            print(f"🔴 Error reading existing output file: {e}. Starting from scratch.")
    else:
        print("📝 No existing output file found. Starting a new session.")

    # 3. Check connection to Ollama server
    try:
        ollama.list()
        print(f"✅ Connected to Ollama. Using model '{OLLAMA_MODEL_NAME}'.")
    except Exception as e:
        print(f"🔴 FATAL: Could not connect to Ollama. Please ensure the Ollama server is running. Error: {e}")
        exit()

    # 4. Load user prompts
    df = pd.read_parquet(INPUT_PATH)
    print(f"📂 Loaded {len(df)} total user prompts from '{INPUT_PATH}'.")

    new_users_processed = 0
    # Use tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating User Descriptions"):
        user_id = int(index)

        # Skip if already processed
        if user_id in processed_user_ids:
            continue

        prompt = row['prompt']

        try:
            # 5. Generate description using Ollama
            # The ollama.generate call is blocking and handles the request to the server
            response = ollama.generate(
                model=OLLAMA_MODEL_NAME,
                prompt=prompt,
                options=OLLAMA_OPTIONS
            )
            
            description = response['response'].strip() if response and 'response' in response else "No description generated"

            # 6. Append result
            user_descriptions.append({'user_id': user_id, 'description': description})
            new_users_processed += 1

            # Optional: Log progress to the console
            # tqdm provides a better visual progress indicator
            # print(f"\n✅ Processed user {user_id}: {description[:80]}...")

            # 7. Incremental save
            if new_users_processed > 0 and new_users_processed % SAVE_INTERVAL == 0:
                pd.DataFrame(user_descriptions).to_csv(OUTPUT_PATH, index=False)
                # print(f"💾 Progress saved for {new_users_processed} new users.") # tqdm makes this verbose

            time.sleep(API_CALL_DELAY_SECONDS)

        except Exception as e:
            print(f"🔴 Error processing user {user_id}: {e}")
            # Wait a bit longer if an error occurs
            time.sleep(2)

    # 8. Final save
    try:
        final_df = pd.DataFrame(user_descriptions)
        # It's good practice to sort to ensure consistent output order
        if not final_df.empty:
            final_df.sort_values(by='user_id', inplace=True)
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n--- Process Complete ---")
        print(f"✅ All {len(final_df)} user descriptions saved to '{OUTPUT_PATH}'")
    except Exception as e:
        print(f"🔴 Error during final save: {e}")
