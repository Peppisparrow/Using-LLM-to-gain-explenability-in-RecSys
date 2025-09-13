from vllm import LLM, SamplingParams
import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- CONFIGURATION ---

# 3. Model settings
#MODEL_NAME = "google/gemma-3-4b-it"
#MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-1M"
MODEL_NAME = "graelo/Qwen2.5-7B-Instruct-1M-AWQ"

# 1. Define your file paths
INPUT_PATH = Path('Prototype/Dataset/ml_small/tuning/history_desc_embedding/user_prompts_DESC.parquet')
OUTPUT_PATH = Path('user_descriptions_from_llm_DESC_ollama_TEST_QWEN_1M.csv')

# 2. Inference settings
API_CALL_DELAY_SECONDS = 0.2  # Lower since local inference
SAVE_INTERVAL = 1 

# --- EXECUTION BLOCK ---

if __name__ == "__main__":
    print("--- Starting User Description Generation Process ---")

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

    # 3. Initialize the vLLM engine
    try:
        llm = LLM(model=MODEL_NAME, dtype="bfloat16", max_model_len=105008)  # Uses BFP16 weights if available
        print(f"✅ Loaded model '{MODEL_NAME}' successfully.")
    except Exception as e:
        print(f"🔴 FATAL: Could not load model: {e}")
        exit()

    # 4. Load user prompts
    df = pd.read_parquet(INPUT_PATH)
    print(f"📂 Loaded {len(df)} total user prompts from '{INPUT_PATH}'.")

    print("-" * 50)
    print(f"Processed user IDs so far: {len(processed_user_ids)}")
    print(f"User ID type: {type(next(iter(processed_user_ids), ''))}")
    print("-" * 50)

    new_users_processed = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating User Descriptions"):

        user_id = int(index)

        # Skip if already processed
        if user_id in processed_user_ids:
            print(f"Skipping user id {user_id}")
            continue

        prompt = row['prompt']

        try:
            # 5. Generate description locally
            print(f"\n➡️ Generating description for user {user_id} with description {prompt[:30]}...")
            
            outputs = llm.generate([prompt])
            description = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else "No description generated"

            # 6. Append result
            user_descriptions.append({'user_id': user_id, 'description': description})
            new_users_processed += 1

            print(f"\n✅ Processed user {user_id}: {description[:80]}...")

            # 7. Incremental save
            if new_users_processed % SAVE_INTERVAL == 0:
                pd.DataFrame(user_descriptions).to_csv(OUTPUT_PATH, index=False)
                print(f"💾 Progress saved for {new_users_processed} new users.")

            time.sleep(API_CALL_DELAY_SECONDS)

        except Exception as e:
            print(f"🔴 Error processing user {user_id}: {e}")
            time.sleep(2)

    # 8. Final save
    try:
        final_df = pd.DataFrame(user_descriptions)
        final_df.sort_values(by='user_id', inplace=True)
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n--- Process Complete ---")
        print(f"✅ All {len(final_df)} user descriptions saved to '{OUTPUT_PATH}'")
    except Exception as e:
        print(f"🔴 Error during final save: {e}")
