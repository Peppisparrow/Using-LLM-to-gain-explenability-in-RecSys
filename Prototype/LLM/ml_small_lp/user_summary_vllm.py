from vllm import LLM, SamplingParams
import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

# Set environment variables for vLLM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['VLLM_ATTENTION_BACKEND'] = 'DUAL_CHUNK_FLASH_ATTN'
os.environ['VLLM_USE_V1'] = '0'

# --- CONFIGURATION ---
API_CALL_DELAY_SECONDS = 0.2  # Lower since local inference
SAVE_INTERVAL = 1

INPUT_PATH = Path('Prototype/Dataset/ml_small/tuning/history_desc_embedding/local_summaries/user_prompts_DESC_TESTED.parquet')
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
OUTPUT_PATH = Path('Prototype/Dataset/ml_small/tuning/history_desc_embedding/local_summaries/user_desc_from_llm_QWEN_30B_128K_VLLM.csv')

# Sampling parameters
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    max_tokens=1024  # At most 1024 tokens for generation
)

# --- EXECUTION BLOCK ---

if __name__ == "__main__":
    # parser = ArgumentParser(description="Generate user descriptions using vLLM backend.")
    # parser.add_argument('--input', type=str, default=str(INPUT_PATH), help='Path to the input parquet file with user prompts.')
    # parser.add_argument('--output', type=str, default=str(OUTPUT_PATH), help='Path to the output CSV file for user descriptions.')
    # parser.add_argument('--model', type=str, default=MODEL_NAME, help='Model path to use for generation.')
    # parser.add_argument('--delay', type=float, default=API_CALL_DELAY_SECONDS, help='Delay between API calls in seconds.')

    # args = parser.parse_args()
    
    # INPUT_PATH = Path(args.input)
    # OUTPUT_PATH = Path(args.output)
    # MODEL_NAME = args.model
    # API_CALL_DELAY_SECONDS = args.delay

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
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 3. Initialize the vLLM engine with specified options
    try:
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=4,
            max_model_len=131072,  # Match max_num_batched_tokens for consistency
            enable_chunked_prefill=True,
            max_num_batched_tokens=131072,
            enforce_eager=True,
            max_num_seqs=1,
            gpu_memory_utilization=0.7,
            quantization="gptq",  # 4-bit quantization
            dtype="half"  # Use fp16 for better compatibility
        )
        print(f"✅ Loaded model '{MODEL_NAME}' successfully.")
        print(f"📊 Sampling params: temp={SAMPLING_PARAMS.temperature}, top_p={SAMPLING_PARAMS.top_p}, top_k={SAMPLING_PARAMS.top_k}, max_tokens={SAMPLING_PARAMS.max_tokens}")
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
            continue

        prompt = row['prompt']

        try:
            # 5. Generate description locally with sampling parameters
            outputs = llm.generate([prompt], SAMPLING_PARAMS)
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