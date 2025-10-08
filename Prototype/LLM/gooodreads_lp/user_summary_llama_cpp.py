import pandas as pd
import requests
import time
from pathlib import Path
from tqdm import tqdm
import json
from argparse import ArgumentParser

# --- CONFIGURATION ---

# 1. Define your file paths
INPUT_PATH = None
OUTPUT_PATH = None

USE_ENHANCING_PROMPT = True

# 2. llama.cpp server settings
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"  # Use chat endpoint for proper instruction following
API_CALL_DELAY_SECONDS = 0.1
SAVE_INTERVAL = 1

# 3. Model generation parameters
GENERATION_PARAMS = {
    "max_tokens": 1024,
}

ENHANCING_PROMPT = """
Style requirements:
Write in continuous prose (3–6 paragraphs).
Do not use headings, bullet points, or Markdown formatting.
Keep the tone professional, fluent, and cinematic — like a cultural critic describing a reader’s personality through their literary tastes.
Avoid overly ornate or academic language.
Keep the length around 400 words.

Example tone:
“Your literary taste gravitates toward intricate narratives and deeply human themes. You are drawn to stories that probe the subtleties of emotion, identity, and morality — whether through sweeping historical epics, sharp psychological portraits, or intimate domestic dramas. Books for you are not mere escapes but mirrors that reflect the contradictions of being alive…”
"""

# --- EXECUTION BLOCK ---

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default=str(INPUT_PATH), help="Path to input parquet file with user prompts")
    parser.add_argument("--output_path", type=str, default=str(OUTPUT_PATH), help="Path to output CSV file for user descriptions")
    parser.add_argument("--llama_server_url", type=str, default=LLAMA_SERVER_URL, help="URL for the llama.cpp server")

    args = parser.parse_args()
    INPUT_PATH = Path(args.input_path)
    OUTPUT_PATH = Path(args.output_path)
    LLAMA_SERVER_URL = args.llama_server_url

    print(f"--- Configuration ---")
    print(f"Input Path: {INPUT_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"llama-server URL: {LLAMA_SERVER_URL}")
    print(f"API Call Delay (s): {API_CALL_DELAY_SECONDS}")
    print(f"Generation Params: {GENERATION_PARAMS}")
    print(f"Use Enhancing Prompt: {USE_ENHANCING_PROMPT}")
    print(f"---------------------\n")
    
    print("--- Starting User Description Generation Process (llama.cpp Backend) ---")

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

    # 3. Check connection to llama-server
    try:
        health_response = requests.get("http://localhost:8080/health", timeout=5)
        if health_response.status_code == 200:
            print(f"✅ Connected to llama-server at {LLAMA_SERVER_URL}")
        else:
            print(f"⚠️ Warning: llama-server responded with status {health_response.status_code}")
    except Exception as e:
        print(f"🔴 FATAL: Could not connect to llama-server. Please ensure the server is running. Error: {e}")
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
        
        if USE_ENHANCING_PROMPT:
            prompt += "\n\n" + ENHANCING_PROMPT.strip() + "\n\n"

        try:
            # 5. Generate description using chat completions API with streaming
            payload = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": True,
                **GENERATION_PARAMS
            }
            
            response = requests.post(
                LLAMA_SERVER_URL,
                json=payload,
                stream=True,
                timeout=900
            )
            
            if response.status_code == 200:
                description = ""
                token_pbar = tqdm(desc=f"User {user_id} tokens", unit=" tok", leave=False, position=1)
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            json_str = line[6:].strip()
                            if json_str == '[DONE]':
                                break
                            try:
                                data = json.loads(json_str)
                                # OpenAI-style chat completions streaming format
                                if 'choices' in data and len(data['choices']) > 0:
                                    choice = data['choices'][0]
                                    # Check both 'delta' (streaming) and 'message' (non-streaming fallback)
                                    if 'delta' in choice:
                                        content = choice['delta'].get('content')
                                    elif 'message' in choice:
                                        content = choice['message'].get('content')
                                    else:
                                        content = None
                                    
                                    if content is not None:
                                        description += content
                                        token_pbar.update(1)
                            except json.JSONDecodeError:
                                continue
                
                token_pbar.close()
                description = description.strip()
                if not description:
                    #description = "No description generated"
                    continue
            else:
                print(f"\n⚠️ Warning: Server returned status {response.status_code} for user {user_id}")
                #description = "Error: Generation failed"
                continue

            # 6. Append result
            user_descriptions.append({'user_id': user_id, 'description': description})
            new_users_processed += 1

            # 7. Incremental save
            if new_users_processed > 0 and new_users_processed % SAVE_INTERVAL == 0:
                pd.DataFrame(user_descriptions).to_csv(OUTPUT_PATH, index=False)

            time.sleep(API_CALL_DELAY_SECONDS)

        except requests.exceptions.Timeout:
            print(f"\n🔴 Timeout processing user {user_id}")
            user_descriptions.append({'user_id': user_id, 'description': "Error: Timeout"})
            time.sleep(2)
        except Exception as e:
            print(f"\n🔴 Error processing user {user_id}: {e}")
            user_descriptions.append({'user_id': user_id, 'description': f"Error: {str(e)}"})
            time.sleep(2)

    # 8. Final save
    try:
        final_df = pd.DataFrame(user_descriptions)
        if not final_df.empty:
            final_df.sort_values(by='user_id', inplace=True)
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n--- Process Complete ---")
        print(f"✅ All {len(final_df)} user descriptions saved to '{OUTPUT_PATH}'")
        print(f"📊 Total processed: {len(final_df)} users ({new_users_processed} new)")
    except Exception as e:
        print(f"🔴 Error during final save: {e}")