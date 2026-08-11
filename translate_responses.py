import pandas as pd
import requests
import json
import re
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = "raw_responses.csv"                   # Path to your existing CSV file
OUTPUT_FILE = "translated_responses.csv"           # Path to save updated output
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gpt-oss:latest"
RUN_PREVIEW_ONLY = True  # Set to False when ready for the full run!
TEST_LIMIT = 2           # Number of Chinese rows to test in preview mode

def extract_clean_translation(raw_response):
    """
    Strips out reasoning/thinking blocks (<think>...</think> or 'Thinking...')
    and extracts ONLY the final English translation.
    """
    if not raw_response:
        return ""
    
    # 1. Remove XML style <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
    
    # 2. Remove CLI style 'Thinking......done thinking.' blocks
    cleaned = re.sub(r'Thinking\b.*?\b\.\.\.done thinking\.', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # 3. Strip residual headers
    cleaned = re.sub(r'^(Translation|English Translation|Output):\s*', '', cleaned.strip(), flags=re.IGNORECASE)
    
    return cleaned.strip()

def translate_chinese_to_english(text):
    """
    Translates Chinese text to English using gpt-oss via Ollama Chat API.
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system", 
                "content": "You are a Chinese-to-English translator. Output ONLY English. Do NOT provide commentary or explanations."
            },
            {
                "role": "user", 
                "content": f"Translate: {text}"
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
        if response.status_code == 200:
            res_json = response.json()
            raw_content = res_json.get("message", {}).get("content", "").strip()
            
            # Clean reasoning blocks out of the output
            clean_english = extract_clean_translation(raw_content)
            return clean_english
        else:
            print(f"\n[HTTP Error {response.status_code}] Failed to translate.")
            return ""
    except Exception as e:
        print(f"\n[Execution Error] {e}")
        return ""

def main():
    print(f"Loading data from '{INPUT_FILE}'...")
    df_original = pd.read_csv(INPUT_FILE)

    required_cols = ['prompt_language', 'cleaned_candidate_response']
    for col in required_cols:
        if col not in df_original.columns:
            raise ValueError(f"Required column '{col}' not found in dataset!")

    df_updated = df_original.copy()

    if 'chinese_translation' not in df_updated.columns:
        df_updated['chinese_translation'] = ""

    # Filter for Chinese prompts
    chinese_mask = df_updated['prompt_language'].astype(str).str.strip().str.lower() == 'chinese'
    all_chinese_indices = df_updated[chinese_mask].index

    target_indices = all_chinese_indices[:TEST_LIMIT] if RUN_PREVIEW_ONLY else all_chinese_indices

    print(f"Total Chinese responses found: {len(all_chinese_indices)}")
    mode_str = f"PREVIEW MODE (First {len(target_indices)} rows)" if RUN_PREVIEW_ONLY else "FULL RUN"
    print(f"Executing: {mode_str}...")

    for idx in tqdm(target_indices, desc="Translating"):
        original_text = df_updated.at[idx, 'cleaned_candidate_response']
        translation = translate_chinese_to_english(original_text)
        
        df_updated.at[idx, 'chinese_translation'] = translation
        
        print(f"\n--- Row Index {idx} ---")
        print(f"[Original Chinese]: {str(original_text)[:70]}...")
        print(f"[Clean English Translation]: {translation}\n")

    # Save to CSV
    save_path = "preview_translated.csv" if RUN_PREVIEW_ONLY else OUTPUT_FILE
    df_updated.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\nExecution finished! Saved to '{save_path}'.")
    if RUN_PREVIEW_ONLY:
        print("To run all Chinese rows, set `RUN_PREVIEW_ONLY = False` at the top of the script!")

if __name__ == "__main__":
    main()