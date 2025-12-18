import os
import time
import ast
import pandas as pd
import numpy as np
import google.generativeai as genai
from tqdm import tqdm
from copy import deepcopy

# --- 1. CONFIGURATION & VARIABLES ---
# It is best practice to set these via environment variables
# e.g., export GOOGLE_API_KEY='your_key_here'
API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")

# File Paths
DATASET_PATH = './dataset/eval.csv'
OUTPUT_CSV = './results/sota/recipe_nlg/evaluation.csv'
OUTPUT_PARQUET = './results/sota/recipe_nlg/evaluation.parquet'

# Model Configuration
MODEL_NAMES = [
    'gemini-2.0-flash', 
    'gemini-2.5-flash', 
    'gemini-2.5-pro'
]

# --- 2. INITIALIZATION ---
def initialize_models(api_key, model_list):
    """Configures GenAI and returns a mapping of model names to objects."""
    genai.configure(api_key=api_key)
    models = {}
    for name in model_list:
        try:
            models[name] = genai.GenerativeModel(name)
        except Exception as e:
            print(f"Error initializing model {name}: {e}")
    return models

# --- 3. UTILITY FUNCTIONS ---
def fix_and_convert_messages(message_str):
    """Cleans raw string message data and converts to list of dicts."""
    try:
        cleaned_str = str(message_str).strip()
        if cleaned_str.startswith('"') and cleaned_str.endswith('"'):
            cleaned_str = cleaned_str[1:-1]
        cleaned_str = cleaned_str.replace('""', '"')
        return ast.literal_eval(cleaned_str)
    except Exception:
        return []

def get_model_response(model_object, prompt, max_retries=3):
    """Handles API calls with basic retry logic and markdown cleaning."""
    for attempt in range(max_retries):
        try:
            response = model_object.generate_content(prompt)
            clean_text = response.text.strip()
            
            # Remove Markdown code blocks if present
            for tag in ["```json", "```"]:
                if clean_text.startswith(tag):
                    clean_text = clean_text[len(tag):]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
            
            return clean_text.strip()
        except Exception as e:
            if any(err in str(e) for err in ["400", "429", "500"]):
                time.sleep(2 ** (attempt + 1)) # Exponential backoff
            else:
                return f"Error: {type(e).__name__}"
    return "Failed after max retries"

# --- 4. MAIN PROCESSING FLOW ---
def main():
    # Load Data
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        # Using the logic from the notebook: skip 4 rows, '?' delimiter
        df = pd.read_csv(DATASET_PATH, delimiter='?', skiprows=4)
    except FileNotFoundError:
        print("Error: Dataset file not found. Check your paths.")
        return

    # Pre-process 'messages' column
    df['messages'] = df['messages'].apply(fix_and_convert_messages)

    # Initialize Models
    models_map = initialize_models(API_KEY, MODEL_NAMES)
    
    # Storage for results
    all_outputs = {name: [] for name in models_map.keys()}
    ground_truths = []

    print(f"Starting inference for models: {list(models_map.keys())}")

    # Process Rows
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Recipes"):
        msgs = row['messages']
        prompt = ""
        gt_text = ""

        # Extraction logic (Prompt: second to last, Ground Truth: last)
        if isinstance(msgs, list) and len(msgs) >= 2:
            prompt = msgs[-2].get('content', '')
            gt_text = msgs[-1].get('content', '')
        
        ground_truths.append(gt_text)

        # Get responses from each model
        for name, model_obj in models_map.items():
            if prompt:
                response = get_model_response(model_obj, prompt)
                all_outputs[name].append(response)
            else:
                all_outputs[name].append(None)

    # --- 5. DATA AGGREGATION & EXPORT ---
    df['ground_truth'] = ground_truths
    for name, outputs in all_outputs.items():
        col_name = f"output_{name.replace('-', '_')}"
        df[col_name] = outputs

    # Create a 'prediction' column (example using the first model's output)
    first_model_col = f"output_{MODEL_NAMES[0].replace('-', '_')}"
    df['prediction'] = df[first_model_col].copy()

    # Save Results
    print(f"Saving results to {OUTPUT_PARQUET}...")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_parquet(OUTPUT_PARQUET, engine='pyarrow')
    
    print("Inference complete.")

if __name__ == "__main__":
    main()