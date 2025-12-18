#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from argparse import ArgumentParser
import evaluate
import os
import json
import io
from sklearn.model_selection import train_test_split
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline,Trainer
from sentence_transformers import SentenceTransformer, util
from peft import LoraConfig, TaskType,get_peft_model
from datasets import Dataset, load_dataset,load_from_disk
import torch
from trl import SFTTrainer
from itertools import zip_longest
from tqdm import tqdm
import gc
from dataclasses import dataclass, field
from typing import Dict
import editdistance
import datasets
from loss import FocalLoss, GDiceLoss, SelfAdjDiceLoss, lovasz_softmax_flat
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers.trainer_utils import seed_worker

from utils import MetricsCalculator
import pandas as pd
import os
from tqdm import tqdm

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import langextract as lx
import traceback


# In[1]:


import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np
tqdm.pandas(desc="Extracting...")


# In[8]:


import logging

# Get the specific logger that is causing the issue by its full name
debug_logger = logging.getLogger("langextract.debug")

# Set its level to INFO to suppress DEBUG messages
debug_logger.setLevel(logging.WARNING)
from absl import logging as logging2

# --- Add this configuration before you run inference ---

# Set the verbosity for the absl logger to INFO.
# This will show INFO, WARNING, and ERROR messages, but hide DEBUG.
logging2.set_verbosity(logging2.WARNING)


# In[9]:


def get_dict_NER_ingredients_quantity(data_list):
    """
    Converts a list of dictionaries into a new dictionary where the key is the ingredient
    and the value is a dictionary containing the quantity and unit.

    Args:
        data_list (list): A list of dictionaries, where each dictionary has keys
                          'quantity', 'unit', and 'ingredient'.

    Returns:
        pd.Series: A Series containing three items: a dictionary of ingredients,
                   a list of ingredient names, and a properly formatted ingredient string.
    """
    try:
        # Filter out invalid entries first for cleaner logic
        valid_items = [
            item for item in data_list 
            if item and item.get("ingredient") and item.get("quantity") and item.get("unit")
        ]

        ingredients_quantity_dict = {
            item['ingredient']: {
                'quantity': item['quantity'],
                'unit': item['unit']
            }
            for item in valid_items
        }

        ingredients_name_list = list(ingredients_quantity_dict.keys())

        # Correctly format the ingredients string
        # This creates a string like: ["1 chicken", "600g chicken gravy", ...]
        ingredients_list_for_string = [
            f"{item['quantity']}{item['unit'] if (item['unit']=='g' or item['unit']=='ml') else ''} {item['ingredient']}"
            for item in valid_items
        ]
        
        # Use json.dumps to get a properly formatted JSON string
        ingredients_string = json.dumps(ingredients_list_for_string)

        return pd.Series([ingredients_quantity_dict, ingredients_name_list, ingredients_string])
    except Exception as e:
        print(f"Error processing data: {e}")
        print(data_list)
        return pd.Series([None, None, None])


# In[10]:


def safe_json_loads(text):
    """
    Tries to decode a JSON string. 
    Returns the parsed object if successful, otherwise returns None.
    """
    # Ensure the input is a string, otherwise json.loads will fail
    if not isinstance(text, str):
        return None

    try:
        text = text.replace("directions","instructions")
        return json.loads(text)
    except json.JSONDecodeError:
        # Return None if the string is not valid JSON
        return None


# In[11]:


def extract_ingredients(json_data):
    """
    Safely extracts the 'ingredients' field from a dictionary.
    
    Returns the list of ingredients if the key is found, 
    otherwise returns an empty string.
    """
    try:
        # Try to access the 'ingredients' key
        return json_data["ingredients"]
    except Exception as e:
        return []


# In[12]:


def extract_ingredients_entities(ingredients):
    """
    Extracts the 'ingredients' entities using LangExtract tool.
    """
    try:
        result = lx.extract(
               text_or_documents=ingredients,
               prompt_description=prompt,
               examples=examples,
               model_id=model_id,
               model_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
               format_type=lx.data.FormatType.JSON,
               temperature=temperature,
               extraction_passes=1,
               use_schema_constraints=True,
           )
        extractions = [res.attributes for res in result.extractions]
        return extractions
    except Exception as e:
        print(e)
        return []


# In[13]:


file_path = os.path.join("dataset", "full_dataset.csv")
on_bad_lines_setting = 'skip'

# --- Step 1: Get the total number of rows for the progress bar ---
# This is a memory-efficient way to count lines in a large file.
total_rows = sum(1 for row in open(file_path, 'r'))

# --- Step 2: Read the CSV in chunks with a progress bar ---
chunk_size = 10000  # Adjust chunk size based on your memory
chunks = []

# Use tqdm to create a progress bar
with tqdm(total=total_rows, desc="Loading CSV") as pbar:
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, on_bad_lines=on_bad_lines_setting):
        chunks.append(chunk)
        pbar.update(len(chunk))
print("\nConcatenating chunks...")
df_recipe_nlg = pd.concat(chunks, axis=0)
df_baseline = df_recipe_nlg.head(6250).copy()


# In[14]:


df_baseline


# The next step is to apply NER on the ingredient list and  convert it to metric system.

# In[15]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# In[4]:


prompt = """You are a precision data extraction tool specializing in culinary text. Your task is to analyze a list of ingredients and extract specific entities into a structured JSON format.

Instructions:

For each ingredient in the input list, extract three fields: quantity, unit, and ingredient.

Quantity: Extract the numerical value. Convert fractions to decimals. If no quantity is specified, assume 1.

Unit: The final unit must be one of: "g" (which stands for grams), "ml" (which stands for milliliters), or "unit". Convert all imperial and household measurements (e.g., 'spoons', 'cups', 'cans') to their estimated metric equivalent. Use 'grams' for solids and 'ml' for liquids. For items that are counted (like "1 egg"), use "unit".

Ingredient: Extract only the core, base name of the ingredient. You must remove all descriptors such as brand, origin (e.g., "Russian"), preparation style (e.g., "chopped", "fresh"), or color."""
temperature = 1.0
model_id = "qwen3:4b"

import langextract as lx

examples = [
    lx.data.ExampleData(
        text='[ "4 Tbsp. butter", "3/4 c. chopped onion", "2 Tbsp. curry powder", "3 (14 oz.) cans Italian tomatoes", "3 c. chicken broth", "1 bay leaf", "1/2 c. sour cream"]',
        extractions=[
            lx.data.Extraction(
                extraction_class="ingredient_details",
                extraction_text="4 Tbsp. butter",
                attributes={
                    "quantity": 60,
                    "unit": "ml",
                    "ingredient": "butter",
                },
            ),
            lx.data.Extraction(
                extraction_class="ingredient_details",
                extraction_text="3/4 c. chopped onion",
                attributes={
                    "quantity": 180,
                    "unit": "ml",
                    "ingredient": "onion",
                },
            ),
            lx.data.Extraction(
                extraction_class="ingredient_details",
                extraction_text="2 Tbsp. curry powder",
                attributes={
                    "quantity": 14,
                    "unit": "g",
                    "ingredient": "curry powder",
                },
            ),
            lx.data.Extraction(
                extraction_class="ingredient_details",
                extraction_text="3 (14 oz.) cans Italian tomatoes",
                attributes={
                    # Total quantity: 3 * 14 oz converted to grams
                    "quantity": 1190,
                    "unit": "g",
                    "ingredient": "tomato",
                },
            ),
            lx.data.Extraction(
                extraction_class="ingredient_details",
                extraction_text="3 c. chicken broth",
                attributes={
                    "quantity": 720,
                    "unit": "ml",
                    "ingredient": "chicken broth",
                },
            ),
            lx.data.Extraction(
                extraction_class="ingredient_details",
                extraction_text="1 bay leaf",
                attributes={
                    "quantity": 1,
                    "unit": "unit",
                    "ingredient": "bay leaf",
                },
            ),
            lx.data.Extraction(
                extraction_class="ingredient_details",
                extraction_text="1/2 c. sour cream",
                attributes={
                    "quantity": 120,
                    "unit": "g",
                    "ingredient": "sour cream",
                },
            ),
        ],
    )
]



df_baseline["entity_ingredients"] = df_baseline["ingredients"].progress_apply(lambda x: extract_ingredients_entities(x))

df_baseline["entity_ingredients"].iloc[3]

mask = df_baseline['entity_ingredients'].apply(lambda l: len(l) > 0 and l is not None)
df_filtered = df_baseline[mask]



df_filtered[["NER_ingredients_quantity","NER_ingredients","ingredients"]] = df_filtered["entity_ingredients"].apply(lambda x: get_dict_NER_ingredients_quantity(x))




df_polished = df_filtered[["title","ingredients","directions","NER_ingredients_quantity","NER_ingredients"]]



df_polished.to_csv('./dataset/curated_dataset_12500.csv')


# In[ ]:


# count rows with empty list and delete them.
# Save field "NER_ingredients", "NER_ingredients_quantity".
#Then you add in parquet_creator.py these two columns.
# Then you modify the improve loss function and create a new one for the topological loss. Here you said 

