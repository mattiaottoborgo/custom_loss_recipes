#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##import libraries
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


# In[3]:
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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


# In[5]:


##NER directions
prompt_directions = """
You are an expert culinary data scientist. Your task is to analyze a list of cooking recipe steps and extract key information for each step into a structured JSON format.

Instructions:

Process the input list of recipe steps in the order they are provided.

For each step, extract the following four entities:

action: The single, primary cooking verb for the step (e.g., "Combine", "Bake", "Stir").

temperature_celsius: The temperature mentioned, converted to Celsius. If not present, use null. Must be a number.

time_minutes: The duration in minutes. If a range is given (e.g., "10-12 minutes"), use the average. If not present, use null. Must be number

ingredients: A list of the food ingredients involved in that specific step. If no ingredients are mentioned, use an empty list []."""
examples_directions = [
    lx.data.ExampleData(
        # The input is the string representation of the list of steps
        text="['Cut the oranges into slices and place them in a large bowl.', 'In a small saucepan, combine the sugar, lime juice, and cornstarch. Bring to a boil and cook until thickened, about 2-3 minutes.', 'Pour the dressing over the fruit and toss gently to coat.', 'Add the strawberries and let sit for at least 10 minutes before serving.']",
        extractions=[
            lx.data.Extraction(
                extraction_class="step_details",
                extraction_text="Cut the oranges into slices and place them in a large bowl.",
                attributes={
                    "action": "Cut",
                    "temperature_celsius": None,
                    "time_minutes": None,
                    "ingredients": ["oranges"]
                },
            ),
            lx.data.Extraction(
                extraction_class="step_details",
                extraction_text="In a small saucepan, combine the sugar, lime juice, and cornstarch. Bring to a boil and cook until thickened, about 2-3 minutes.",
                attributes={
                    "action": "Combine",
                    "temperature_celsius": None,
                    "time_minutes": 2.5,
                    "ingredients": ["sugar", "lime juice", "cornstarch"]
                },
            ),
            lx.data.Extraction(
                extraction_class="step_details",
                extraction_text="Pour the dressing over the fruit and toss gently to coat.",
                attributes={
                    "action": "Pour",
                    "temperature_celsius": None,
                    "time_minutes": None,
                    "ingredients": ["dressing", "fruit"]
                },
            ),
            lx.data.Extraction(
                extraction_class="step_details",
                extraction_text="Add the strawberries and let sit for at least 10 minutes before serving.",
                attributes={
                    "action": "Add",
                    "temperature_celsius": None,
                    "time_minutes": 10,
                    "ingredients": ["strawberries"]
                },
            ),
        ],
    )
]


# In[6]:


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
def extract_directions(json_data):
    """
    Safely extracts the 'directions' field from a dictionary.
    
    Returns the list of directions if the key is found, 
    otherwise returns an empty string.
    """
    try:
        # Try to access the 'ingredients' key
        return json_data["instructions"]
    except Exception as e:
        return []
def extract_ingredients_entities(json):
    """
    Extracts the 'ingredients' entities using LangExtract tool.
    """
    try:
        ingredients = str(json["ingredients"])
        print("ingr",ingredients)
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
def extract_directions_entities(json):
    """
    Extracts the 'ingredients' entities using LangExtract tool.
    """
    try:
        directions = str(json["instructions"])
        result = lx.extract(
               text_or_documents=directions,
               prompt_description=prompt_directions,
               examples=examples_directions,
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
        # If the key doesn't exist or the input isn't a valid dict/object,
        # return an empty string instead of raising an error.
        return []
def calculate_ingredients_scores(row):
    """
    Calculates ingredient recall and quantity precision for a single row of a DataFrame.
    Quantity precision is calculated on normalized values.
    """
    try:
        predictions = row["entity_ingredients_prediction"]
        ground_truths = row["entity_ingredients_ground_truth"]

        # --- Data Preparation ---
        pred_ingredients_set = {item['ingredient'] for item in predictions if item and item.get('ingredient')}
        gt_ingredients_set = {item['ingredient'] for item in ground_truths if item and item.get('ingredient')}
        
        # This map uses the ORIGINAL quantities for lookup before normalization
        gt_quantity_map = {item['ingredient']: item['quantity'] for item in ground_truths if item and item.get('ingredient') and item.get('quantity') is not None}

        # --- Calculate recall_ingredients ---
        if not gt_ingredients_set:
            recall_score = 1.0 if not pred_ingredients_set else 0.0
        else:
            recalled_ingredients = pred_ingredients_set.intersection(gt_ingredients_set)
            recall_score = len(recalled_ingredients) / len(gt_ingredients_set)

        # --- Calculate precision_quantity with Normalization ---
        if not predictions:
            quantity_score = 0.0
        else:
            # 1. Find the maximum quantities in each list for normalization
            gt_quantities = [item['quantity'] for item in ground_truths if item and item.get('quantity') is not None]
            pred_quantities = [item['quantity'] for item in predictions if item and item.get('quantity') is not None]

            max_gt_quantity = max(gt_quantities) if gt_quantities else 1.0
            max_pred_quantity = max(pred_quantities) if pred_quantities else 1.0
            
            # Avoid division by zero if a max quantity happens to be 0
            if max_gt_quantity == 0: max_gt_quantity = 1.0
            if max_pred_quantity == 0: max_pred_quantity = 1.0

            total_quantity_score = 0
            for pred_item in predictions:
                if pred_item is None or not pred_item.get('ingredient') or pred_item.get('quantity') is None:
                    continue
                
                pred_ing = pred_item['ingredient']
                pred_qty = pred_item['quantity']
                
                if pred_ing in gt_quantity_map:
                    gt_qty = gt_quantity_map[pred_ing]
                    
                    # 2. Normalize both quantities before comparison
                    normalized_pred_qty = pred_qty / max_pred_quantity
                    normalized_gt_qty = gt_qty / max_gt_quantity

                    # 3. Calculate percentage error on the NORMALIZED values
                    if normalized_gt_qty == 0 and normalized_pred_qty == 0:
                        percentage_error = 0
                    elif normalized_gt_qty == 0:
                        percentage_error = float('inf')
                    else:
                        percentage_error = abs(normalized_pred_qty - normalized_gt_qty) / normalized_gt_qty
                    
                    # 4. Assign score based on the error (this logic is unchanged)
                    if percentage_error <= 0.10:
                        total_quantity_score += 1
                    elif percentage_error <= 0.20:
                        total_quantity_score += 0.5
                    elif percentage_error <= 0.50:
                        total_quantity_score += 0
                    elif percentage_error <= 1.00:
                        total_quantity_score -= 0.5
                    else:
                        total_quantity_score -= 1
            
            quantity_score = total_quantity_score / len(predictions)
            if quantity_score < 0:
                quantity_score = 0

        return {"recall_ingredients": recall_score, "precision_quantity": quantity_score}

    except Exception:
        traceback.print_exc()
        return {"recall_ingredients": -1.0, "precision_quantity": -1.0}



# In[8]:


embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B',device=device)

def calculate_order_error(similarity_matrix, num_gt_actions):
    """
    Calculates a normalized score based on Spearman's Footrule Distance using a
    pre-computed cosine similarity matrix. The final score is the average
    displacement error per ground truth action.
    """
    try:
        # Handle cases with no actions to compare
        if num_gt_actions == 0:
            return 0
        if similarity_matrix.nelement() == 0:
            return 0
            
        num_preds, num_gts = similarity_matrix.shape
        total_distance = 0
        used_gt_indices = set()

        for i in range(num_preds):
            best_gt_index = -1
            max_sim = -2.0

            for j in range(num_gts):
                if j not in used_gt_indices:
                    sim = similarity_matrix[i][j].item()
                    if sim > max_sim:
                        max_sim = sim
                        best_gt_index = j
            
            if best_gt_index != -1:
                total_distance += abs(i - best_gt_index)
                used_gt_indices.add(best_gt_index)
        
        # Normalize the total distance by the number of ground truth actions
        return total_distance / num_gt_actions
        
    except Exception:
        traceback.print_exc()
        return -1

def calculate_direction_scores(row):
    """
    Calculates and returns a dictionary of scores for recipe directions
    based on a single DataFrame row, using pre-computed embeddings.
    """
    scores = {"action_precision": -1, "temperature_precision": -1, "time_precision": -1, "order_error": -1}

    try:
        predictions = row["entity_directions_prediction"]
        ground_truths = row["entity_directions_ground_truth"]
        
        #filter out predictions and ground_truths that are empty
        predictions = [prediction for prediction in predictions if prediction]
        ground_truths = [ground_truth for ground_truth in ground_truths if ground_truth]
        
        # --- Handle empty cases ---
        if not ground_truths:
            return {"action_precision": -1, "temperature_precision": -1, "time_precision": -1, "order_error": -1}
        if not predictions:
            return {"action_precision": -1, "temperature_precision": -1, "time_precision": -1, "order_error": -1}

        # --- Step 1: Pre-compute all embeddings and the similarity matrix ---
        pred_actions = [item.get('action') for item in predictions if item and item.get('action')]
        gt_actions = [item.get('action') for item in ground_truths if item and item.get('action')]

        if not gt_actions or not pred_actions:
            # If there are no actions to compare, score accordingly
            scores["action_precision"] = -1
            scores["order_precision"] = -1 # Or handle as per your logic
            # Temp/Time would need separate handling if actions are missing
            return scores
        
        pred_embeddings = embedding_model.encode(pred_actions, convert_to_tensor=True)
        gt_embeddings = embedding_model.encode(gt_actions, convert_to_tensor=True)
        similarity_matrix = util.pytorch_cos_sim(pred_embeddings, gt_embeddings)

        # --- Step 2: Greedily match predictions to ground truths using the matrix ---
        total_action_points = 0
        total_temp_points = 0
        relevant_gt_temps = sum(1 for gt in ground_truths if gt and gt.get('temperature_celsius',None) is not None)
        total_time_points = 0
        relevant_gt_times = sum(1 for gt in ground_truths if gt and gt.get('time_minutes',None) is not None)
        
        used_gt_indices = set()

        for i in range(len(pred_actions)):
            best_gt_index = -1
            max_sim = -2.0

            for j in range(len(gt_actions)):
                if j not in used_gt_indices:
                    sim = similarity_matrix[i][j].item()
                    if sim > max_sim:
                        max_sim = sim
                        best_gt_index = j
            
            if best_gt_index != -1:
                # Use the best match to score all attributes
                pred_item = predictions[i]
                best_gt_item = ground_truths[best_gt_index]

                if max_sim > 0.7:
                    total_action_points += 1

                pred_temp = pred_item.get('temperature_celsius',None)
                gt_temp = best_gt_item.get('temperature_celsius',None)
                if gt_temp is not None and pred_temp is not None and gt_temp > 0:
                    if abs(pred_temp - gt_temp) / gt_temp <= 0.15:
                        total_temp_points += 1
                
                pred_time = pred_item.get('time_minutes',None)
                gt_time = best_gt_item.get('time_minutes',None)
                if gt_time is not None and pred_time is not None and gt_time > 0:
                    if abs(pred_time - gt_time) / gt_time <= 0.15:
                        total_time_points += 1

                used_gt_indices.add(best_gt_index)

        # --- Step 3: Final Score Calculation ---
        scores["action_precision"] = total_action_points / len(ground_truths)
        scores["temperature_precision"] = total_temp_points / relevant_gt_temps if relevant_gt_temps > 0 else 1.0
        scores["time_precision"] = total_time_points / relevant_gt_times if relevant_gt_times > 0 else 1.0
        scores["order_error"] = calculate_order_error(similarity_matrix,len(gt_actions))

        return scores

    except Exception:
        traceback.print_exc()
        return scores

def calculate_all_scores(row):
    """
    Calculates both the action-only and the flattened action-ingredient
    edit distances for a single DataFrame row.
    """
    try:
        ground_truth = row['entity_directions_ground_truth']
        prediction = row['entity_directions_prediction']

        # Handle cases where the cell might be empty or not a list
        if not isinstance(ground_truth, (list, np.ndarray)): ground_truth = []
        if not isinstance(prediction, (list, np.ndarray)): prediction = []

        # --- 1. Calculate Action-Only Edit Distance ---
        gt_actions = [s['action'] for s in ground_truth if isinstance(s, dict) and 'action' in s and s.get('action') is not None]
        pred_actions = [s['action'] for s in prediction if isinstance(s, dict) and 'action' in s and s.get('action') is not None]
        action_distance = editdistance.eval(gt_actions, pred_actions)

        # --- 2. Calculate Flattened Action-Ingredient Edit Distance ---
        def flatten_recipe(step_list):
            flat_list = []
            for step in step_list:
                if isinstance(step, dict):
                    if step.get('action'):
                        flat_list.append(step['action'])
                    if isinstance(step.get('ingredients'), (list, np.ndarray)):
                        flat_list.extend(step['ingredients'])
            return flat_list
        
        gt_flat = flatten_recipe(ground_truth)
        pred_flat = flatten_recipe(prediction)
        flattened_distance = editdistance.eval(gt_flat, pred_flat)

        return pd.Series([action_distance, flattened_distance])

    except Exception as e:
        print(e)
        # Return error codes if the row is malformed
        return pd.Series([-1, -1])
    
    
    
# In[9]:

def main(args):
    ##Evaluation Configs
    ##model_name: name of the model to evaluate
    model_name = "Gemini-2.0-Flash"
    task = "recipe_nlg"
    baseline_name = "gemini-2.0-flash"
    ##experiment_name : experiment to evaluate against Sota


    ##load baseline and ground truth
    file_path_baseline = f"./results/sota/{task}/{baseline_name}.parquet"
    # Read the Parquet file into a pandas DataFrame
    df_baseline = pd.read_parquet(file_path_baseline)
    
    if args.test_reducted_size:
        print(f"reducted size evaluation to {args.test_reducted_size}")
        df_baseline = df_baseline.head(args.test_reducted_size)

    #

    # Display the first 10 lines of the DataFrame
    print(df_baseline.columns)


    # In[12]:


    df_baseline


    # In[14]:


    ##calculate BLEU,ROUGE, Semantic similarity
    metrics = [
        'bleu',
        'rouge',
        'bertscore',
    ]


    # In[15]:


    ##evaluation of baseline
    results = []
    for metric in metrics:
        print(f"evaluating {metric}")
        evaluator = evaluate.load(metric)
        for index, row in tqdm(df_baseline.iterrows(),total=df_baseline.shape[0],desc="Processing Rows"):
            # Get the output model and ground truth
            output_model = row["output"]
            ground_truth = row["ground_truth"]
            title = row["title"]
            # Compute the score
            if metric == "bertscore":
                # For BERTScore, we need to pass the model name
                score = evaluator.compute(predictions=[output_model], references=[ground_truth],lang="en")
            else:
                try:
                    score = evaluator.compute(predictions=[output_model], references=[ground_truth])
                except Exception as e:
                    score = {'precision': [0.0], 'recall': [0.0], 'f1': [0.0]}
            results.append({
                'experiment_name' : f"{baseline_name}",
                'title': title,
                'metric': metric,
                'score': score
            })


    # In[163]:


    results_df = pd.DataFrame(results)


    # In[164]:


    file_path_metrics = f"eval/intermediate/{baseline_name}_basic_metrics.parquet"


    # In[165]:


    results_df.to_parquet(file_path_metrics,engine='pyarrow', index=False)


    # In[17]:


    df = results_df.copy()


    # In[18]:


    #extract score and merge to df_baseline
    #crea all_metrics list by taking distinct values from the metric column
    df_total= pd.DataFrame()
    for metric in metrics:
        df_metric = df[df['metric'] == metric]
        df_metric.reset_index(drop=True, inplace=True)
        if metric == "bleu":
            print(f"Processing metric: {metric}")
            df_metric["bleu"] =  df_metric['score'].apply(lambda x: x[metric] if isinstance(x, dict) else 0)
            #calculate the mean and standard devation of the bleu score for each experiment_name
            #df_metric_mean = df_metric.groupby( 'experiment_name')["bleu"].mean().reset_index()
            df_metric_mean = df_metric.groupby('experiment_name')["bleu"].agg(['mean', 'std']).reset_index()
            df_metric_mean["metric"] = metric.upper()
            df_total = pd.concat([df_total, df_metric_mean], ignore_index=True)
        #concatenate the df_metric_mean to df_total


        elif metric == "rouge":
            print(f"Processing metric: {metric}")
            df_metric["rouge1"] =  df_metric['score'].apply(lambda x: x["rouge1"] if isinstance(x, dict) else 0)
            #calculate the mean and standard devation of the rouge score for each experiment_name
            df_metric_mean = df_metric.groupby('experiment_name')["rouge1"].agg(['mean', 'std']).reset_index()
            df_metric_mean["metric"] = "rouge1".upper()
            df_total = pd.concat([df_total, df_metric_mean], ignore_index=True)

            df_metric["rouge2"] =  df_metric['score'].apply(lambda x: x["rouge2"] if isinstance(x, dict) else 0)
            #calculate the mean and standard devation of the rouge score for each experiment_name
            df_metric_mean = df_metric.groupby('experiment_name')["rouge2"].agg(['mean', 'std']).reset_index()
            df_metric_mean["metric"] = "rouge2".upper()
            df_total = pd.concat([df_total, df_metric_mean], ignore_index=True)

            df_metric["rougeL"] =  df_metric['score'].apply(lambda x: x["rougeL"] if isinstance(x, dict) else 0)
            #calculate the mean and standard devation of the rouge score for each experiment_name
            df_metric_mean = df_metric.groupby('experiment_name')["rougeL"].agg(['mean', 'std']).reset_index()
            df_metric_mean["metric"] = "rougeL".upper()
            #concatenate the df_metric_mean to df_total
            df_total = pd.concat([df_total, df_metric_mean], ignore_index=True)
        elif metric == "bertscore":
            print(f"Processing metric: {metric}")
            df_metric["bertscore_f1"] =  df_metric['score'].apply(lambda x: x["f1"][0] if isinstance(x, dict) else 0)
            #calculate the mean and standard devation of the bertscore for each experiment_name
            df_metric_mean = df_metric.groupby('experiment_name')["bertscore_f1"].agg(['mean', 'std']).reset_index()
            df_metric_mean["metric"] = "bertscore_f1".upper()
            #concatenate the df_metric_mean to df_total
            df_total = pd.concat([df_total, df_metric_mean], ignore_index=True)
            df_metric["bertscore_precision"] =  df_metric['score'].apply(lambda x: x["precision"][0] if isinstance(x, dict) else 0)
            #calculate the mean and standard devation of the bertscore for each experiment_name
            df_metric_mean = df_metric.groupby('experiment_name')["bertscore_precision"].agg(['mean', 'std']).reset_index()
            df_metric_mean["metric"] = "bertscore_precision".upper()
            #concatenate the df_metric_mean to df_total
            df_total = pd.concat([df_total, df_metric_mean], ignore_index=True)
            df_metric["bertscore_recall"] =  df_metric['score'].apply(lambda x: x["recall"][0] if isinstance(x, dict) else 0)
            #calculate the mean and standard devation of the bertscore for each experiment_name
            df_metric_mean = df_metric.groupby('experiment_name')["bertscore_recall"].agg(['mean', 'std']).reset_index()
            df_metric_mean["metric"] = "bertscore_recall".upper()
            #concatenate the df_metric_mean to df_total
            df_total = pd.concat([df_total, df_metric_mean], ignore_index=True)    
    df_total


    # In[20]:


    ##calculate graph
    df_baseline["json_prediction"] = df_baseline["prediction"].apply(safe_json_loads)
    df_baseline["json_ground_truth"] = df_baseline["ground_truth"].apply(safe_json_loads)


    # In[21]:


    df_baseline["ingredients_ground_truth"] = df_baseline["json_ground_truth"].apply(extract_ingredients)
    df_baseline["ingredients_prediction"] = df_baseline["json_prediction"].apply(extract_ingredients)


    # In[22]:


    df_baseline["directions_ground_truth"] = df_baseline["json_ground_truth"].apply(extract_directions)
    df_baseline["directions_prediction"] = df_baseline["json_prediction"].apply(extract_directions)


    # In[23]:




    # In[24]:


    df_baseline_reduced = df_baseline.copy()
    #df_baseline_reduced = df_baseline.head(1).copy()


    # In[25]:


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
    tqdm.pandas(desc="Extracting...")


    # In[26]:


    df_baseline_reduced["entity_ingredients_ground_truth"] = df_baseline_reduced["json_ground_truth"].progress_apply(lambda x:extract_ingredients_entities(x))


    # In[27]:


    df_baseline_reduced["entity_ingredients_prediction"] = df_baseline_reduced["json_prediction"].progress_apply(lambda x:extract_ingredients_entities(x))


    # In[28]:


    pd.set_option('display.max_columns', None)


    # In[29]:


    df_baseline_reduced["ingredients_scores"] = df_baseline_reduced.apply(calculate_ingredients_scores,axis=1)


    # In[30]:


    def ensure_list(value):
        """
        Ensures the value is a list. If it's not, it returns an empty list.
        """
        if isinstance(value, list):
            return value
        else:
            # If the value is a string, None, or anything else, return an empty list
            return []


    # In[31]:


    df_baseline_reduced["entity_ingredients_prediction"] = df_baseline_reduced["entity_ingredients_prediction"].apply(ensure_list)
    df_baseline_reduced["entity_ingredients_ground_truth"] = df_baseline_reduced["entity_ingredients_ground_truth"].apply(ensure_list)


    try:
        file_path_intermediate_1 = f"eval/intermediate/{baseline_name}_ingredients_scores.parquet"
        print("parquet saved")
        df_baseline_reduced.to_parquet(file_path_intermediate_1, engine='pyarrow', index=False)
        df_baseline_reduced = pd.read_parquet(file_path_intermediate_1)
    except Exception as e:
        print("csv saved")
        output_path = f"eval/intermediate/{baseline_name}_ingredients_scores.jsonl"

        with open(output_path, "w") as f:
            f.write(df_baseline_reduced.to_json(orient='records', lines=True, force_ascii=False))

    ## calculate scores on directions
    df_baseline_reduced["entity_directions_ground_truth"] = df_baseline_reduced["json_ground_truth"].progress_apply(lambda x:extract_directions_entities(x))
    df_baseline_reduced["entity_directions_prediction"] = df_baseline_reduced["json_prediction"].progress_apply(lambda x:extract_directions_entities(x))


    # In[38]:


    df_baseline_reduced["entity_directions_prediction"] = df_baseline_reduced["entity_directions_prediction"].apply(ensure_list)
    df_baseline_reduced["entity_directions_ground_truth"] = df_baseline_reduced["entity_directions_ground_truth"].apply(ensure_list)



    # In[40]:


    df_baseline_reduced.iloc[0]["entity_directions_prediction"]


    # In[41]:


    #TODO: compute directions score. Save dataframe evaluation as parquet to make sure you can reuse it.


    # In[42]:


    df_baseline_reduced["directions_scores"] = df_baseline_reduced.apply(calculate_direction_scores,axis=1)


    # In[101]:


    df_baseline_reduced["ingredients_scores"].iloc[0]


    # In[118]:

    try:
        file_path_intermediate_2 = f"eval/intermediate/{baseline_name}_directions_scores.parquet"
        df_baseline_reduced.to_parquet(file_path_intermediate_2, engine='pyarrow', index=False)
    except Exception as e:
        output_path = f"eval/intermediate/{baseline_name}_directions_scores.jsonl"

        with open(output_path, "w") as f:
            f.write(df_baseline_reduced.to_json(orient='records', lines=True, force_ascii=False))




    df_metrics_recipe = pd.DataFrame()


    # In[150]:


    df_baseline_reduced["action_precision"] = df_baseline_reduced["directions_scores"].apply(lambda x : x["action_precision"] if isinstance(x,dict) else 0)
    df_baseline_reduced["temperature_precision"] = df_baseline_reduced["directions_scores"].apply(lambda x : x["temperature_precision"] if isinstance(x,dict) else 0)
    df_baseline_reduced["time_precision"] = df_baseline_reduced["directions_scores"].apply(lambda x : x["time_precision"] if isinstance(x,dict) else 0)
    df_baseline_reduced["order_error"] = df_baseline_reduced["directions_scores"].apply(lambda x : x["order_error"] if isinstance(x,dict) else 0)


    # In[151]:


    df_baseline_reduced["recall_ingredients"] = df_baseline_reduced["ingredients_scores"].apply(lambda x : x["recall_ingredients"] if isinstance(x,dict) else 0)
    df_baseline_reduced["precision_quantity"] = df_baseline_reduced["ingredients_scores"].apply(lambda x : x["precision_quantity"] if isinstance(x,dict) else 0)


    ##calculate edit distance on pure actions and actions + ingredients
    df_baseline_reduced[['action_edit_distance', 'step_edit_distance']] = df_baseline_reduced.apply(calculate_all_scores, axis=1)

    try:
        file_path_intermediate_3 = f"eval/intermediate/{baseline_name}_all_scores.parquet"
        df_baseline_reduced.to_parquet(file_path_intermediate_3, engine='pyarrow', index=False)
    except Exception as e:
        output_path = f"eval/intermediate/{baseline_name}_all_scores.jsonl"

        with open(output_path, "w") as f:
            f.write(df_baseline_reduced.to_json(orient='records', lines=True, force_ascii=False))

    ## metrics aggregation


    direction_metrics = ["action_precision","temperature_precision","time_precision","order_error","precision_quantity","recall_ingredients","action_edit_distance","step_edit_distance"]
    #direction_metrics = ["precision_quantity","recall_ingredients"]


    for metric in direction_metrics:
        #filter away all rows having any metric equal to -1.
        df_baseline_reduced = df_baseline_reduced[df_baseline_reduced[metric] != -1]
        #calculate mean of action_precision,temperature_precision,time_precision,order_error
        df_metric = df_baseline_reduced[metric].agg(['mean', 'std']).reset_index()
        df_metric["metric"] = metric
        df_metric["experiment_name"] = baseline_name
        df_wide = df_metric.pivot(
        index=['experiment_name', 'metric'], 
            columns='index', 
            values=metric
        ).reset_index()
        # The columns might have a name from the pivot operation, you can remove it
        df_wide.columns.name = None
        df_metrics_recipe = pd.concat([df_metrics_recipe,df_wide],ignore_index=True)



    df_metrics = pd.concat([df_total,df_metrics_recipe],ignore_index=True)


    file_path_metrics = f"eval/{baseline_name}_metrics.parquet"


    # In[160]:

    try:
        df_metrics.to_parquet(file_path_metrics,engine='pyarrow', index=False)
    except Exception as e:
        output_path = f"eval/intermediate/{baseline_name}_metrics.jsonl"
        with open(output_path, "w") as f:
            f.write(df_metrics.to_json(orient='records', lines=True, force_ascii=False))
        
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_reducted_size",type=int)
    args = parser.parse_args()
    main(args)


