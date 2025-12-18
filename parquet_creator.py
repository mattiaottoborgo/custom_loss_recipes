import os
from tqdm import tqdm
import re
from argparse import ArgumentParser
from itertools import chain, product
from typing import Dict
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset, load_dataset,DatasetDict,DatasetInfo
from datetime import datetime
from more_itertools import chunked
from transformers import AutoTokenizer, LlamaTokenizer
from functools import partial
import json
import pandas as pd
import pprint

def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset_source",type=str)
    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=["qa", "gsm", "math_qa", "hellaswag","recipe_nlg"],
    )
    return parser.parse_args()


def _qa_prompt(example):
    choices = "\n".join(
        [
            f"{l}. {t}"
            for l, t in zip(example["choices"]["label"], example["choices"]["text"])
        ]
    )
    processed = {
        "text": f"Question: {example['question_stem']}\n{choices}\nContext: {example['fact1']}\nAnswer: ",
        "labels": example["answerKey"],
    }
    return processed


def _gsm_prompt(example):
    formulas = re.findall(r"(<<[^>]+>>)", example["answer"])
    final_answer = re.findall(r"(#### .+)", example["answer"])
    processed = {
        "text": f"Question: {example['question']}\nAnswer: ",
        "labels": " ".join(formulas + final_answer),
    }
    return processed
def _recipe_nlg_prompt():
    pass
def _math_qa_prompt(example: Dict[str, str]):
    index = ord(example["correct"]) - ord("a")
    answer = example["options"].split(",")[index].strip()
    answer = re.sub(r"[a-e]\)", "", answer)
    answer = re.sub(r"^[^\d\-\+]*", "", answer)
    answer = re.sub(r"[^\d\-\+]*$", "", answer)
    steps = example["linear_formula"].split("|")[:-1]
    steps = [f"<<{f}>>" for f in steps]
    try:
        answer = float(answer)
    except ValueError:
        return {"text": "", "labels": ""}
    return {
        "text": f"Question: {example['Problem']}\nAnswer: ",
        "labels": " ".join(steps + [f"#### {answer}"]),
    }

def _hellaswag_prompt(example):
    choices_lab = map(lambda x: chr(ord("A") + x), range(len(example["endings"])))
    choices = "\n".join([f"{l}. {t}" for l, t in zip(choices_lab, example["endings"])])
    processed = {
        "text": f"Question: {example['ctx']}\n{choices}\nAnswer: ",
        "labels": chr(ord("A") + int(example["label"])),
    }
    return processed

def _tokenize(example, tokenizer, max_length):
    model_inputs = tokenizer(
        example["text"] + example["labels"] + tokenizer.eos_token,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    label_length = len(tokenizer(example["labels"] + tokenizer.eos_token).input_ids)
    model_inputs["labels_position_id"] = [len(model_inputs["input_ids"]) - label_length]
    return model_inputs

####Functions for conversational datasets 
def format_qa_to_messages(example):
    """Formats a question-answer pair into the standard message format."""
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["answer"]},
        ],
        "id": example["task_type"] # Preserve the task type
    }

def apply_formatting(example):
    """Dispatcher function to apply the correct formatting based on task_type."""
    if example['task_type'] == 'recipe':
        # Your existing recipe formatting logic is called here
        return format_to_structured_chat_recipe_nlg(example)
    else:
        # The new QA formatting logic is called for other tasks
        return format_qa_to_messages(example)
def tokenize_conversations(example, tokenizer, max_length=1024):
    prompt_messages = example["messages"][:-1]
    prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    labels_start_position_char = len(prompt_str)

    full_formatted_prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    
    result = tokenizer(
        full_formatted_prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    
    labels_start_position_token = result.char_to_token(labels_start_position_char)
    if labels_start_position_token is None:
        labels_start_position_token = max_length
    result["labels_position_id"] = [labels_start_position_token]
    
    assistant_response_str = example["messages"][-1]['content']
    
    # --- ROBUST INGREDIENT TOKEN FINDING ---
    # Use regex to find the "ingredients": [...] block. This is robust to whitespace and key order.
    # The re.DOTALL flag allows '.' to match newline characters, which is important for pretty-printed JSON.
    match = re.search(r'"ingredients"\s*:\s*(\[.*?\])', assistant_response_str, re.DOTALL)
    
    ingredients_start_token = -1
    ingredients_end_token = -1

    if match:
        # match.start(1) gives the start index of the captured group, which is the list itself '['.
        # match.end(1) gives the end index.
        ingredients_start_char = match.start(1)
        ingredients_end_char = match.end(1)
        
        # Convert character positions to absolute positions in the full formatted prompt
        abs_start_char = labels_start_position_char + ingredients_start_char
        abs_end_char = labels_start_position_char + ingredients_end_char
        
        # Convert absolute character positions to token positions
        ingredients_start_token = result.char_to_token(abs_start_char)
        # For the end token, we look at the character just before the end of the string
        ingredients_end_token = result.char_to_token(abs_end_char - 1)

        # Handle cases where conversion fails (e.g., out of bounds due to truncation)
        if ingredients_start_token is None: ingredients_start_token = -1
        if ingredients_end_token is None: ingredients_end_token = -1
        
    result["ingredients_start_token"] = [ingredients_start_token]
    result["ingredients_end_token"] = [ingredients_end_token]
    
    return result

def format_to_structured_chat_recipe_nlg(example):
    system_prompt = "You are a silent JSON generation API. Your sole purpose is to generate a single, valid JSON object based on the user's request. Do not provide any thoughts, explanations, or conversational text. Your response must start with { and end with }."

    # --- The few-shot example ---
    few_shot_user_prompt = """You are an expert chef. Your task is to generate a recipe for Chocolate Chip Cookies.

Your response must be a single, valid JSON object that follows this exact schema:
- "ingredients": A list of strings, where each string describes the item and its quantity.
- "instructions": A list of strings."""

    few_shot_assistant_response = """{
  "ingredients": [
    "250g of all-purpose flour",
    "5g of baking soda",
    "2g of salt",
    "115g of unsalted butter, softened",
    "100g of granulated sugar",
    "100g of brown sugar",
    "5ml of vanilla extract",
    "1 large egg",
    "200g of chocolate chips"
  ],
  "instructions": [
    "Preheat oven to 190°C (375°F).",
    "In a small bowl, whisk together flour, baking soda, and salt.",
    "In a large bowl, cream together the butter and sugars until light and fluffy.",
    "Beat in the vanilla and egg until well combined.",
    "Gradually add the dry ingredients to the wet ingredients and mix until just combined.",
    "Stir in the chocolate chips.",
    "Drop rounded tablespoons of dough onto ungreased baking sheets.",
    "Bake for 10-12 minutes, or until golden brown.",
    "Let cool on the baking sheets for a few minutes before transferring to a wire rack to cool completely."
  ]
}"""

    # --- The actual prompt for the current data sample ---
    actual_user_prompt = f"""You are an expert chef. Your task is to generate a recipe for {example["title"]}.

Your response must be a single, valid JSON object that follows this exact schema:
- "ingredients": A list of strings, where each string describes the item and its quantity.
- "instructions": A list of strings."""

    # --- The ground truth assistant response for the current data sample ---
    try:
        # json.loads() is the correct tool to parse a string that contains a valid JSON array.
        ingredients_list = json.loads(example['ingredients'])
        instructions_list = json.loads(example['directions'])
    except (json.JSONDecodeError, TypeError):
        # This handles cases where the data might be malformed (e.g., not a valid list string)
        # or is not a string at all (e.g., NaN from pandas).
        print(f"Warning: Could not parse ingredients/directions for title: {example.get('title', 'Unknown')}. Skipping.")
        ingredients_list = []
        instructions_list = []
    assistant_content_dict = {
        "ingredients": ingredients_list,
        "instructions": instructions_list 
    }
    actual_assistant_response = json.dumps(assistant_content_dict, indent=None)

    # --- Assemble the final message list in the correct, alternating format ---
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": few_shot_user_prompt},
            {"role": "assistant", "content": few_shot_assistant_response},
            {"role": "user", "content": actual_user_prompt},
            {"role": "assistant", "content": actual_assistant_response}
        ],
        "id": example['title']
    }


PROMPT_MAP = {
    "qa": _qa_prompt,
    "gsm": _gsm_prompt,
    "math_qa": _math_qa_prompt,
    "hellaswag": _hellaswag_prompt,
    "recipe_nlg": format_to_structured_chat_recipe_nlg,
}

old_format_dataset = ["hellaswag","math_qa","openbookqa","gsm"]
conversational_format_dataset = ["recipe_nlg"]

def main():
    args = _parse_args()

    extra_args = {}
    max_length = 0
    local_dataset = False
    if "openbookqa" in args.dataset:
        max_length = 128
        extra_args["name"] = "additional"
    elif "gsm" in args.dataset:
        max_length = 128
        extra_args["name"] = "main"
    elif "math_qa" in args.dataset:
        max_length = 1024
    elif "hellaswag" in args.dataset:
        max_length = 256
    elif "recipe_nlg" in args.dataset:
        max_length = 1024 #TODO: check if it makes sense
        local_dataset = True
    if local_dataset:
        file_path = os.path.join("dataset", "cooking_questions.csv")
        on_bad_lines_setting = 'skip'
        total_rows = sum(1 for row in open(file_path, 'r'))

        # --- Step 2: Read the CSV in chunks with a progress bar ---
        chunk_size = 10000  # Adjust chunk size based on your memory
        chunks = []

        # Use tqdm to create a progress bar
        with tqdm(total=total_rows, desc="Loading CSV") as pbar:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, on_bad_lines=on_bad_lines_setting):
                chunks.append(chunk)
                pbar.update(len(chunk))
        df_recipe_questions = pd.concat(chunks, axis=0)
        print("dataset_source",args.dataset_source)
        file_path = os.path.join("dataset", f"{args.dataset_source}.csv")
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
        df_subset = df_recipe_nlg.head(1250).copy()
        df_subset["task_type"] = "recipe"

        # --- Step 4: First split (80% train, 20% temp) ---
        # The temp set will hold our future validation and test data
        train_df, temp_df = train_test_split(
            df_subset,
            #df_recipe_nlg,
            test_size=0.2,       # 20% of data goes to the temporary set
            random_state=42      # Use a fixed random state for reproducibility
        )

        # --- Step 5: Second split (50% validation, 50% test from the temp set) ---
        # This splits the 20% temp set in half, resulting in 10% validation and 10% test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,       # 50% of the temp data goes to the test set
            random_state=42      # Use the same random state for consistency
        )
        ##add questions dataset
        print("adding question dataset!")
        train_df_questions, temp_df_questions = train_test_split(
            df_recipe_questions,
            #df_recipe_nlg,
            test_size=0.2,       # 20% of data goes to the temporary set
            random_state=42      # Use a fixed random state for reproducibility
        )

        # --- Step 5: Second split (50% validation, 50% test from the temp set) ---
        # This splits the 20% temp set in half, resulting in 10% validation and 10% test
        val_df_questions, test_df_questions = train_test_split(
            temp_df_questions,
            test_size=0.5,       # 50% of the temp data goes to the test set
            random_state=42      # Use the same random state for consistency
        )
        #train_df = pd.concat([train_df,train_df_questions],axis=0)
        #val_df = pd.concat([val_df,val_df_questions],axis=0)
        #test_df = pd.concat([test_df,test_df_questions],axis=0)
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        ##add ingredient metadata to the dataset. It consists of all ingredients in the dataset
        #train_dataset.info["metadata"]['all_ingredients'] = ingredient_list

        # --- Step 6: Combine the individual datasets into a DatasetDict ---
        recipe_dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        for k,v in recipe_dataset_dict.items():
            print("column_names")
            print(v.column_names)
            print(f"key: {k}")
            formatted_dataset = v.map(
            apply_formatting, 
            #remove_columns=v.column_names
            )
            recipe_dataset_dict[k] = formatted_dataset
        # --- Step 7. Define paths and check for existence ---
        output_dir = "dataset"
        train_path = os.path.join(output_dir, "train")
        val_path = os.path.join(output_dir, "validation")
        test_path = os.path.join(output_dir, "test")

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Check if the files already exist
        if not os.path.exists(train_path):
            print("Train dataset not found. Creating and saving...")
            recipe_dataset_dict["train"].save_to_disk(train_path)
            print(f"train dataset saved in {train_path}")
        if not os.path.exists(val_path):
            print("Validation dataset not found. Creating and saving...")
            recipe_dataset_dict["validation"].save_to_disk(val_path)
            print(f"validation dataset saved in {val_path}")
        if not os.path.exists(test_path):
            print("Test dataset not found. Creating and saving...")
            recipe_dataset_dict["test"].save_to_disk(test_path)
            print(f"test dataset saved in {test_path}")
    else:   
        datasets = load_dataset(args.dataset, cache_dir="cache", **extra_args)
        if "hellaswag" in args.dataset:
            datasets.pop("test")
        if "test" not in datasets:
            datasets["test"] = datasets["validation"]
        if "validation" not in datasets:
            train_val = datasets["train"].train_test_split(test_size=0.1, seed=42)
            datasets["train"] = train_val["train"]
            datasets["validation"] = train_val["test"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        #use_fast=True,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.dataset in old_format_dataset:
        for k, v in datasets.items():
            v = (
                v.map(PROMPT_MAP[args.prompt_type])
                .filter(lambda x: len(x["text"]) > 0)
                .map(
                    lambda x: _tokenize(x, tokenizer, max_length),
                    remove_columns=v.column_names,
                )
                .remove_columns(["text", "labels"])
            )
            model_name = args.tokenizer.split("/")[-1]
            dataset_name = args.dataset.split("/")[-1]
            os.makedirs(f"data/{dataset_name}/{model_name}", exist_ok=True)
            v.to_parquet(f"data/{dataset_name}/{model_name}/{k}.parquet")
    elif args.dataset in conversational_format_dataset:
        tokenization_func = partial(tokenize_conversations, tokenizer=tokenizer, max_length=450)
        for k,v in recipe_dataset_dict.items():
            print("column_names 2")
            pprint.pp(v[0])
            tokenized_dataset = v.map(
                tokenization_func,
                batched=False, # Process one by one, can be set to True for larger datasets
                #remove_columns=["title","ingredients","directions","Unnamed: 0","id","messages","__index_level_0__","prompt","Unnamed: 0.1"] # Remove old columns
            #)
            remove_columns=["title","ingredients","directions","Unnamed: 0","id","messages","__index_level_0__"] # Remove old columns
            )
            model_name = args.tokenizer.split("/")[-1]
            dataset_name = args.dataset.split("/")[-1]
            os.makedirs(f"data/{dataset_name}/{model_name}", exist_ok=True)
            pprint.pp(tokenized_dataset[0])
            #tokenized_dataset.info.metadata = {"ingredient_list": ingredient_list}
            #tokenized_dataset.to_parquet(f"data/{dataset_name}/{model_name}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/{k}.parquet")
            tokenized_dataset.to_parquet(f"data/{dataset_name}/{model_name}/{args.dataset_source}/{k}.parquet")


if __name__ == "__main__":
    main()
