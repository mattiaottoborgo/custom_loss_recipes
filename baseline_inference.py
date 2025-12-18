import os
import pathlib
import re
from argparse import ArgumentParser
from datetime import datetime
import polars as pl
import torch
#torch._dynamo.disable()
#torch._dynamo.config.suppress_errors = True
#from auto_gptq import exllama_set_max_input_length
from datasets import load_dataset,load_from_disk
from peft import PeftModel,LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from transformers.pipelines.pt_utils import KeyDataset
#from utils import (
#    GSM_FEW_SHOTS,
#    HELLASWAG_FEW_SHOTS,
#    MATH_QA_FEW_SHOTS,
#    OPENBOOKQA_FEW_SHOTS,
#)
from huggingface_hub import login
from typing import List
from pydantic import BaseModel
import outlines
class Ingredient(BaseModel):
    ingredient_name: str
    ingredient_quantity: int
    ingredient_measure_unit:str
class Recipe(BaseModel):
    ingredients: List[str]
    instructions: List[str]

# Your token from https://huggingface.co/settings/tokens
token = "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" 

login(token=token)


def main(args):
    MODELS = {
        "mammoth": "TIGER-Lab/MAmmoTH-7B",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "wizardlm": "TheBloke/wizardLM-7B-HF",
        "llemma": "EleutherAI/llemma_7b",
        "metamath": "meta-math/MetaMath-7B-V1.0",
        "qwen2-1.5" : "Qwen/Qwen2-1.5B-Instruct",
        "qwen3-4" : "Qwen/Qwen3-4B-Instruct-2507",
        "qwen3-14" : "Qwen/Qwen3-14B",
        "qwen3-8": "Qwen/Qwen3-8B",
        "gemma3": "google/gemma-3-4b-it",
    }
    if args.model_finetuned_id:
        print(f"Finetuned model to evaluate: {args.model_finetuned_id}")
    model_name = args.model
    if args.model not in MODELS:
        year = "2023" if "2023" in args.model else "2024"
        model_name = args.model.split("/")[-2].split(year)[0].strip("-")

    MAX_NEW_TOKENS = {
        "closed_qa": 1 if "stablelm" in model_name else 256,
        "xsum": 64,
        "gsm": 256 if "wizard" not in model_name else 512,
        "math_qa": 256 if "wizard" not in model_name else 512,
        "conala": 64,
        "recipe_nlg": 512
    }
    if "llemma" in model_name:
        MAX_NEW_TOKENS["closed_qa"] = 64

    TEMPLATES = {
        "mammoth": lambda x: f"#### Instruction:\n{x}\n\n#### Response: ",
        "mistral": lambda x: f"<s>[INST]{x}[/INST]",
        "stablelm-3b-4e1t": lambda x: f"Question: {x}\nAnswer:",
        "wizardmath": lambda x: f"#### Instruction:\n{x}\n\n#### Response: ",
        "wizardlm": lambda x: f"#### Instruction:\n{x}\n\n#### Response: ",
        "llemma": lambda x: x,
        "metamath": lambda x: f"#### Instruction:\n{x}\n\n#### Response: ",
    }

    def gsm_prompt(example):
        formulas = re.findall(r"(<<[^>]+>>)", example["answer"])
        final_answer = re.findall(r"(#### .+)", example["answer"])
        text = example["question"]
        if "mistral" in model_name:
            text += "\nPut the answer inside angles brackets."
        if "llemma" in model_name:
            text = GSM_FEW_SHOTS + text + "\nAnswer:"
        return {
            "text": TEMPLATES[model_name](text),
            "labels": " ".join(formulas + final_answer),
        }

    def _hellaswag_prompt(example):
        choices_lab = map(lambda x: chr(ord("A") + x), range(len(example["endings"])))
        choices = "\n".join(
            [f"{l}. {t}" for l, t in zip(choices_lab, example["endings"])]
        )
        text = example["ctx"]
        if "llemma" in model_name:
            text = HELLASWAG_FEW_SHOTS + text + "\nAnswer:"
        elif model_name != "stablelm-3b-4e1t":
            text = (
                "Choose the most appropriate answer to complete the following sentence:"
                + text
                + "\nAnswer with a single letter."
            )
        processed = {
            "text": TEMPLATES[model_name](f"{text}\n{choices}"),
            "labels": chr(ord("A") + int(example["label"])),
        }
        return processed

    def _math_qa_prompt(example):
        index = ord(example["correct"]) - ord("a")
        answer = example["options"].split(",")[index].strip()
        answer = re.sub(r"[a-e]\)", "", answer)
        answer = re.sub(r"^[^\d\-\+]*", "", answer)
        answer = re.sub(r"[^\d\-\+]*$", "", answer)
        steps = example["linear_formula"].split("|")[:-1]
        steps = [f"<<{f}>>" for f in steps]
        text = example["Problem"]
        if "llemma" in model_name:
            text = MATH_QA_FEW_SHOTS + text + "\nAnswer:"
        try:
            answer = float(answer)
        except ValueError:
            return {
                "text": "",
                "labels": "",
                "rational": "",
            }
        return {
            "text": TEMPLATES[model_name](text),
            "labels": " ".join(steps + [f"#### {answer}"]),
            "rational": example["Rationale"],
        }

    def _qa_prompt(example):
        choices = "\n".join(
            [
                f"{l}. {t}"
                for l, t in zip(example["choices"]["label"], example["choices"]["text"])
            ]
        )
        text = f"{example['question_stem']}\n{choices}\nContext: {example['fact1']}"
        if "llemma" in model_name:
            text = OPENBOOKQA_FEW_SHOTS + text + "\nAnswer:"
        else:
            text += "\nAnswer with a single letter."
        processed = {
            "text": TEMPLATES[model_name](text),
            "labels": example["answerKey"],
        }
        return processed
    def _recipe_nlg_prompt(prompt):
        messages = [{"role": "user", "content": prompt}]
        return messages
    
    if args.dataset == "hellaswag":
        test_set = load_dataset("Rowan/hellaswag")["validation"].map(_hellaswag_prompt)
    elif args.dataset == "openbookqa":
        test_set = load_dataset("allenai/openbookqa", name="additional")["test"].map(
            _qa_prompt
        )
    elif args.dataset == "gsm8k":
        test_set = load_dataset("gsm8k", name="main")["test"].map(gsm_prompt)
    elif args.dataset == "math_qa":
        test_set = (
            load_dataset("math_qa")["test"]
            .map(_math_qa_prompt)
            .filter(lambda x: len(x["text"]) > 0)
        )
    elif args.dataset == "recipe_nlg":
        test_directory = "./dataset/test"
        test_set = load_from_disk(test_directory)
        print("recipe_nlg dataset loaded")

    current = None
    if pathlib.Path("current.parquet").exists():
        current = pl.read_parquet("current.parquet")
        test_set = test_set.select(range(current.height, len(test_set)))

    tokenizer = AutoTokenizer.from_pretrained(
        MODELS.get(model_name, args.model), padding_side="left", truncation_side="left"
    )
    if args.test_reducted_size:
        print(f"test on reducted size dataset: {args.test_reducted_size} elements")
        test_set = test_set.select(range(args.test_reducted_size))
    model = None
    if "stablelm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stablelm-3b-4e1t",
            trust_remote_code=True,
            cache_dir="/data1/hf_cache/models",
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            args.model,
            trust_remote_code=True,
            cache_dir="/data1/hf_cache/models",
        )
    if "wizardmath" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="/data1/hf_cache/models",
            low_cpu_mem_usage=True,
            device_map="cuda",
        )
        model = exllama_set_max_input_length(model, 4096)
    if "qwen2-1.5" in model_name or "gemma3" in model_name or "qwen3-4" in model_name or "qwen3-8" in model_name or "qwen3-14" in model_name:
        if "gemma3" in model_name:
            print("disable dynamo")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load the model with device_map="auto" for robust device handling
        model = AutoModelForCausalLM.from_pretrained(
            MODELS.get(model_name),
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("loading model qwen2 1.5B, break")
    if args.model_finetuned_id :
        print("using finetuned model")
        adapter_path = f"finetuned_models/{args.model_finetuned_id}"
        model = PeftModel.from_pretrained(model, adapter_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.structured_output:
        print("using outlines for structured output")
        model = outlines.from_transformers(
        model,tokenizer
        )
    else:
        pip = pipeline(
            "text-generation",
            model=MODELS.get(model_name, model) if "gemma3" in model_name else model,
            tokenizer=tokenizer,
            #device="cuda" if "wizardmath" not in model_name else None,
            model_kwargs={
                "low_cpu_mem_usage": True,
                #"torch_dtype": torch.bfloat16,
                "cache_dir": "./data1/hf_cache/models",
            },
        )
    if model_name in ["qwen2-1.5", "gemma3", "qwen3-4","qwen3-8","qwen3-14"] and args.dataset == "recipe_nlg":
        print("starting inference...")
        predictions = []
        ground_truths = []
        titles = []
        batch_size = args.batch_size
        print("apply template")
        # 1. Prepare all prompts correctly using the chat template
        # This turns the list of messages into the exact string the model needs
        #requests_only = [[message for message in conversation["messages"] if message['role'] == 'user' or message['role' == 'system'] for conversation in test_set]
        requests_only = [conversation["messages"][:-1] for conversation in test_set]
        prompts = [
            tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            for request in tqdm(requests_only, desc="Formatting Prompts")
        ]
        print("example of prompt")
        print(prompts[0])
        print("run inference")
        # 2. Run the pipeline on the prepared list of prompt strings
        if args.structured_output:
            print("structured_output")
            for index,prompt in enumerate(tqdm(prompts, desc="Generating recipes through structured output")):
                try:
                    item = test_set[index]
                    print("The prompt is:")
                    print(prompt)
                    prediction = model(prompt,Recipe,max_new_tokens=500)
                    prediction = Recipe.model_validate_json(prediction)
                    prediction = str(prediction.model_dump()).replace("'",'"')
                    print("the prediction is")
                    print(prediction)
                    predictions.append(prediction)
                    ground_truths.append(item["messages"][-1]["content"])
                    titles.append(item.get("title"))
                    
                except Exception as e:
                    print(e)
                
            
        else:
            print("normal output")
            for i in tqdm(range(0, len(prompts), batch_size), desc="Generating in Batches"):

                # 1. Create a batch of prompts from your list
                batch = prompts[i : i + batch_size]

                # 2. Call the pipeline on the entire batch
                # The pipeline will process these in parallel on the GPU
                outs = pip(
                    batch,
                    # The pipeline handles batching internally, but we're giving it a pre-made batch
                    max_new_tokens=MAX_NEW_TOKENS[args.task],
                    pad_token_id=tokenizer.pad_token_id,
                    return_full_text=False,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.95,
                )

                # 3. Process the results from the batch
                # The output 'outs' is a list of lists of dicts
                for out in outs:
                    predictions.append(out[0]["generated_text"].strip())

            # The rest of your code for creating and saving the DataFrame is correct
            ground_truths = [example["messages"][-1]["content"] for example in test_set]
            titles = [example.get("title") for example in test_set]

        
        
        df = (
            pl.DataFrame({"output": predictions, "ground_truth": ground_truths,"title":titles})
            .with_columns(
                pl.col("output").alias("prediction"),
                pl.col("ground_truth"),
            )
            .fill_null("")
        )
        os.makedirs(f"results/sota/{args.dataset}", exist_ok=True)
        if args.model_finetuned_id:
            file_name = f"results/sota/{args.dataset}/{args.model_finetuned_id}"
        else: 
            file_name = f"results/sota/{args.dataset}/{model_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        if pathlib.Path(file_name + ".parquet").exists():
            file_name += "_1"
        df.write_parquet(file_name + ".parquet")
    else:
        pip._preprocess_params |= {
            "truncation": True,
            "max_length": 128 if args.dataset != "hellaswag" else 256,
        }

        pattern = r"### Response: (.+)"

        predictions = []
        ground_truths = test_set["labels"]
        batch_size = args.batch_size
        for out in tqdm(
            pip(
                KeyDataset(test_set, "text"),
                batch_size=batch_size,
                min_length=1,
                max_new_tokens=MAX_NEW_TOKENS[args.task],
                pad_token_id=tokenizer.pad_token_id,
            ),
            total=len(test_set),
        ):
            predictions += [x["generated_text"].strip() for x in out]

        pattern = r"### Response: (.+)"

        df = (
            pl.DataFrame({"output": predictions, "ground_truth": ground_truths})
            .with_columns(
                pl.col("output").alias("prediction"),
                pl.col("ground_truth").str.strip_prefix(": "),
            )
            .fill_null("")
        )
        df = df.with_columns(pl.col("prediction").str.extract(pattern, group_index=1))
        os.makedirs(f"results/sota/{args.dataset}", exist_ok=True)
        file_name = f"results/sota/{args.dataset}/{model_name}"
        if pathlib.Path(file_name + ".parquet").exists():
            file_name += "_1"
        df.write_parquet(file_name + ".parquet")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_finetuned_id", type=str)
    parser.add_argument("--structured_output", type=bool)
    parser.add_argument("--test_reducted_size",type=int)
    args = parser.parse_args()
    main(args)
