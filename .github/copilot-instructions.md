# Custom Loss Functions Research Project

## Project Overview
Research project for fine-tuning LLMs with custom loss functions (Dice, Focal, Topological, Lovasz) optimized for recipe generation and ingredient prediction. Uses Hydra for configuration management, PEFT/LoRA for efficient training, and supports multi-task learning (text generation + regression).

## Architecture

### Core Training Pipeline
- **Entry point**: [improved_loss.py](improved_loss.py) via `@hydra.main(config_path="configs", config_name="loss")`
- **Custom Trainers**: [model/loss_model.py](model/loss_model.py) (`ImprovedLossTrainer`) and [model/multitask_loss_model.py](model/multitask_loss_model.py) (`MultiTaskImprovedLossTrainer`) extend HuggingFace `Trainer`
- **Multi-task model**: [model/multitask_model.py](model/multitask_model.py) wraps base LLM with regression heads for ingredient quantities, presence, temperature, time

### Loss Functions ([loss/](loss/))
Custom losses instantiated once in Trainer `__init__` (not per-step):
- **GDiceLoss**: Segmentation-inspired loss using einsum operations
- **FocalLoss**: For class imbalance (γ=2 default)
- **TopologicalLoss**: Compares embedding point clouds via Sinkhorn divergence (geomloss)
- **MultiLabelDiceLoss**: For ingredient presence prediction

### Configuration System (Hydra)
Three-level composition in [configs/](configs/):
1. **Root config** (e.g., `loss_topological-recipenlg_Qwen3-4B.yaml`): Sets `experiment_name`, `task`, `multitask` flag
2. **Model** ([configs/model/](configs/model/)): Just `model_name` (e.g., `Qwen/Qwen3-4B-Instruct-2507`)
3. **Dataset** ([configs/dataset/](configs/dataset/)): Parquet file paths for train/val/test splits per task
4. **Trainer** ([configs/trainer/](configs/trainer/)): Training hyperparameters + **critical** `losses` dict mapping tasks to loss weights:
   ```yaml
   losses:
     question_answering:
       cross_entropy: 0.6
       topological: 0.4
   ```

## Key Workflows

### 1. Dataset Preparation
Create tokenized parquet files from raw datasets:
```bash
# Recipe NLG datasets
python parquet_creator.py \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --dataset recipe_nlg \
  --prompt_type recipe_nlg

# Math QA datasets
python parquet_creator.py \
  --tokenizer mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset math_qa \
  --prompt_type math_qa
```
**What it does**: Tokenizes datasets and adds custom fields required for training:
- `labels_position_id`: Start index of answer tokens (for masking prompts during loss calculation)
- `ingredients_start_token`, `ingredients_end_token`: Token positions marking ingredient spans for targeted losses
- `NER_ingredients_quantity`, `NER_ingredients`: Structured data for multi-task regression heads
- `task_type`: Task identifier for multi-task learning
- `messages`: Conversational format with system/user/assistant roles

Outputs parquet files to `data/{dataset_name}/{model_name}/{dataset_source}/` (train/validation/test splits).

### 2. Training with Custom Losses
Run fine-tuning with Hydra config composition:
```bash
# Use specific config by name (references configs/loss_*.yaml)
python improved_loss.py --config-name loss_topological-recipenlg_Qwen3-4B
python improved_loss.py --config-name loss_dice
python improved_loss.py --config-name loss_cross_entropy_mathqa_qwen2_1_5B
```
**Config resolution**: Hydra composes `defaults` from root config → model → dataset → trainer YAML files.
**Outputs**: 
- Checkpoints: `checkpoints/{experiment_name}-{timestamp}/`
- Final LoRA adapters: `finetuned_models/{experiment_name}-{timestamp}/`
- Logs: CometML experiment tracking (if `log_comet: true`)

### 3. Inference/Evaluation
Run trained models on test sets:
```bash
# Baseline (pre-trained) model
python baseline_inference.py \
  --model qwen3-4 \
  --dataset recipe_nlg \
  --task recipe_nlg \
  --batch_size 8

# Fine-tuned model (requires model_finetuned_id from training output)
python baseline_inference.py \
  --model qwen3-4 \
  --dataset recipe_nlg \
  --task recipe_nlg \
  --batch_size 8 \
  --model_finetuned_id "Qwen3-4B-CrossEntropyLoss-RecipeNLG-2025-09-01-16-13-08"

# Structured output (uses Outlines for JSON schema validation)
python baseline_inference.py \
  --model qwen3-4 \
  --dataset recipe_nlg \
  --task recipe_nlg \
  --batch_size 8 \
  --structured_output true
```
**Model aliases** (in `baseline_inference.py` MODELS dict): `qwen2-1.5`, `qwen3-4`, `qwen3-8`, `mistral`, `mammoth`, `gemma3`.
**Outputs**: Predictions saved to `results/sota/{dataset}/{model_name_or_id}_{timestamp}.parquet` with columns: `prediction`, `ground_truth`, `title` (for recipes).

### 4. Metrics Computation
[evaluation.py](evaluation.py) computes task-specific metrics via [utils/metric_calculator.py](utils/metric_calculator.py):
- **Recipe NLG**: Ingredient overlap, instruction similarity
- **Math QA**: Formula equivalence (handles operator commutativity), numeric accuracy
- **General**: Edit distance, BLEU, ROUGE

## Critical Implementation Details

### LoRA Configuration
Target modules vary by model architecture (see [improved_loss.py](improved_loss.py) L116-135):
- **Qwen/SmolLM**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Falcon**: `["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]`
- **LLaMA/StableLM**: All attention + MLP projections + lm_head

### Multi-task Training Gotchas
- Must untie embeddings before LoRA: `model.llm.config.tie_word_embeddings = False` and clone lm_head weights
- Apply LoRA only to `.llm` attribute of `MultiTaskRecipeModel`
- Use `MultiTaskImprovedLossTrainingArgs` which passes `ingredient_list` to trainer
- Data collator must set `return_tensors=None` for custom fields

### Loss Computation Pattern
In `compute_loss()`:
1. Shift logits/labels for autoregressive modeling: `logits[..., :-1, :]`, `labels[..., 1:]`
2. Create `ingredients_mask` from pre-computed token positions for targeted losses
3. Apply `processed_labels` masking (set to -100 before answer start) for non-CE losses
4. Compute only losses specified in `self.args.losses[task]` dict
5. Return weighted sum: `sum([losses[k] * w for k, w in task_losses.items()])`

### Memory Management
Training step calls `gc.collect()` and `torch.cuda.empty_cache()` every `logging_steps` to prevent OOM.

## Dataset Structure
Parquet files must contain (beyond standard HF fields):
- `labels_position_id`: List with single int (answer start index)
- `ingredients_start_token`, `ingredients_end_token`: Ints marking ingredient span
- `attention_mask`: Cast to `Sequence(Value("bool"))` explicitly
- `task`: Int index from `MetricsCalculator.task_map`

For multi-task: Add `NER_ingredients_quantity`, `NER_ingredients`, `task_type`, `answer` (ground truth for regression).

## Common Pitfalls
- **Hydra configs**: Missing defaults or wrong nesting breaks config resolution silently
- **Token masking**: Off-by-one errors in `labels_position_id` cause losses to train on prompts
- **Multi-task**: Forgetting to resize embeddings after tokenizer updates causes shape mismatches
- **Loss registration**: Adding new loss requires updating both Trainer's `__init__` and `compute_loss()` logic

## Security
Follow [.github/instructions/snyk_rules.instructions.md](.github/instructions/snyk_rules.instructions.md): Run `snyk_code_scan` on all new code, fix issues, rescan until clean.

## Dependencies
Key packages: `transformers`, `peft`, `hydra-core`, `datasets`, `geomloss` (for topological loss), `comet-ml` (experiment tracking). See [requirements.txt](requirements.txt).
