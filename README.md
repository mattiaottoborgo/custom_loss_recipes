# Custom Loss Functions for Structured Recipe Generation

Research codebase for paper "Losses that Cook: Topological Optimal Transport for Structured Recipe Generation", currently under Review in ACL (link to research: https://arxiv.org/abs/2601.02531). with custom loss functions optimized for structured recipe generation. This project demonstrates that carefully designed loss functions can significantly improve ingredient recall, quantity precision, and procedural accuracy without increasing model size or inference-time complexity.

## 🎯 Problem Statement

Generating usable cooking recipes with language models requires more than fluent text: models must produce ingredients, quantities, and step-by-step instructions that are **factually correct**, **numerically plausible**, and **procedurally executable**. Errors on a few key tokens (e.g., omitting "eggs" in carbonara pasta or doubling the cooking temperature) can render the entire recipe unusable, even if the text is fluent.

**Standard cross-entropy (CE) loss is ill-suited** to this challenge because it treats all tokens as equally important, despite a strong asymmetry between:
- **High-impact tokens**: ingredients, quantities, times, temperatures, core actions
- **Low-impact tokens**: connective words, articles, prepositions

This misalignment manifests in common failure modes: poor ingredient recall, inaccurate quantities, and instruction sequences that are syntactically plausible but procedurally incorrect.

## 💡 Solution: Custom Loss Functions

This project investigates four custom loss functions that explicitly model the structured nature of recipes:

### 1. **Topological Loss** (Novel Contribution)
Represents ingredient lists as **point clouds in embedding space** and minimizes Sinkhorn divergence between predicted and gold ingredients. This explicitly encodes ingredient-level structure beyond token-wise CE.

**Implementation**: [loss/topological_loss.py](loss/topological_loss.py)
- Uses `geomloss` library for differentiable optimal transport
- Compares predicted vs. ground truth ingredient embeddings
- Targets tokens within ingredient spans using pre-computed masks

### 2. **Generalized Dice Loss (GDice)**
Adapted from medical image segmentation, uses einsum operations to compute overlap between predicted and target distributions.

**Implementation**: [loss/dice_loss.py](loss/dice_loss.py)
- Handles class imbalance naturally
- Effective for emphasizing rare but critical tokens

### 3. **Focal Loss**
Addresses class imbalance by down-weighting easy examples and focusing on hard-to-predict tokens (γ=2 default).

**Implementation**: [loss/focal_loss.py](loss/focal_loss.py)
- Particularly effective for time/temperature precision

### 4. **Lovasz Loss**
Optimizes directly for IoU (Intersection over Union) using the Lovász extension of submodular functions.

**Implementation**: [loss/lovasz.py](loss/lovasz.py)

### Loss Combination Strategy
Losses can be weighted and combined via Hydra configuration:
```yaml
losses:
  question_answering:
    cross_entropy: 0.6
    topological: 0.4
```

## 📊 Evaluation: Recipe-Specific Metrics

Standard NLG metrics (ROUGE, BERTScore) capture fluency and semantic similarity but fail to measure whether a recipe accurately specifies ingredients, quantities, and cooking parameters.

**Custom metrics** implemented in [utils/metric_calculator.py](utils/metric_calculator.py):
- **Ingredient Recall**: Percentage of ground truth ingredients present in prediction
- **Quantity Precision**: Numerical accuracy of ingredient amounts
- **Action/Step Edit Distance**: Similarity of procedural instructions
- **Time/Temperature Precision**: Accuracy of cooking parameters
- **Formula Equivalence** (Math QA): Handles operator commutativity for arithmetic tasks

## 🏗️ Architecture

### Core Components

**Training Pipeline**:
- [improved_loss.py](improved_loss.py) - Main entry point with Hydra configuration
- [model/loss_model.py](model/loss_model.py) - `ImprovedLossTrainer` extends HuggingFace Trainer
- [model/multitask_loss_model.py](model/multitask_loss_model.py) - Multi-task trainer for text + regression

**Multi-Task Learning**:
- [model/multitask_model.py](model/multitask_model.py) - Wraps base LLM with regression heads:
  - Ingredient quantities (multi-output regression)
  - Ingredient presence (multi-label classification)
  - Temperature, time, single quantity predictions

**Configuration System** (Hydra):
```
configs/
├── loss_*.yaml              # Root configs (experiment definitions)
├── model/                   # Model specifications
├── dataset/                 # Dataset paths and splits
└── trainer/                 # Training hyperparameters + loss weights
```

## 🚀 Quick Start

### 1. Dataset Preparation

Create tokenized parquet files with custom fields for targeted loss application:

```bash
# Recipe NLG (pasta, rice, sandwiches subset)
python parquet_creator.py \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --dataset recipe_nlg \
  --prompt_type recipe_nlg

# Math QA (for comparison experiments)
python parquet_creator.py \
  --tokenizer mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset math_qa \
  --prompt_type math_qa
```

**What it does**:
- Tokenizes conversational format (system/user/assistant messages)
- Adds `labels_position_id` for prompt masking during loss calculation
- Computes `ingredients_start_token`/`ingredients_end_token` for targeted losses
- Creates train/validation/test splits (80%/10%/10%)

**Outputs**: `data/{dataset_name}/{model_name}/{dataset_source}/*.parquet`

### 2. Training with Custom Losses

Run fine-tuning with PEFT/LoRA for efficient training:

```bash
# Topological loss (best for ingredient recall)
python improved_loss.py --config-name loss_topological-recipenlg_Qwen3-4B

# Dice loss (best for time/temperature precision)
python improved_loss.py --config-name loss_dice-recipenlg_Qwen3-4B

# Mixed loss (balanced trade-offs)
python improved_loss.py --config-name loss_mixed-recipenlg_Qwen3-4B

# Standard cross-entropy baseline
python improved_loss.py --config-name loss_cross_entropy-recipenlg_Qwen3-4B
```

**Outputs**:
- Checkpoints: `checkpoints/{experiment_name}-{timestamp}/`
- LoRA adapters: `finetuned_models/{experiment_name}-{timestamp}/`
- CometML logs (if `log_comet: true` in config)

### 3. Inference & Evaluation

Run trained models on test sets:

```bash
# Baseline (pre-trained) model
python baseline_inference.py \
  --model qwen3-4 \
  --dataset recipe_nlg \
  --task recipe_nlg \
  --batch_size 8

# Fine-tuned model
python baseline_inference.py \
  --model qwen3-4 \
  --dataset recipe_nlg \
  --task recipe_nlg \
  --batch_size 8 \
  --model_finetuned_id "Qwen3-4B-TopologicalLoss-RecipeNLG-2025-09-01-16-13-08"

# Structured output with schema validation (Outlines)
python baseline_inference.py \
  --model qwen3-4 \
  --dataset recipe_nlg \
  --task recipe_nlg \
  --batch_size 8 \
  --structured_output true
```

**Supported models**: `qwen2-1.5`, `qwen3-4`, `qwen3-8`, `qwen3-14`, `mistral`, `mammoth`, `gemma3`

**Outputs**: `results/sota/{dataset}/{model_id}_{timestamp}.parquet` with columns:
- `prediction`: Generated recipe (JSON format)
- `ground_truth`: Reference recipe
- `title`: Recipe name

### 4. Compute Metrics

Evaluate predictions using recipe-specific metrics:

```bash
python evaluation.py
```

Computes task-specific metrics via [utils/metric_calculator.py](utils/metric_calculator.py).

## 📈 Key Results

Experiments on RECIPE-NLG (pasta, rice, sandwiches) demonstrate:

- **Topological Loss**: Substantially improves ingredient recall, quantity precision, and procedural accuracy over CE alone
- **Dice Loss**: Excels in time and temperature precision
- **Mixed Loss**: Yields well-rounded trade-offs and synergistic gains (e.g., in quantity and time precision)

**Key insight**: Many improvements over CE and single custom losses show that carefully designed loss functions meaningfully improve structured recipe generation without increasing model size or inference-time complexity.

## 🔧 Implementation Details

### LoRA Configuration (Model-Specific)
Target modules vary by architecture ([improved_loss.py](improved_loss.py) L116-135):
```python
# Qwen/SmolLM
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Falcon
target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

# LLaMA/StableLM
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj", "lm_head"]
```

### Loss Computation Pattern
In `ImprovedLossTrainer.compute_loss()`:
1. **Shift logits/labels** for autoregressive modeling: `logits[..., :-1, :]`, `labels[..., 1:]`
2. **Create ingredient mask** from pre-computed token positions
3. **Apply prompt masking** (set labels to -100 before answer start)
4. **Compute task-specific losses** from config: `self.args.losses[task]`
5. **Return weighted sum**: `sum([losses[k] * w for k, w in task_losses.items()])`

### Multi-Task Training
For models that jointly learn text generation + regression:
- Untie embeddings before LoRA: `model.llm.config.tie_word_embeddings = False`
- Apply LoRA only to `.llm` attribute (not regression heads)
- Use `CustomDataCollator` with `return_tensors=None` for non-tensor fields
- Pass `ingredient_list` to `MultiTaskImprovedLossTrainingArgs`

### Memory Management
Training calls `gc.collect()` + `torch.cuda.empty_cache()` every `logging_steps` to prevent OOM on GPU.

## 📂 Project Structure

```
.
├── improved_loss.py              # Main training script (Hydra entry point)
├── parquet_creator.py            # Dataset tokenization & preprocessing
├── baseline_inference.py         # Inference with base/fine-tuned models
├── evaluation.py                 # Metric computation
├── configs/                      # Hydra configuration hierarchy
│   ├── loss_*.yaml              # Experiment definitions
│   ├── model/*.yaml             # Model specifications
│   ├── dataset/*.yaml           # Dataset paths
│   └── trainer/*.yaml           # Training args + loss weights
├── loss/                         # Custom loss implementations
│   ├── topological_loss.py      # Sinkhorn divergence on embeddings
│   ├── dice_loss.py             # GDice + MultiLabelDice
│   ├── focal_loss.py            # Class imbalance handling
│   └── lovasz.py                # IoU optimization
├── model/                        # Trainer extensions
│   ├── loss_model.py            # ImprovedLossTrainer (single-task)
│   ├── multitask_loss_model.py  # Multi-task trainer
│   └── multitask_model.py       # LLM + regression heads
└── utils/
    └── metric_calculator.py      # Recipe-specific metrics
```

## 📦 Dependencies

Key packages:
- `transformers` - HuggingFace model library
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `hydra-core` - Configuration management
- `datasets` - Dataset loading & processing
- `geomloss` - Sinkhorn divergence for topological loss
- `comet-ml` - Experiment tracking
- `torch` - PyTorch backend
- `outlines` - Structured output generation

See [requirements.txt](requirements.txt) for complete list.

## 🎓 Dataset

**RECIPE-NLG Corpus** (Bień et al., 2020) - Focused subset:
- Categories: Pasta, rice, sandwiches
- Structured format: Ingredients (with quantities) + Instructions
- Split: 80% train / 10% validation / 10% test
- Stored as parquet with custom fields for loss targeting

