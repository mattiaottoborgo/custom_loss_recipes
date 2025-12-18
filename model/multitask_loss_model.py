import gc
from dataclasses import dataclass, field
from typing import Dict

import datasets
import torch
from loss import FocalLoss, GDiceLoss,MultiLabelDiceLoss, SelfAdjDiceLoss, lovasz_softmax_flat, TopologicalLoss
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import seed_worker
from functools import partial  # <--- ADD THIS IMPORT
from utils import MetricsCalculator
import ast

class MultiTaskImprovedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        """
        Initializes the Trainer. We instantiate the loss functions here
        so they are not recreated on every training step, which is more efficient.
        """
        super().__init__(*args, **kwargs)
        
        # Instantiate all required loss functions once.
        # The specific losses to be used for a given task will be
        # determined by the training arguments inside compute_loss.
        self.topological_loss_fn = TopologicalLoss()
        self.ce_loss_fn = torch.nn.CrossEntropyLoss()
        self.focal_loss_fn = FocalLoss(gamma=2, reduction="mean")
        self.mse_loss_fn = torch.nn.MSELoss()
        
    def training_step(self, model, inputs,num_items_in_batch):
        out = super().training_step(model, inputs,num_items_in_batch)
        if self.state.global_step % self.args.logging_steps == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return out

    def compute_loss(self, model, inputs,num_items_in_batch=1, return_outputs=False):
        # The 'labels' are for the text generation part
        ingredient_list = self.args.ingredient_list
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        ground_truth_regression = inputs["ground_truth_prediction"]

                #print(f"Ingredient: '{ingredient_name}' (at index {index}) has quantity: {quantity}")
        task_type = inputs["task_type"][0]


        # Forward pass through your custom multi-task model
        outputs = model(
            input_ids=input_ids,
            attention_mask=(
                attention_mask
            ),
            labels=labels,
        )
        logits = outputs["lm_logits"]
        predicted_quantities = outputs["quantities"]
        presence_logits = outputs["presence_logits"]
        predicted_temperature = outputs.get("temperature")
        predicted_single_quantity = outputs.get("single_quantity")
        predicted_time = outputs.get("time")

        #print(f"Prediction shape: {predicted_quantities.shape}")
        #print(f"Reshaped target shape: {quantity_targets_tensor.shape}")
        # --- Calculate the two separate losses ---

        # 1. Cross-Entropy Loss for text
        # (Your existing logic for shifting logits and labels)
        #loss_fct_ce = torch.nn.CrossEntropyLoss()
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        #loss_ce = loss_fct_ce(logits.view(-1, logits.size(-1)), labels.view(-1))

        # 2. MSE Loss for quantities
        #loss_fct_mse = torch.nn.MSELoss()
        #loss_mse = loss_fct_mse(masked_predicted_quantities, masked_quantity_targets)

        # --- Combine the losses ---
        #alpha = 0.8 # Example weight: 70% text, 30% regression
        #total_loss = alpha * loss_ce + (1 - alpha) * loss_mse
        #print("congrats,you manage to make a run of compute loss!")
        #return (total_loss, outputs) if return_outputs else total_loss
        task_map = MetricsCalculator.task_map
        task_losses = self.args.losses[task_map[inputs["task"].min()]]

        losses = {}
        if task_type in ["recipe","missing_ingredient_identification","substitution_validation","recipe_scaling"]:
            if task_type == "recipe" and inputs["NER_ingredients_quantity"] is not None:
                ner_ingredients_quantity = ast.literal_eval(inputs["NER_ingredients_quantity"][0])
                quantity_targets = []
                for ingredient in ingredient_list:
                    quantity = ner_ingredients_quantity.get(ingredient, {'quantity': 0})['quantity']
                    if isinstance(quantity,str):
                        try:
                            quantity = ast.literal_eval(quantity)
                            quantity_targets.append(quantity)
                        except Exception as e:
                            quantity_targets.append(0)
                    else:
                        quantity_targets.append(quantity)
                    
                #quantity_targets = [ner_ingredients_quantity.get(ingredient, {'quantity': 0})['quantity'] for ingredient in ingredient_list]
                #print(len(quantity_targets))
                ingredient_indexes = [index for index, ingredient in enumerate(ingredient_list) if ingredient in ner_ingredients_quantity]
                #print("Indexes of the recipe's ingredients:")
                #print(ingredient_indexes)
                #print("\nLooping through indexes to get quantities:")
                for index in ingredient_indexes:
                    # Get the ingredient name and quantity using the index
                    ingredient_name = ingredient_list[index]
                    quantity = quantity_targets[index]
            if task_type == "recipe":
                ing_start_tokens = inputs["ingredients_start_token"]
                ing_end_tokens = inputs["ingredients_end_token"]
                ingredients_mask = torch.zeros_like(labels)

                for i in range(labels.size(0)):
                    start_idx = ing_start_tokens[i].item()
                    # The end token from the data script is inclusive, so we add 1 for Python slicing.
                    end_idx = ing_end_tokens[i].item() + 1

                    # Ensure indices are valid before creating the mask slice.
                    if start_idx != -1 and end_idx > start_idx:
                        ingredients_mask[i, start_idx:end_idx] = 1
            if "cross_entropy" in task_losses:
                #loss_fct = torch.nn.CrossEntropyLoss()
                #lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                lm_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                losses["cross_entropy"] = lm_loss

            if "gdice" in task_losses:
                quantity_targets_tensor = torch.tensor(
                        quantity_targets, 
                        device=predicted_quantities.device, 
                        dtype=predicted_quantities.dtype
                    )
                presence_labels = (quantity_targets_tensor > 0).float()
                #dice_fct = GDiceLoss(sigmoid=True) 
                dice_fct = MultiLabelDiceLoss()
                dice_loss = dice_fct(presence_logits, presence_labels.float())

                #dice_fct = GDiceLoss(apply_nonlin=torch.nn.Softmax(dim=1))
                #dice_loss = dice_fct(masked_predicted_quantities, masked_quantity_targets)
                losses["gdice"] = dice_loss

            if "focal" in task_losses:
                focal_fct = FocalLoss(gamma=2, reduction="mean")
                focal_loss = focal_fct(
                    processed_logits.reshape(-1, logits.size(-1)),
                    processed_labels.reshape(-1),
                )
                losses["focal"] = focal_loss

            if "lovasz" in task_losses:
                processed_probs = torch.nn.functional.softmax(processed_logits, dim=-1)
                lovasz_loss = lovasz_softmax_flat(
                    processed_probs.reshape(-1, logits.size(-1)),
                    processed_labels.reshape(-1),
                )
                losses["lovasz"] = lovasz_loss

            if "self_adj_dice" in task_losses:
                dice_fct = SelfAdjDiceLoss()
                dice_loss = dice_fct(
                    processed_logits.reshape(-1, logits.size(-1)),
                    processed_labels.reshape(-1),
                )
                losses["self_adj_dice"] = dice_loss

            if "focal_ingredients" in task_losses:
                print("this is focal_ingredients in action!")
                # Use the single, shared ingredients_mask
                active_positions = (ingredients_mask == 1) #& (labels != -100)
                masked_logits = logits[active_positions]
                masked_labels = labels[active_positions]

                if masked_labels.numel() > 0:
                    print(f"Ingredients found! Masked labels count: {masked_labels.numel()}")
                    focal_loss = self.focal_loss_fn(masked_logits, masked_labels)
                    losses["focal_ingredients"] = focal_loss
                else:
                    print("No ingredients found in this batch, setting focal loss to 0.")
                    # Add a zero loss to prevent a KeyError during the final summation
                    losses["focal_ingredients"] = torch.tensor(0.0, device=logits.device)

            if "topological" in task_losses:
                embedding_matrix = model.llm.get_input_embeddings().weight

                # Call the instantiated loss class with the appropriate inputs.
                topo_loss = self.topological_loss_fn(logits, labels, ingredients_mask, embedding_matrix)
                losses["topological"] = topo_loss
            if "mse" in task_losses and task_type == "recipe":
                loss_fct_mse = torch.nn.MSELoss()
                quantity_targets_tensor = torch.tensor(
                        quantity_targets, 
                        device=predicted_quantities.device, 
                        dtype=predicted_quantities.dtype
                    )
                presence_labels = (quantity_targets_tensor > 0).float()
                quantity_targets_tensor = torch.log1p(quantity_targets_tensor)
                quantity_targets_tensor = quantity_targets_tensor.unsqueeze(0)
                mask = (quantity_targets_tensor > 0).float()
                # Apply the mask to both the predictions and targets
                masked_predicted_quantities = predicted_quantities * mask
                masked_quantity_targets = quantity_targets_tensor * mask
                loss_mse = loss_fct_mse(masked_predicted_quantities, masked_quantity_targets) 
                losses["mse"] = loss_mse
        elif task_type in ["quantity_ingredient", "time", "temperature"]:
            if isinstance(ground_truth_regression[0],str):
                print("evaluating regression ground truth to convert to number!")
                ground_truth_regression = ast.literal_eval(ground_truth_regression[0])
            
            
            # 1. Select the correct prediction head based on the task type
            prediction = None
            if task_type == "quantity_ingredient":
                prediction = predicted_single_quantity
            elif task_type == "time":
                prediction = predicted_time
            elif task_type == "temperature":
                prediction = predicted_temperature
            
            # 2. Calculate MSE loss if prediction exists
            if prediction is not None:
                # Squeeze the prediction tensor from [batch_size, 1] to [batch_size]
                # to match the ground truth shape.
                prediction = prediction.squeeze(-1)
                
                # Ensure ground truth tensor has the correct device and dtype
                #ground_truth = ground_truth_regression.to(prediction.device, dtype=prediction.dtype)
                print("debug ground_truth_regression")
                print(type(ground_truth_regression),ground_truth_regression)
                ground_truth = torch.tensor(ground_truth_regression, device=prediction.device, dtype=prediction.dtype)
                # Calculate the loss
                loss_mse = self.mse_loss_fn(prediction, ground_truth)
                losses["mse_regression"] = loss_mse
        else:
            print("something went wrong, why this task_type?")
            print(task_type)
            exit(0)
        total_loss = 0
        for k,w in task_losses.items():
            loss = losses.get(k,None)
            if loss is not None:
                loss = loss * w
                total_loss = total_loss + loss
        #loss = sum(losses.get(k, 0.0) * w for k, w in task_losses.items())

        return total_loss if not return_outputs else (total_loss, outputs)
    
    def get_train_dataloader(self) -> DataLoader:
        train_dataloaders = {
            k: self.get_single_dataloader(v) for k, v in self.train_dataset.items()
        }
        if len(train_dataloaders.keys()) != 1:
            raise ValueError("Training multiple dataloader is not supported")
        return list(train_dataloaders.values())[0]

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        eval_dataloaders = {
            k: self.get_single_dataloader(v) for k, v in self.eval_dataset.items()
        }
        if len(eval_dataloaders.keys()) != 1:
            raise ValueError("Training multiple dataloader is not supported")
        return list(eval_dataloaders.values())[0]

    def get_single_dataloader(self, dataset) -> DataLoader:
            data_collator = self.data_collator
            if isinstance(dataset, datasets.Dataset):
                dataset = self._remove_unused_columns(dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(
                    data_collator, description="training"
                )

            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }

            if not isinstance(dataset, IterableDataset):
                dataloader_params["drop_last"] = self.args.dataloader_drop_last

                # --- MANUALLY CREATE THE WORKER_INIT_FN ---
                # This is the robust way to fix the original TypeError
                worker_init_fn = partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )
                dataloader_params["worker_init_fn"] = worker_init_fn

            return DataLoader(dataset, **dataloader_params)


@dataclass
class MultiTaskImprovedLossTrainingArgs(TrainingArguments):
    losses: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {"summarization": {"cross_entropy": 1.0}},
        metadata={"help": "losses to use and respective weights"},
    )
    ingredient_list: list = field(
        default=None, 
        metadata={"help": "A list of all unique ingredients in the dataset."}
    )
