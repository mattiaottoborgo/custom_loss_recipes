import torch
from torch import nn
from transformers import AutoModelForCausalLM

class MultiTaskRecipeModel(nn.Module):
    def __init__(self, model_name, num_unique_ingredients):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Get the size of the LLM's hidden states
        hidden_size = self.llm.config.hidden_size
        
        # --- Head 1: Causal LM Head (already part of self.llm) ---
        # We will access its output via the standard llm forward pass.

        # --- Head 2: Multi-Output Regression Head ---
        self.quantity_regression_head = nn.Linear(hidden_size, num_unique_ingredients)
        
        # --- Head 3: Multi-Label Classification for Ingredient PRESENCE
        self.ingredient_presence_head = nn.Linear(hidden_size, num_unique_ingredients)
        # --- Head 4: Head for temperature regression (single value) ---
        self.temperature_head = nn.Linear(hidden_size, 1)

        # --- Head 5: Head for single ingredient quantity regression (single value) ---
        self.single_quantity_head = nn.Linear(hidden_size, 1)

        # --- Head 6: Head for time regression (single value) ---
        self.time_head = nn.Linear(hidden_size, 1)
        

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get the full output from the base language model
        transformer_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            labels=labels  # Pass labels to get lm_loss if needed
        )
        
        # --- Get Outputs for Each Head ---
        
        # 1. For the Causal LM Head: The standard logits
        lm_logits = transformer_outputs.logits
        
        # 2. For the Regression Head:
        # We use the hidden state of the very last token as a summary of the sequence.
        last_hidden_state = transformer_outputs.hidden_states[-1][:, -1, :]
        presence_logits = self.ingredient_presence_head(last_hidden_state)
        predicted_quantities = self.quantity_regression_head(last_hidden_state)
        predicted_temperature = self.temperature_head(last_hidden_state)
        predicted_single_quantity = self.single_quantity_head(last_hidden_state)
        predicted_time = self.time_head(last_hidden_state)
        
        # Return a dictionary containing all the outputs
        return {
            "lm_logits": lm_logits,
            "presence_logits": presence_logits,
            "quantities": predicted_quantities,
            "temperature": predicted_temperature,
            "single_quantity": predicted_single_quantity,
            "time": predicted_time,
            "lm_loss": transformer_outputs.loss # The model can still return the text loss
        }