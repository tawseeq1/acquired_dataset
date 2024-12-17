import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from transformers import AutoProcessor, AutoModelForImageTextToText
# Define model name and directory to save it
#model_name = 'microsoft/phi-2'
#model_id = "google/paligemma-3b-mix-224"
model_save_dir = '/scratch/22ch10090/models'  # Path to save the model

# Ensure the directory exists
os.makedirs(model_save_dir, exist_ok=True)

# Download model and tokenizer
#print(f"Downloading model {model_id}...")
processor = AutoProcessor.from_pretrained("facebook/chameleon-7b")
model = AutoModelForImageTextToText.from_pretrained("facebook/chameleon-7b")

# Save the model and tokenizer locally
model.save_pretrained(model_save_dir)
processor.save_pretrained(model_save_dir)

print(f"Model and tokenizer saved to {model_save_dir}")
