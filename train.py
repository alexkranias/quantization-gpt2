from config_quantize import config

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Get model+tokenizer from HuggingFace
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
