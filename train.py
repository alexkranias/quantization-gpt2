from config_quantize import config
import loralib as lora
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

def replaceWithLoRALayers(model: nn.Module, rank: int):
    """
    Replaces all Linear and Conv1D layers with the
    respective LoRA layers of the specified rank.

    rank = rank of all LoRA layers
    """
    new_model = nn.Module()
    for idx, (name, module) in enumerate(model.named_modules()):
        print(idx, name, module, type(module))
        isConv1d = isinstance(module, transformers.pytorch_utils.Conv1D) or isinstance(module, lora.Conv1d)
        isLinear = isinstance(module, nn.Linear) or isinstance(module, lora.Linear)

        # swap respective modules with new LoRA counterpart
        if "attn.c_attn" in name: 
            in_feat = module.weight.shape[1]
            out_feat = module.weight.shape[0]
            # replace with lora.Conv1d
            lora_layer = lora.Linear(in_features=in_feat, out_features=out_feat, r=rank)
            setattr(new_model, name, lora_layer)
        elif "attn.c_proj" in name:
            in_feat = module.weight.shape[1]
            out_feat = module.weight.shape[0]
            """
            assuming query, key, value ordering...
            
            enable_lora=[True, False, True]...sets query (LoRA), key (No LoRA), value (LoRA)
            """
            lora_layer = lora.MergedLinear(in_features=in_feat, out_features=3*in_feat, enable_lora=[True, False, True], r=rank)
            setattr(new_model, name, lora_layer)
        elif "mlp.c_fc" in name:
            in_feat = module.weight.shape[1]
            out_feat = module.weight.shape[0]
            # replace with lora.Linear
            lora_layer = lora.Linear(in_features=in_feat, out_features=out_feat, r=rank)
            setattr(new_model, name, lora_layer)
        # elif "lm_head":
        #      # replace with lora.Linear
        #     lora_layer = lora.Linear(module.in_channels, module.out_channels, r=rank)
        #     setattr(model, name, lora_layer)
        else:
            setattr(new_model, name, module)
    return new_model

# Get model+tokenizer from HuggingFace
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

print(model)

model = replaceWithLoRALayers(model, rank=16)

print(model)