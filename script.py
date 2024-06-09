# Load model directly
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import loralib as lora
import torch
import torch.nn as nn


model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Load config. There are 49 total linear layers
quant_layer_precision = [0]*49
lora_layer_ranks = [0]*49

# Create the quantization configurations list
layer_quant_configs = [
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 8, 'weight_bitwidth': 4},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3},
    {'activation_bitwidth': 6, 'weight_bitwidth': 3}
]

# Function to create FakeQuantize modules
def create_fake_quant_modules(activation_bitwidth, weight_bitwidth):
    fq_activation = torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MinMaxObserver.with_args(
            quant_min=0, 
            quant_max=2**activation_bitwidth - 1,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False
        )
    )
    fq_weights = torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAveragePerChannelMinMaxObserver.with_args(
            quant_min=-(2 ** (weight_bitwidth - 1)),
            quant_max=(2 ** (weight_bitwidth - 1)) - 1,
            dtype=torch.qint8, 
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            ch_axis=0
        )
    )
    return fq_activation, fq_weights

# # Add LoRA modules
# layer_id = 0
# for name, module in model.named_modules():
#     if isinstance(module, transformers.pytorch_utils.Conv1D) or isinstance(module, nn.Linear):
#         if isinstance(module, nn.Linear):
#             # replace with LoRA linear layer
#             setattr(model, name, lora.Linear(module.in_features, module.out_features, r=16))
#             layer_id += 1
#         elif isinstance(module, nn.Conv1d):
#             # replace with LoRA convolutional layer
#             setattr(model, name, lora.Conv1d(module.in_channels, module.out_channels, module.kernel_size[0], r=16))
#             layer_id += 1

# Quantize model
layer_id = 0
for name, module in model.named_modules():
    if isinstance(module, transformers.pytorch_utils.Conv1D) or isinstance(module, nn.Linear) or isinstance(module, lora.Linear):
        config = layer_quant_configs[layer_id]
        activation_bitwidth = config['activation_bitwidth']
        weight_bitwidth = config['weight_bitwidth']
        
        fq_activation, fq_weights = create_fake_quant_modules(activation_bitwidth, weight_bitwidth)
        qconfig = torch.quantization.QConfig(activation=fq_activation, weight=fq_weights)
        
        # Apply the quantization configuration to the module
        module.qconfig = qconfig
        print(f"Quantizing layer {layer_id} with activation bitwidth {activation_bitwidth} and weight bitwidth {weight_bitwidth}")


        layer_id += 1

# Set the model to evaluation mode
model.eval()

# Prepare the model for quantization
torch.quantization.prepare(model, inplace=False)
quantized_model = torch.quantization.convert(model, inplace=False)
print(quantized_model)

# Assuming `tokenizer` is already defined and you have a sample input
sample_text = "This is a sample input."
inputs = tokenizer(sample_text, return_tensors="pt")

# Regular model output
with torch.no_grad():
    regular_output = model(**inputs)

# Quantized model output
with torch.no_grad():
    quantized_output = quantized_model(**inputs)

# Compare outputs
print("Regular Model Output:")
print(regular_output)
print("\nQuantized Model Output:")
print(quantized_output)

# Compare output tensors element-wise
output_equal = torch.equal(regular_output.logits, quantized_output.logits)
print("\nOutputs are equal:", output_equal)
