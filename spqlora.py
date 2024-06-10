from instantnet.quantize import QLinear, Quantize, QParams, calculate_qparams
from loralib.layers import Linear as LoRALinear, MergedLinear as LoRAMergedLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
SPQ-LoRA: Switchable-Precision Quantized LoRA modules
"""
class SPQLinear(QLinear, LoRALinear):
    def __init__(
            self, 
            in_features, 
            out_features, 
            bias=True, 
            num_bits=8, 
            num_bits_weight=8, 
            num_bits_grad=8, 
            biprecision=False,
            r: int = 0, 
            lora_alpha: int = 1, 
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs):
        QLinear.__init__(self, in_features, out_features, bias, num_bits, num_bits_weight, num_bits_grad, biprecision)
        LoRALinear.__init__(self, in_features, out_features, r, lora_alpha, lora_dropout, fan_in_fan_out, merge_weights, **kwargs)
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    """
    If rank is defined passes forward pass through LoRA module.
    """
    def pass_through_LoRA(self, input: torch.Tensor, weight, bias):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(input, T(weight), bias=bias)            
            result += (self.lora_dropout(input) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(input, T(weight), bias=bias)
        
    def forward(self, input, num_bits):
        if num_bits < 32:
            if self.training:
                qparams = calculate_qparams(
                        input, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=0, reduce_type='extreme')
                with torch.no_grad():
                    self.running_zero_point.mul_(self.momentum).add_(
                        qparams.zero_point * (1 - self.momentum))
                    self.running_range.mul_(self.momentum).add_(
                        qparams.range * (1 - self.momentum))
            else:
                qparams = QParams(range=self.running_range,
                  zero_point=self.running_zero_point, num_bits=num_bits)

            qinput = Quantize(input, qparams=qparams, dequantize=True,
                               stochastic=False, inplace=False)

            weight_qparams = calculate_qparams(
                self.weight, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
            qweight = Quantize(self.weight, qparams=weight_qparams)

            if self.bias is not None:
                qbias = Quantize(
                    self.bias, num_bits=num_bits,
                    flatten_dims=(0, -1))
            else:
                qbias = None

            output = self.pass_through_LoRA(qinput, qweight, qbias)

        else:
            output = self.pass_through_LoRA(input, self.weight, self.bias)

        return output
    
class SQPMergedLinear(QLinear, LoRAMergedLinear):
    # implement this