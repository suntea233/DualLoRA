import math
import torch.nn as nn
import torch
from loralib import LoRALayer
from torch import Tensor
import torch.nn.functional as F

class ConLoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_con_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_con_B = nn.Parameter(self.weight.new_zeros((out_features, r)))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_con_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            nn.init.zeros_(self.lora_con_B)


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


    def forward(self, x, prompt_emb=None,prompt_embed_mask=None,hidden_attention_mask=None,global_prompt=None,p_bias=None):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result = result + (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            glora_result = (self.lora_dropout(global_prompt) @ self.lora_con_A.transpose(0, 1) @ self.lora_con_B.transpose(0,1)) * self.scaling
            result = result + 0.5 * glora_result.mean(dim=1).unsqueeze(dim=1)
            return result
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result = result + p_bias.mean(dim=1).unsqueeze(dim=1)
            return result

class ControlPrompt(nn.Module):
    def __init__(self,in_proj,out_proj,lora_r,lora_alpha,lora_dropout,bias,args):
        super().__init__()
        self.lora = ConLoRALinear(in_proj, out_proj, lora_r, lora_alpha, lora_dropout=lora_dropout, bias=bias)

    def forward(self,x,prompt_emb=None,prompt_embed_mask=None,hidden_attention_mask=None,global_prompt=None,p_bias=None):
        hidden_state = self.lora(x,prompt_emb,prompt_embed_mask,hidden_attention_mask,global_prompt,p_bias)
        return hidden_state
