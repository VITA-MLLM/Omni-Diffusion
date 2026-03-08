import torch
import torch.nn as nn

import math

class ResamplerProjector(nn.Module):
    """
    A projection module designed to map input features (e.g., from a vision encoder)
    to a target hidden size. It flattens spatial dimensions into a sequence 
    and applies an MLP.
    """
    def __init__(self, proj_input_size, hidden_size):
        super().__init__()

        self.pre_proj_layernorm = torch.nn.LayerNorm(proj_input_size)

        self.mlp = nn.Sequential(
            nn.Linear(proj_input_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.mlp.apply(init_weights)
        self.pre_proj_layernorm.apply(init_weights)

    def forward(self, x, *args, **kwargs):
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.pre_proj_layernorm(x)
        x = self.mlp(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    if isinstance(m, nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

