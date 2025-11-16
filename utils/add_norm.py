import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self,normalized_shape:int,dropout:float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x:torch.Tensor,sublayer_output:torch.Tensor)->torch.Tensor:
        return self.layer_norm(x+self.dropout(sublayer_output))

