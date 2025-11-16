import torch
import torch.nn as nn
from utils import *

class EncoderLayer(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,dropout:float):
        super().__init__()
        self.self_attn = MultiHead_Attention(d_model,num_heads)
        self.add_norm1 = AddNorm(d_model,dropout)
        self.ffn = FFN(d_model,d_ff,dropout)
        self.add_norm2 = AddNorm(d_model,dropout)
        pass
    
    def forward(self,x:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
        attn_output = self.self_attn(q=x,k=x,v=x,mask=mask)
        x=self.add_norm1(x,attn_output)
        ffn_output = self.ffn(x)
        x=self.add_norm2(x,ffn_output)
        return x
