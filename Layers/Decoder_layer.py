import torch
import torch.nn as nn
from utils import *

class DecoderLayer(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,dropout:float):
        super().__init__()
        self.masked_attn = MultiHead_Attention(d_model,num_heads)
        self.add_norm1 = AddNorm(d_model,dropout)
        self.en_de_attn = MultiHead_Attention(d_model,num_heads)
        self.add_norm2 = AddNorm(d_model,dropout)
        self.ffn = FFN(d_model,d_ff,dropout)
        self.add_norm3 = AddNorm(d_model,dropout)
    
    def forward(self,x:torch.Tensor,encoder_out:torch.Tensor
                ,look_ahead_mask:torch.Tensor,padding_mask:torch.Tensor)->torch.Tensor:
        attn_out = self.masked_attn(q=x,k=x,v=x,mask=look_ahead_mask)
        x=self.add_norm1(x,attn_out)
        enc_dec_attn_out = self.en_de_attn(q=x,k=encoder_out,v=encoder_out,mask=padding_mask)
        x= self.add_norm2(x,enc_dec_attn_out)
        ffn_out = self.ffn(x)
        x=self.add_norm3(x,ffn_out)
        return x
    