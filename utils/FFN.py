import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.l1=nn.Linear(d_model,d_ff)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.l2=nn.Linear(d_ff,d_model)
        
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.act(self.l1(x))
        x=self.dropout(x)
        return self.l2(x)
    
