import torch
import torch.nn as nn
from helper import scaled_dot_product_attention


class MultiHead_Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model%num_heads==0,"d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        
        #Linear projections for all heads
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        
        #Final Projection layer after Concatenation
        
        self.W_o = nn.Linear(d_model,d_model)
        
    def split_heads(self,x:torch.Tensor)->torch.Tensor:
        batch_size,seq_len,_ = x.size()
        x=x.view(batch_size,seq_len,self.num_heads,self.d_k)
        return x.transpose(1,2)
    
    def combine_heads(self,x:torch.Tensor)->torch.Tensor:
        batch_size,_,seq_len,_ = x.size()
        x=x.transpose(1,2).contiguous()
        return x.view(batch_size,seq_len,self.d_model)
        
        
    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor=None)->torch.Tensor:
        
        #q,k,v -> [batch_size,seq_len,d_model]
        #mask  -> [batch_size,1,seq_len,seq_len]
        
        q= self.W_q(q)  #[batch_size,seq_len,d_model]
        k=self.W_k(k)   #[batch_size,seq_len,d_model]
        v=self.W_v(v)   #[batch_size,seq_len,d_model]
        q=self.split_heads(q)   #[batch_size,num_heads,seq_len,d_k]
        k=self.split_heads(k)   #[batch_size,num_heads,seq_len,d_k]
        v=self.split_heads(v)   #[batch_size,num_heads,seq_len,d_k]
        
        attention_scores,_ = scaled_dot_product_attention(q,k,v,mask)
        #attention_scores ->[batch_size,num_heads,seq_len,d_k]
        #attention_weights->[batch_size,num_heads,seq_len,d_k]
        
        output = self.combine_heads(attention_scores) #[batch_size,seq_len,d_model]
        
        output = self.W_o(output)       #[batch_size,seq_len,d_model]
        return output
        
        
        
        
        
        
