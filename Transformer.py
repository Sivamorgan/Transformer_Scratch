import torch
import torch.nn as nn
from Layers import EncoderLayer,DecoderLayer
from utils import Positional_Embedding
import math

class Transformer(nn.Module):
    def __init__(self,num_encoder_layers:int,num_decoder_layers:int,
                 d_model:int,d_ff:int,num_heads:int,input_vocab_size:int,
                 target_vocab_size:int,max_seq_len:int,dropout:float=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size,d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size,d_model)
        self.positional_embedding = Positional_Embedding(d_model,dropout,max_seq_len)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_decoder_layers)])
        self.final_layer = nn.Linear(d_model,target_vocab_size)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def create_padding_mask(self,seq:torch.Tensor,pad_token_idx:int=0)->torch.Tensor:
        mask= (seq!=pad_token_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_look_ahead_mask(self,size:int)->torch.Tensor:
        mask = torch.triu(torch.ones(size,size),diagonal=1).bool()
        return ~mask.unsqueeze(0).unsqueeze(0)
    
    def encode(self,src:torch.Tensor,src_mask:torch.Tensor)->torch.Tensor:
        src_embd = self.encoder_embedding(src)*math.sqrt(self.d_model)      
        src_pos_emb = self.positional_embedding(src_embd)
        enc_output = self.dropout(src_pos_emb)
        for layer in self.encoder_layers:
            enc_output=layer(enc_output,src_mask)
        return enc_output
    
    def decode(self,tgt:torch.Tensor,encoder_output:torch.Tensor,
              look_ahead_mask:torch.Tensor,padding_mask:torch.Tensor)->torch.Tensor:
        tgt_embd = self.decoder_embedding(tgt)* math.sqrt(self.d_model)
        tgt_pos_emd = self.positional_embedding(tgt_embd)
        dec_output = self.dropout(tgt_pos_emd)
        for layer in self.decoder_layers:
            dec_output=layer(dec_output,encoder_output,look_ahead_mask,padding_mask)
        return dec_output
    
    def forward(self,src:torch.Tensor,tgt:torch.Tensor)->torch.Tensor:
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)
        look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)
        combined_look_ahead_mask = torch.logical_and(tgt_padding_mask.transpose(-2,-1),look_ahead_mask)
        encoder_out = self.encode(src,src_padding_mask)
        decoder_out = self.decode(tgt,encoder_out,combined_look_ahead_mask,src_padding_mask)
        output = self.final_layer(decoder_out)
        return output



if __name__=="__main__":
    transformer_model = Transformer(
    num_encoder_layers=6, num_decoder_layers=6,
    d_model=512, num_heads=8, d_ff=2048,
    input_vocab_size=10000, target_vocab_size=12000,
    max_seq_len=500, dropout=0.1
)

    # Dummy input for shape check (assuming batch_size=2)
    src_dummy = torch.randint(1, 10000, (2, 100)) # (batch, src_len)
    tgt_dummy = torch.randint(1, 12000, (2, 120))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model.to(device)
    src_dummy = src_dummy.to(device)
    tgt_dummy = tgt_dummy.to(device)

    output_logits = transformer_model(src_dummy, tgt_dummy)
    print("Final output shape (logits):", output_logits.shape)
            
        