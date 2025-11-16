import torch
import torch.nn as nn
import math



class Positional_Embedding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        pe = torch.zeros(max_len, d_model)

        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer(
            "pe", pe
        )  # As PE is not a parameters(non-trainable) we use buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

        x - Tensor of Shape [batch,seq_len,d_model]
        Returns
        Tensor of shape [batch,seq_len,d_model]

        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
