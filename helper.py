import torch
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    """Generates attention scores and returns the Attention Scores and Attention Weights
    Inputs: q-Query (Tensor) , k-Key(Tensor), v-Value(Tensor),mask(Tensor)
    Outputs: Attention Scores(Tensor), Attention_weights(Tensor)
    """
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores.masked_fill(mask, -1e9)

    attn_weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, v)
    return output, attn_weights


def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    for k in range(max_len):
        for i in range(d_model / 2):
            pe[k, 2 * i] = torch.sin(k / (torch.pow(10000, 2 * i / d_model)))
            if 2 * i + 1 < d_model:
                pe[k, 2 * i + 1] = torch.cos(k / (torch.pow(10000, 2 * i / d_model)))
    return pe
