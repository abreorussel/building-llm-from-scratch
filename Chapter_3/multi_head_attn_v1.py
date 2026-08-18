import torch
import torch.nn as nn
from causal_attn import CausalAttention

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your (x^1)
   [0.55, 0.87, 0.66], # journey (x^2)
   [0.57, 0.85, 0.64], # starts (x^3)
   [0.22, 0.58, 0.33], # with    
   [0.77, 0.25, 0.10], # one     
   [0.05, 0.80, 0.55]] # step    
)

d_in = inputs.shape[1]   # input_embedding dimension
d_out = 2  # output embedding dimension

print(f"Inputs Shape : {inputs.shape}\n")
print("Stack the inputs twice to simulate two batches\n")

batch = torch.stack((inputs, inputs), dim=0)
print(f"Batch shape : {batch.shape}")



class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)





"""
If you put PyTorch modules inside a standard Python list, PyTorch will not "see" them. It won't register their weights (like W_query, W_key) as parameters belonging to the model. By wrapping the list in nn.ModuleList, you tell PyTorch: "Hey, track all the weights inside these modules so the optimizer can update them, and move them to the GPU when I ask."
"""



torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, 0.0, num_heads=2
        )
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)