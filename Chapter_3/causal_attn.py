import torch
import torch.nn as nn

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


class CausalAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
    super().__init__()
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.d_out = d_out
    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
      'mask',
      torch.triu(torch.ones(context_length, context_length), diagonal=1)
      # creates an upper-triangular matrix where the values above the main diagonal are 1, and everything else is 0.
    )

  def forward(self, x):
    b, num_tokens, d_in = x.shape # (2, 6, 3)
    queries = self.W_query(x)
    values = self.W_value(x)
    keys = self.W_key(x)

    attn_scores = queries @ keys.transpose(1, 2)
    attn_scores.masked_fill_(
      self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
    )
    attn_weights = torch.softmax(
      attn_scores / (keys.shape[-1] ** 0.5), dim=-1
    )
    attn_weights = self.dropout(attn_weights)

    context_vec = attn_weights @ values
    return context_vec


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)



"""
  In PyTorch, register_buffer is a method used to add a tensor to your model's state without treating it as a learnable parameter.
  When you create a custom neural network class (like the CausalAttention module, you generally have two types of tensors:
    - Parameters: Weights and biases that the optimizer updates during training using gradients (e.g., self.W_query, self.W_key).
    - Buffers: Tensors that are essential to the model's computation but should not be updated by gradient descent (e.g., the attention mask).

  This is the most common reason to use it. When you move your model to a GPU using model.to('cuda'), PyTorch will automatically move all registered buffers to the GPU alongside the learnable parameters.
  Registered buffers are automatically included in the model's state_dict(). This means when you save your model to disk and load it later, the buffer's exact state is preserved and restored.
"""
