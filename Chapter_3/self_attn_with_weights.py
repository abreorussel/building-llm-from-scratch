import torch



inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your (x^1)
   [0.55, 0.87, 0.66], # journey (x^2)
   [0.57, 0.85, 0.64], # starts (x^3)
   [0.22, 0.58, 0.33], # with    
   [0.77, 0.25, 0.10], # one     
   [0.05, 0.80, 0.55]] # step    
)

# First we only consider x_2 as the query vector 
x_2 = inputs[1]
d_in = inputs.shape[1]   # input_embedding dimension
d_out = 2  # output embedding dimension

print(f"x_2 shape : {x_2.shape}")


torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(f"W_query shape : {W_query.shape}")

# now compute the query, key and value vectors.
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(f"query_2 :{query_2}")


# Now lets compute all the keys and values for all the inputs
keys = inputs @ W_key
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# Now the next step is to compute the attention score
# The unscaled attention score is calculated as the dot product between the query and the key vectors

# NOw lets compute teh attention score w22

keys_2 = keys[1]
attn_score_22  = query_2.dot(keys_2)
print(attn_score_22)

# Compute all the attention scores for query 2
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2.shape)


# Now lets compute the attention weight by scaling the attention scores using the softmax function
d_k = keys.shape[-1]  # getting the key embedding dimension
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**2, dim = -1)
print(attn_weights_2)


# NOw we mulitply the value vector with the attention weight and then summing them up to get the context vector
context_vector_2 = attn_weights_2 @ values
print(context_vector_2)