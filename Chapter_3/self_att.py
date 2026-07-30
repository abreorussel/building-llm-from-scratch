import torch


# Consider the following input sentence, which has already been embedded into
# three-dimensional vectors. 

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your (x^1)
   [0.55, 0.87, 0.66], # journey (x^2)
   [0.57, 0.85, 0.64], # starts (x^3)
   [0.22, 0.58, 0.33], # with    
   [0.77, 0.25, 0.10], # one     
   [0.05, 0.80, 0.55]] # step    
)



def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

    

# Computing the attention weights with the second input token as the query
# Dot product of x^2 with every other input token
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())


# Computing the context vector
query = inputs[1]        
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)




# Now computing all the attention weights
# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i][j] = torch.dot(x_i, x_j)

# print(attn_scores)


# However for loops are generally slow . We could do input * input.T
attn_scores = inputs @ inputs.T
print(attn_scores)

print(attn_scores.shape)
# Normalization
# Done across the columsn , so tht each row sums up to 1
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# Can verify each row sums to 1
print(attn_weights.sum(dim=-1))


# now that we have the attention weights lets create the context vectors

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)   # Each row is a 3 dim context vec




