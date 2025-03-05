import torch
import torch.nn as nn

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, d_in, d_out,
        context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #A
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) #B

    def qkv_attention(self, queries, keys, values):
        raise NotImplementedError()
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x) #C
        queries = self.W_query(x) #C
        values = self.W_value(x) #C
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) #D
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) #D
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)#D
        context_vec = self.qkv_attention(queries, keys, values)
        #J
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) #K
        return context_vec
        

class MultiHeadAttention(MultiHeadAttentionBase):
    def __init__(self, d_in, d_out,
        context_length, dropout, num_heads, qkv_bias=False):
        super().__init__(d_in, d_out, context_length, dropout, num_heads, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
        'mask',
        torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def qkv_attention(self, queries, keys, values):
        b, num_tokens, _, _ = queries.shape
        keys = keys.transpose(1, 2) #E
        queries = queries.transpose(1, 2) #E
        values = values.transpose(1, 2) #E
        attn_scores = queries @ keys.transpose(2, 3) #F
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] #G
        attn_scores.masked_fill_(mask_bool, -torch.inf) #H
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2) #I

        return context_vec
    
class MultiHeadFlashAttention(MultiHeadAttentionBase):
    def __init__(self, d_in, d_out,
        context_length, dropout, num_heads, qkv_bias=False):
        super().__init__(d_in, d_out, context_length, dropout, num_heads, qkv_bias)
        self.dropout = dropout
    
    def qkv_attention(self, queries, keys, values):
        return flash_attn_func(queries, keys, values, self.dropout if self.training else 0.0, causal=True)
    
    #A Reduce the projection dim to match desired output dim
    #B Use a Linear layer to combine head outputs
    #C Tensor shape: (b, num_tokens, d_out)
    #D We implicitly split the matrix by adding a `num_heads` dimension. Then we unroll last dim: (b,
    # num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
    #E Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
    #F Compute dot product for each head
    #G Mask truncated to the number of tokens
    #H Use the mask to fill attention scores
    #I Tensor shape: (b, num_tokens, n_heads, head_dim)
    #J Combine heads, where self.d_out = self.num_heads * self.head_dim
    #K Add an optional linear projection

def check_shape(a, b):
    assert a.shape == b.shape

def assign_layer_weights(layer, W_key, W_query, W_value):
    check_shape(layer.W_key.weight, W_key)
    layer.W_key.weight = nn.Parameter(W_key.clone())
    check_shape(layer.W_query.weight, W_query)
    layer.W_query.weight = nn.Parameter(W_query.clone())
    check_shape(layer.W_value.weight, W_value)
    layer.W_value.weight = nn.Parameter(W_value.clone())

if __name__ == '__main__':
    (d_in, d_out,
    context_length, dropout, num_heads, qkv_bias)=(64, 64, 128, 0.2, 8, False)

    attn_layer = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads, qkv_bias)
    flash_attn_layer = MultiHeadFlashAttention(d_in, d_out, context_length, dropout, num_heads, qkv_bias)

    W_key = torch.randn(d_in, d_out)
    W_query = torch.randn(d_in, d_out)
    W_value = torch.randn(d_in, d_out)

    assign_layer_weights(attn_layer, W_key, W_query, W_value)
    assign_layer_weights(flash_attn_layer, W_key, W_query, W_value)

    device = torch.device("cuda")
    batch_size, num_tokens = 4, 16

    X = torch.randn(batch_size, num_tokens, d_in)

    X = X.to(device)

    attn_layer = attn_layer.to(device)
    flash_attn_layer = flash_attn_layer.to(device)

    attn_layer.eval()
    flash_attn_layer.eval()

    Y1 = attn_layer(X)
    Y2 = flash_attn_layer(X)

    assert Y1.detach()==Y2.detach()