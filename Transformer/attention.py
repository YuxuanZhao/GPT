from torch import matmul
from torch.nn import Module, Linear, Softmax, Dropout
from math import sqrt

class MultiHeadAttention(Module):
    """Multi-head scaled dot-product attention."""
    def __init__(self, embedding_size, num_heads, dropout):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.query_linear = Linear(embedding_size, embedding_size)
        self.key_linear = Linear(embedding_size, embedding_size)
        self.value_linear = Linear(embedding_size, embedding_size)
        self.out_linear = Linear(embedding_size, embedding_size)
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        head_dim = self.embedding_size // self.num_heads
        batch_size = query.size(0)
        
        # Linear projections for query, key, and value
        projected_query = self.query_linear(query)
        projected_key = self.key_linear(key)
        projected_value = self.value_linear(value)
        
        # Reshape and transpose for multi-head attention
        query_heads = projected_query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key_heads = projected_key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value_heads = projected_value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # [Batch, Head, Window, Dimension] * [Batch, Head, Dimension, Window] = [Batch, Head, Window, Window]
        scores = matmul(query_heads, key_heads.transpose(-2, -1)) / sqrt(head_dim) # normalization
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('inf')) # for softmax
        
        attention_weights = self.dropout(self.softmax(scores))
        # [Batch, Head, Window, Window] * [Batch, Head, Window, Dimension] = [Batch, Head, Window, Dimension]
        attention_output = matmul(attention_weights, value_heads)
        
        # Concatenate heads and apply final linear layer [Batch, Window, Embedding]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_size)
        return self.out_linear(attention_output)