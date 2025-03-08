from torch import matmul
from torch.nn import Module, Linear, Softmax, Dropout
from math import sqrt

class MultiHeadAttention(Module):
    def __init__(self, batch_size, embedding_size, num_head, dropout):
        super().__init__()
        self.embedding_size, self.num_head, self.batch_size = embedding_size, num_head, batch_size
        self.q = Linear(embedding_size, embedding_size)
        self.k = Linear(embedding_size, embedding_size)
        self.v = Linear(embedding_size, embedding_size)
        self.output = Linear(embedding_size, embedding_size)
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = self.embedding_size//self.num_head
        q, k, v = self.q(q), self.k(k), self.v(v)
        q = q.view(self.batch_size, -1, self.num_head, d_k).transpose(1, 2)
        k = k.view(self.batch_size, -1, self.num_head, d_k).transpose(1, 2)
        v = v.view(self.batch_size, -1, self.num_head, d_k).transpose(1, 2)

        # [Batch, Head, Window, Dimension] * [Batch, Head, Dimension, Window] = [Batch, Head, Window, Window]
        scores = matmul(q, k.transpose(-2, -1)) / sqrt(d_k) # normalization

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('inf')) # for softmax

        attention = self.dropout(self.softmax(scores))

        # [Batch, Head, Window, Window] * [Batch, Head, Window, Dimension] = [Batch, Head, Window, Dimension]
        x = matmul(attention, v)
        # [Batch, Window, Embedding]
        x = x.transpose(1, 2).contiguous().view(self.batch_size, -1, self.embedding_size)
        return self.output(x)