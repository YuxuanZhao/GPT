from torch import ones, zeros, randn, tril, sqrt
from torch.nn import Module, Parameter, Linear, Dropout, ReLU
from attention import MultiHeadAttention

# batchNorm 是对 batch 这个维度进行 normalize（不同的样本在同一个 channel 上分布应当均匀）
# LayerNorm 是对 channel 这个维度进行 normalize（同一个样本在不同的 channel 上分布应当均匀）
class LayerNorm(Module):
    def __init__(self, embedding_size, epsilon=1e-10):
        super().__init__()
        self.gamma = Parameter(ones(embedding_size))
        self.beta = Parameter(zeros(embedding_size))
        self.epsilon = epsilon

    def forward(self, input):
        mean = input.mean(-1, keepdim=True)
        var = input.var(-1, unbiased=False, keepdim=True)
        res = (input - mean) / sqrt(var + self.epsilon)
        return res * self.gamma + self.beta
    
class PositionwiseFeedForward(Module):
    def __init__(self, embedding_size, hidden_size, dropout=0.1):
        super().__init__()
        self.l1 = Linear(embedding_size, hidden_size)
        self.l2 = Linear(hidden_size, embedding_size)
        self.a1 = ReLU()
        self.d = Dropout(dropout)

    def forward(self, input):
        return self.l2(self.d(self.a1(self.l1(input))))

class Encoder(Module):
    def __init__(self, num_head, embedding_size, hidden_size, dropout, batch_size):
        super().__init__()
        self.attention = MultiHeadAttention(batch_size, embedding_size, num_head, dropout)
        self.norm1 = LayerNorm(embedding_size)
        self.drop1 = Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(embedding_size, hidden_size, dropout)
        self.norm2 = LayerNorm(embedding_size)
        self.drop2 = Dropout(dropout)

    def forward(self, input, mask=None):
        input_copy = input
        input = self.attention(input, input, input, mask)
        input = self.drop1(input)
        input = self.norm1(input + input_copy)
        input_copy = input
        input = self.feed_forward(input)
        input = self.drop2(input)
        input = self.norm2(input + input_copy)
        return input
    
batch_size = 2        # Number of sequences in a batch.
window_size = 5        # Length of each sequence.
embedding_size = 8    # Dimensionality of input embeddings.
hidden_size = 16      # Hidden layer size in feed-forward network.
num_head = 2          # Number of attention heads.
dropout = 0.1         # Dropout probability.

# Instantiate the Encoder.
encoder = Encoder(num_head, embedding_size, hidden_size, dropout, batch_size)

# Create a random input tensor (shape: batch_size x seq_length x embedding_size).
input_tensor = randn(batch_size, window_size, embedding_size)

# Optionally, you can define a mask (here we use None for simplicity).
mask = tril(ones(window_size, window_size, dtype=bool))

# Pass the input tensor through the Encoder.
output = encoder(input_tensor, mask)

# Print the output.
print("Encoder output:")
print(output)