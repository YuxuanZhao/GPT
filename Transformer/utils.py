from torch.nn import Module, Parameter, Linear, Dropout, ReLU, Sequential
from torch import ones, zeros, sqrt

# batchNorm 是对 batch 这个维度进行 normalize（不同的样本在同一个 channel 上分布应当均匀）
# LayerNorm 是对 channel 这个维度进行 normalize（同一个样本在不同的 channel 上分布应当均匀）对于语言输入的好处在于每个样本的长度会有差异，所以对不同样本取 variance 会有更大的波动
class LayerNorm(Module):
    """Layer normalization over the channel dimension."""
    def __init__(self, embedding_size, epsilon=1e-10):
        super().__init__()
        self.gamma = Parameter(ones(embedding_size))
        self.beta = Parameter(zeros(embedding_size))
        self.epsilon = epsilon

    def forward(self, input_tensor):
        mean = input_tensor.mean(-1, keepdim=True)
        variance = input_tensor.var(-1, unbiased=False, keepdim=True)
        normalized_tensor = (input_tensor - mean) / sqrt(variance + self.epsilon)
        return normalized_tensor * self.gamma + self.beta


class PositionwiseFeedForward(Module):
    """Applies a two-layer feed-forward network to each position."""
    def __init__(self, embedding_size, hidden_size, dropout=0.1):
        super().__init__()
        self.model = Sequential(
            Linear(embedding_size, hidden_size),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_size, embedding_size)
        )

    def forward(self, input_tensor):
        return self.model(input_tensor)