from torch import zeros, arange, sin, cos
from torch.nn import Module, Dropout, Embedding

class PositionEmbedding(Module):
    def __init__(self, embedding_size, maxLen, device):
        super().__init__()
        self.encoding = zeros(maxLen, embedding_size, device=device)
        self.encoding.requires_grad_(False)

        position = arange(0, maxLen, device=device).float().reshape(-1, 1)
        even = arange(0, embedding_size, 2, device=device)

        self.encoding[:, 0::2] = sin(position / (1e4 ** (even / embedding_size)))
        self.encoding[:, 1::2] = cos(position / (1e4 ** (even / embedding_size)))

    def forward(self, input):
        return self.encoding[:input[1], :]
    
class TransformerEmbedding(Module):
    def __init__(self, vocab_size, embedding_size, maxLen, device, dropout):
        super().__init__()
        self.tokenEmbedding = Embedding(vocab_size, embedding_size)
        self.positionEmbedding = PositionEmbedding(embedding_size, maxLen, device)
        self.dropout = Dropout(dropout)

    def forward(self, input):
        return self.dropout(self.tokenEmbedding(input) + self.positionEmbedding(input))