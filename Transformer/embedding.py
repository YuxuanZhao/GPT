from torch import zeros, arange, sin, cos
from torch.nn import Module, Dropout, Embedding, Sequential

class PositionEmbedding(Module):
    """Generates positional embeddings using sine and cosine functions."""
    def __init__(self, embedding_size, max_length, device):
        super().__init__()
        self.encoding = zeros(max_length, embedding_size, device=device)
        self.encoding.requires_grad_(False)

        position = arange(0, max_length, device=device).float().unsqueeze(1)
        even_indices = arange(0, embedding_size, 2, device=device)

        self.encoding[:, 0::2] = sin(position / (1e4 ** (even_indices / embedding_size)))
        self.encoding[:, 1::2] = cos(position / (1e4 ** (even_indices / embedding_size)))

    def forward(self, token_indices):
        return self.encoding[:token_indices.size(1), :]
    
class TransformerEmbedding(Module):
    """Combines token and positional embeddings with dropout."""
    def __init__(self, vocab_size, embedding_size, max_length, device, dropout):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, embedding_size)
        self.position_embedding = PositionEmbedding(embedding_size, max_length, device)
        self.dropout = Dropout(dropout)

    def forward(self, token_indices):
        token_embeds = self.token_embedding(token_indices)
        pos_embeds = self.position_embedding(token_indices)
        embeddings = token_embeds + pos_embeds
        return self.dropout(embeddings)