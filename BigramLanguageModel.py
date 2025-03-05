from torch.nn import Module, Embedding, Linear
from torch.nn.functional import cross_entropy
from torch import arange

class BigramLanguageModel(Module):

    def __init__(self, vocab_size, embedding_size, window_size, batch_size, device):
        super().__init__()
        self.vocab_size, self.embedding_size, self.window_size, self.batch_size, self.device = vocab_size, embedding_size, window_size, batch_size, device
        self.token_embedding = Embedding(vocab_size, embedding_size)
        self.position_embedding = Embedding(window_size, embedding_size)
        self.head = Linear(embedding_size, vocab_size)

    def forward(self, index, targets):
        token_embedding = self.token_embedding(index) # (batch_size, window_size, embedding_size)
        position_embedding = self.position_embedding(arange(self.window_size, device=self.device)) # (window_size, embedding_size)
        representation = self.head(token_embedding + position_embedding) # (batch_size, window_size, vocab_size)

        if not targets: return representation, None

        representation.view(self.batch_size * self.window_size, self.vocab_size)
        targets.view(self.batch_size * self.window_size)
        return representation, cross_entropy(representation, targets)