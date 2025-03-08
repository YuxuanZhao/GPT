from torch.nn import Module, Dropout, ModuleList
from attention import MultiHeadAttention
from utils import LayerNorm, PositionwiseFeedForward
from embedding import TransformerEmbedding

class EncoderLayer(Module):
    """A single encoder layer composed of multi-head self-attention and feed-forward network with residual connections."""
    def __init__(self, num_heads, embedding_size, hidden_size, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_size, num_heads, dropout)
        self.norm_after_attention = LayerNorm(embedding_size)
        self.attention_dropout = Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(embedding_size, hidden_size, dropout)
        self.norm_after_ffn = LayerNorm(embedding_size)
        self.ffn_dropout = Dropout(dropout)

    def forward(self, input_tensor, mask=None):
        # Self-attention sublayer with residual connection
        residual = input_tensor
        attention_output = self.self_attention(input_tensor, input_tensor, input_tensor, mask)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.norm_after_attention(residual + attention_output)

        # Feed-forward sublayer with residual connection
        residual = attention_output
        ffn_output = self.feed_forward(attention_output)
        ffn_output = self.ffn_dropout(ffn_output)
        output = self.norm_after_ffn(residual + ffn_output)
        return output
    
class Encoder(Module):
    def __init__(self, vocab_size, maxLen, embedding_size, hidden_size, num_layers, num_heads, dropout, device):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, embedding_size, maxLen, device, dropout)
        self.layers = ModuleList(
            [EncoderLayer(num_heads, embedding_size, hidden_size, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x