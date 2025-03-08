from torch.nn import Module, Dropout, ModuleList, Linear
from attention import MultiHeadAttention
from utils import LayerNorm, PositionwiseFeedForward
from embedding import TransformerEmbedding
    
class DecoderLayer(Module):
    """A single decoder layer with self-attention, encoder-decoder attention, and a feed-forward network."""
    def __init__(self, embedding_size, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_size, num_heads, dropout)
        self.norm_after_self_attn = LayerNorm(embedding_size)
        self.self_attn_dropout = Dropout(dropout)

        self.encoder_decoder_attention = MultiHeadAttention(embedding_size, num_heads, dropout)
        self.norm_after_enc_dec_attn = LayerNorm(embedding_size)
        self.enc_dec_attn_dropout = Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(embedding_size, hidden_size, dropout)
        self.norm_after_ffn = LayerNorm(embedding_size)
        self.ffn_dropout = Dropout(dropout)

    def forward(self, decoder_tokens, encoder_output, target_mask, source_target_mask):
        # Self-attention for decoder tokens
        residual = decoder_tokens
        # tril mask，不希望在预测的时候能看到未来的信息
        self_attn_output = self.self_attention(decoder_tokens, decoder_tokens, decoder_tokens, target_mask)
        self_attn_output = self.norm_after_self_attn(residual + self_attn_output)
        self_attn_output = self.self_attn_dropout(self_attn_output)

        # Encoder-decoder attention sublayer
        if encoder_output is not None:
            residual = self_attn_output
             # 训练的时候可能会有 padding，不需要关注到这些信息
            enc_dec_attn_output = self.encoder_decoder_attention(self_attn_output, encoder_output, encoder_output, source_target_mask)
            enc_dec_attn_output = self.norm_after_enc_dec_attn(residual + enc_dec_attn_output)
            enc_dec_attn_output = self.enc_dec_attn_dropout(enc_dec_attn_output)
            x = enc_dec_attn_output
        else:
            x = self_attn_output

        # Feed-forward network sublayer
        residual = x
        ffn_output = self.feed_forward(x)
        ffn_output = self.norm_after_ffn(residual + ffn_output)
        output = self.ffn_dropout(ffn_output)
        return output


class Decoder(Module):
    """Transformer decoder consisting of an embedding layer and multiple decoder layers, with final linear projection."""
    def __init__(self, vocab_size, max_length, embedding_size, hidden_size, num_layers, num_heads, dropout, device):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, embedding_size, max_length, device, dropout)
        self.layers = ModuleList([
            DecoderLayer(embedding_size, hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.fc = Linear(embedding_size, vocab_size)

    def forward(self, decoder_tokens, encoder_output, target_mask, source_target_mask):
        embedded_decoder = self.embedding(decoder_tokens)
        x = embedded_decoder
        for layer in self.layers:
            x = layer(x, encoder_output, target_mask, source_target_mask)
        return self.fc(x)