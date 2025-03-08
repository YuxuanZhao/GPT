from torch.nn import Module
from encoder import Encoder
from decoder import Decoder
from torch import tril, ones, bool, device, randint

class Transformer(Module):
    """Complete Transformer model integrating the encoder and decoder."""
    def __init__(self, source_padding_index, target_padding_index,
                 encoder_vocab_size, decoder_vocab_size, max_length,
                 embedding_size, hidden_size, num_layers, num_heads,
                 dropout, device):
        super().__init__()
        self.encoder = Encoder(encoder_vocab_size, max_length, embedding_size, hidden_size,
                               num_layers, num_heads, dropout, device)
        self.decoder = Decoder(decoder_vocab_size, max_length, embedding_size, hidden_size,
                               num_layers, num_heads, dropout, device)
        self.source_padding_index = source_padding_index
        self.target_padding_index = target_padding_index
        self.device = device

    def make_pad_mask(self, query, key, query_pad_index, key_pad_index):
        query_len, key_len = query.size(1), key.size(1)
        query_mask = query.eq(query_pad_index).unsqueeze(1).expand(-1, key_len, -1)
        key_mask = key.eq(key_pad_index).unsqueeze(1).expand(-1, query_len, -1)
        combined_mask = (query_mask | key_mask).unsqueeze(1)
        return combined_mask

    def make_causal_mask(self, query, key):
        return tril(ones((query.size(1), key.size(1)), device=self.device, dtype=bool))
    
    def forward(self, source_tokens, target_tokens):
        # Create masks for padding and future tokens
        source_mask = self.make_pad_mask(source_tokens, source_tokens,
                                         self.source_padding_index, self.source_padding_index)
        target_padding_mask = self.make_pad_mask(target_tokens, target_tokens,
                                                 self.target_padding_index, self.target_padding_index)
        causal_mask = self.make_causal_mask(target_tokens, target_tokens)
        target_mask = target_padding_mask * causal_mask
        source_target_mask = self.make_pad_mask(source_tokens, target_tokens,
                                                self.source_padding_index, self.target_padding_index)

        # Encode source and decode target
        encoder_output = self.encoder(source_tokens, source_mask)
        decoder_output = self.decoder(target_tokens, encoder_output, target_mask, source_target_mask)
        return decoder_output