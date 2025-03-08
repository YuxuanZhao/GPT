from torch import device, randint
from transformer import Transformer

source_padding_index = 0
target_padding_index = 0
encoder_vocab_size = 100   # Example vocab size for encoder
decoder_vocab_size = 100   # Example vocab size for decoder
max_length = 10            # Maximum sequence length
embedding_size = 64
hidden_size = 128
num_layers = 2
num_heads = 8
dropout = 0.1
device = device('cuda')
batch_size = 4

# Instantiate the transformer model
model = Transformer(source_padding_index, target_padding_index,
                    encoder_vocab_size, decoder_vocab_size,
                    max_length, embedding_size, hidden_size,
                    num_layers, num_heads, dropout, device)
model.to(device)

# Create sample input sequences: generate random tokens (avoid padding index 0)
source_tokens = randint(1, encoder_vocab_size, (batch_size, max_length)).to(device)
target_tokens = randint(1, decoder_vocab_size, (batch_size, max_length)).to(device)

# Simulate padding for the first batch in the last two positions
source_tokens[0, -2:] = source_padding_index  
target_tokens[0, -2:] = target_padding_index  

# Run forward pass through the model
output = model(source_tokens, target_tokens)
print("Output shape:", output.shape)