import BigramLanguageModel
import Train
import Test
from torch.optim import AdamW

model = BigramLanguageModel(vocab_size=100, embedding_size=384, window_size=256, batch_size=64, device='cuda')
model.to('cuda')
optimizer = AdamW(model.parameters(), lr=3e-4)

Train()
Test()