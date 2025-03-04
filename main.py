import BigramLanguageModel
import Train
import Test
from torch.optim import AdamW

model = BigramLanguageModel()
model.to('cuda')
optimizer = AdamW(model.parameters(), lr=3e-4)

Train()
Test()