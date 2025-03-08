import BigramLanguageModel
import Train
import Test
from torch.optim import AdamW
from Data import Preprocess, DatasetLite
from torch.utils.data import DataLoader

device = 'cuda'
window_size = 256
batch_size = 64

p = Preprocess()
vocab_size, train, test = p.process('input.txt')
train_dataset = DatasetLite(device, window_size, train)
test_dataset = DatasetLite(device, window_size, test)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)


model = BigramLanguageModel(vocab_size=100, embedding_size=384, window_size=window_size, batch_size=batch_size, device=device)
model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-4)

Train()
Test()