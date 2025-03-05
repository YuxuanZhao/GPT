with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
s2i = {ch:i for i,ch in enumerate(chars)}
i2s = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [s2i[c] for c in s]
decode = lambda l: ''.join([i2s[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train = data[:n]
test = data[n:]

def get_batch(isTrain):
    data = train if isTrain == 'train' else test
    randomIndex = torch.randint(len(data) - block_size, (batch_size, )) # block size 4, batch size 2: 
    x = torch.stack([data[i: i + block_size] for i in randomIndex]) # [[55, 36, 84, 92], [63, 45, 13, 62]]
    y = torch.stack([data[i + 1: i + block_size + 1] for i in randomIndex]) # [[36, 84, 92, 10], [45, 13, 62, 33]]
    x, y = x.to(device), y.to(device)
    return x, y