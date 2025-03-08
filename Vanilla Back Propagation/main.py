import numpy as np
from mlp import MLP

batch_size = 10
input_feature_size = 3
hidden_feature_size = 5
output_feature_size = 1
learning_rate = 5e-2
num_epoch = 10

x = np.random.randn(batch_size, input_feature_size)
y_true = np.random.randn(batch_size, output_feature_size)
model = MLP(input_feature_size, hidden_feature_size, output_feature_size)
losses = []

for _ in range(num_epoch):
    y = model.forward(x)
    losses.append(np.mean((y - y_true)**2)) # MSE
    grad = 2 * (y - y_true) / y_true.size # 这里除以 10 是为了对应上面 MSE 的定义
    model.backward(grad)
    model.update(learning_rate)

print(losses)