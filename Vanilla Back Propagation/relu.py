import numpy as np

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)
    
    def backward(self, grad):
        grad = grad.copy()
        grad[self.x <= 0] = 0
        return grad