import numpy as np

class Linear:
    def __init__(self, inFeature, outFeature):
        self.W = np.random.randn(inFeature, outFeature) * 0.01 # [in, out]
        self.b = np.zeros((1, outFeature)) # [out]
        self.gW = np.zeros_like(self.W) # [in, out]
        self.gb = np.zeros_like(self.b) # [out]
        self.x = None

    def forward(self, x):
        self.x = x # [batch, in]
        return x.dot(self.W) + self.b # [batch, in].[in, out] + [out] (broadcast)
    
    def backward(self, grad): # [batch, out]
        # gL/gW = gL/gy * gy/gW = grad * x
        self.gW = self.x.T.dot(grad) # [in, batch].[batch, out]
        # gL/gb = gL/gy * gy/gb = grad * 1
        self.gb = np.sum(grad, axis=0, keepdims=True) # [sum(batch), out] => [out]
        # gL/gy1 = gL/gy2 * gy2/gy1 = grad * W
        return grad.dot(self.W.T) # [batch, out].[out, in]
    
    def update(self, learning_rate):
        self.W -= learning_rate * self.gW
        self.b -= learning_rate * self.gb