from relu import ReLU
from linear import Linear

class MLP:
    def __init__(self, inFeature, hiddenFeature, outFeature):
        self.l1 = Linear(inFeature, hiddenFeature)
        self.r1 = ReLU()
        self.l2 = Linear(hiddenFeature, outFeature)

    def forward(self, x):
        return self.l2.forward(self.r1.forward(self.l1.forward(x)))
    
    def backward(self, grad):
        return self.l1.backward(self.r1.backward(self.l2.backward(grad)))
    
    def update(self, learning_rate):
        self.l1.update(learning_rate)
        self.l2.update(learning_rate)