import numpy as np
import copy

class SGD:
    """
    func step()
        - takes gradient descent steps
        - updates weights
    ** In future ** 
    func clear_gradients()
        - makes all previous gradients to zero.
        - useful when RNNs accumulate gradients, or when using mini batch; to remove gradients.

    """
    def __init__(self, lr, params, gradients, ema_momentum = 0.9, momentum = False):
        self.lr = lr
        self.params = params
        self.gradients = gradients
        self.ema_momentum = ema_momentum
        self.momentum = momentum      

    def step(self): 
        n = len(self.gradients)
        num_layers = n // 2

        for i in range(1, num_layers):
            self.params[f'W{i}'] -= self.lr * self.gradients[f'dW{i}']
            self.params[f'b{i}'] -= self.lr * self.gradients[f'db{i}']
        
        return self.params
    
    ## def clear_gradients(self):
        n = len(self.gradients)
        if n == 0:
            pass
        if n > 0:
            for i in range(n):
                self.gradients[f'dW{i}'] = np.zeros(self.gradients[f'dW{i}'])
                self.gradients[f'db{i}'] = np.zeros(self.gradients[f'dW{i}'])