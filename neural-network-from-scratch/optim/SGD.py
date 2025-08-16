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

    if momentum == True:
        Vtheta = B*Vt-1 + (1-B)*Vt

    """
    def __init__(self, lr, params, gradients, ema_momentum = 0.9, momentum = True):
        self.lr = lr
        self.params = params
        self.gradients = gradients
        self.ema_momentum = ema_momentum
        self.momentum = momentum
        self.VdW = {}
        self.Vdb = {}      

    def step(self): 
        n = len(self.gradients)
        num_layers = n // 2

        if self.momentum == True:
            
            # Initialize Velocity and history
            for i in range(1, num_layers):    
                self.VdW[f'W{i}'] = np.zeros_like(self.params[f'W{i}'])
                self.Vdb[f'b{i}'] = np.zeros_like(self.params[f'b{i}'])

            for i in range(1, num_layers):
                
                # Applying Velocity for momentum
                self.VdW[f'W{i}'] = self.ema_momentum * self.VdW[f'W{i}'] + (1 - self.ema_momentum) * self.gradients[f'dW{i}']
                self.Vdb[f'b{i}'] = self.ema_momentum * self.Vdb[f'b{i}'] + (1 - self.ema_momentum) * self.gradients[f'db{i}']

                # Update rule
                self.params[f'W{i}'] -= self.lr * self.VdW[f'W{i}']
                self.params[f'b{i}'] -= self.lr * self.Vdb[f'b{i}']

        else:      

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