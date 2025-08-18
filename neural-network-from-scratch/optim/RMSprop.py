import numpy as np
import copy

class RMSprop:
    """
    epsilon is used for numerical stability
    
    func step()
        - takes gradient descent steps
        - updates weights

    RMSprop uses EWMA momentum to give more loss moving in an horizontal direction and less importance to loss moving in a vertical direction
    allowing for greater learning rates, and accelerated learning of the loss function.

        Vtheta = B_2 * Vt-1 + (1-B_2) * Vt

    """
    def __init__(self, lr, params, gradients, ema_momentum = 0.9, momentum = False):
        self.lr = lr
        self.params = params
        self.gradients = gradients
        self.ema_momentum = ema_momentum
        self.momentum = momentum
        self.SdW = {}
        self.Sdb = {}      

    def step(self): 

        n = len(self.gradients)
        num_layers = n // 2
        epsilon = 1e-8
        # Initialize Velocity and history
        for i in range(1, num_layers):    
            self.SdW[f'W{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.Sdb[f'b{i}'] = np.zeros_like(self.params[f'b{i}'])

        for i in range(1, num_layers):
            # Applying Velocity for momentum
            self.SdW[f'W{i}'] = self.ema_momentum_beta2 * self.SdW[f'W{i}'] + (1 - self.ema_momentum_beta2) * (self.gradients[f'dW{i}'])**2
            self.Sdb[f'b{i}'] = self.ema_momentum_beta2 * self.Sdb[f'b{i}'] + (1 - self.ema_momentum_beta2) * (self.gradients[f'db{i}'])**2
            
            # Update rule
            self.params[f'W{i}'] -= self.lr * self.gradients[f'W{i}'] / np.sqrt(self.Sdw[f'W{i}'] + epsilon)
            self.params[f'b{i}'] -= self.lr * self.gradients[f'b{i}'] / np.sqrt(self.Sdb[f'b{i}'] + epsilon)
        
        return self.params
