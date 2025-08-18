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
    def __init__(self, lr, params, gradients, epsilon = 1e-8, ema_momentum_beta2 = 0.9):
        self.lr = lr
        self.params = params
        self.gradients = gradients
        self.ema_momentum_beta2 = ema_momentum_beta2
        self.SdW = {}
        self.Sdb = {}      
        self.epsilon = epsilon

        # Initialize Velocity and history
        for i in range(1, len(self.gradients) // 2):    
            self.SdW[f'W{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.Sdb[f'b{i}'] = np.zeros_like(self.params[f'b{i}'])
    
    def step(self):

        for i in range(1, len(self.gradients) // 2):
            # Applying Velocity for momentum
            self.SdW[f'W{i}'] = self.ema_momentum_beta2 * self.SdW[f'W{i}'] + (1 - self.ema_momentum_beta2) * (self.gradients[f'dW{i}'])**2
            self.Sdb[f'b{i}'] = self.ema_momentum_beta2 * self.Sdb[f'b{i}'] + (1 - self.ema_momentum_beta2) * (self.gradients[f'db{i}'])**2
            
            # Update rule
            self.params[f'W{i}'] -= self.lr * self.gradients[f'dW{i}'] / np.sqrt(self.SdW[f'W{i}'] + self.epsilon)
            self.params[f'b{i}'] -= self.lr * self.gradients[f'db{i}'] / np.sqrt(self.Sdb[f'b{i}'] + self.epsilon)
        
        return self.params
