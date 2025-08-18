import numpy as np
import copy

class Adam:
    """
    epsilon for numerical stability 

    func step()
        - takes gradient descent steps
        - updates weights
    

    first moment : 
        VdW = ema_momentum * VdW + (1 - ema_momentum) * VdW
        Vdb = ema_momentum * Vdb + (1 - ema_momentum) * Vdb
        
        # Bias Correction
        VdW_corrected = VdW / (1 - ema_momentum ** t)
        Vdb_corrected = Vdb / (1 - ema_momentum ** t)
    
    second moment :
        SdW = ema_momentum_beta2 * SdW + (1 - ema_momentum_beta2) * dW ** 2
        Sdb = ema_momentum_beta2 * Sdb + (1 - ema_momentum_beta2) * db ** 2

        # Bias Correction
        SdW_corrected = SdW / (1 - ema_momentum_beta2 ** t)
        Sdb_corrected = Sdb / (1 - ema_momentum_beta2 ** t)
    
    update rule:
        W := W - lr * VdW / np.sqrt(SdW + epsilon)
        b := b - lr * Vdb / np.sqrt(Sdb + epsilon)

    """
    def __init__(self, lr, params, gradients, ema_momentum = 0.9, ema_momentum_beta2 = 0.99, epsilon = 1e-8):
        self.lr = lr
        self.params = params
        self.gradients = gradients
        self.ema_momentum = ema_momentum
        self.ema_momentum_beta2 = ema_momentum_beta2
        self.VdW = {}
        self.Vdb = {}
        self.SdW = {}
        self.Sdb = {}      
        self.epsilon = epsilon
        
    def step(self): 
        n = len(self.gradients)
        num_layers = n // 2
        epsilon = 1e-8
        
        # Initialize Velocity and history - Momentum
        for t in range(1, num_layers):    
            self.VdW[f'W{t}'] = np.zeros_like(self.params[f'W{t}'])
            self.Vdb[f'b{t}'] = np.zeros_like(self.params[f'b{t}'])
                
            # Initiatilize Velocity and history - RMSprop
            self.SdW[f'W{t}'] = np.zeros_like(self.params[f'W{t}'])
            self.Sdb[f'b{t}'] = np.zeros_like(self.params[f'b{t}'])

        for i in range(1, num_layers):
            # Applying Velocity for momentum & RMSprop
            self.VdW[f'W{t}'] = self.ema_momentum * self.VdW[f'W{t}'] + (1 - self.ema_momentum) * (self.gradients[f'dW{t}'])
            self.SdW[f'W{t}'] = self.ema_momentum_beta2 * self.SdW[f'W{t}'] + (1 - self.ema_momentum_beta2) * self.gradients[f'dW{t}']**2

            self.Vdb[f'b{t}'] = self.ema_momentum * self.Vdb[f'b{t}'] + (1 - self.ema_momentum) * (self.gradients[f'db{t}'])
            self.Sdb[f'b{t}'] = self.ema_momentum_beta2 * self.Sdb[f'b{t}'] + (1 - self.ema_momentum_beta2) * self.gradients[f'db{t}']**2
                    
            # Bias corrected momentum & RMSprop
            self.VdW_corrected = self.VdW[f'W{t}']/(1 - (self.ema_momentum**i)) 
            self.SdW_corrected = self.SdW[f'W{t}']/(1 - (self.ema_momentum_beta2**i))

            self.Vdb_corrected = self.Vdb[f'b{t}']/(1 - (self.ema_momentum**i)) 
            self.Sdb_corrected = self.Sdb[f'b{t}']/(1 - (self.ema_momentum_beta2**i))
            
            # Update rule 
            self.params[f'W{t}'] -= self.lr * self.VdW_corrected / np.sqrt(self.SdW_corrected + self.epsilon)
            self.params[f'b{t}'] -= self.lr * self.Vdb_corrected / np.sqrt(self.Sdb_corrected + self.epsilon)

        
        return self.params
