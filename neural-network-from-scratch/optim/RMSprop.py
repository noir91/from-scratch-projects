import numpy as np
import copy

class RMSprop:
    """
    eps is used for numerical stability

    func step()
        - takes gradient descent steps
        - updates weights
e
    RMSprop uses EWMA momentum to give more loss moving in an horizontal direction and less importance to loss moving in a vertical direction
    allowing for greater learning rates, and accelerated learning of the loss function.

        Vtheta = B_2 * Vt-1 + (1-B_2) * Vt

    """
    def __init__(self, params, lr, eps = 1e-8, beta2 = 0.9, batch_norm = False):
        self.lr = lr
        self.params = params
        self.batch_norm = batch_norm

        self.eps = eps
        self.beta2 = beta2
        
        self.SdW, self.SdB, self.SdG, self.Sdb = {}, {}, {}, {}

        self.num_layers = max(int(k[1:]) for k in self.params if k.startswith('W'))

        # Initialize Velocity and history
        for i in range(1, self.num_layers + 1): 
            if self.batch_norm:
            # Update Rule for last layer where Gamma and Beta are absent
                if i == self.num_layers:
                    # Params
                    W, b = self.params[f'W{i}'], self.params[f'b{i}']
                    
                    self.SdW[f'W{i}'] = np.zeros_like(W)
                    self.Sdb[f'b{i}'] = np.zeros_like(b)
                    break

                #  Params
                W, B, G = self.params[f'W{i}'], self.params[f'B{i}'], self.params[f'G{i}'] 

                self.SdW[f'W{i}'] = np.zeros_like(W)
                self.SdB[f'B{i}'] = np.zeros_like(B)
                self.SdG[f'G{i}'] = np.zeros_like(G)
            else:
                # Params
                W, b = self.params[f'W{i}'], self.params[f'b{i}'] 

                self.SdW[f'W{i}'] = np.zeros_like(W)
                self.Sdb[f'b{i}'] = np.zeros_like(b)

    def step(self, gradients):

        self.num_layers = max(int(k[2:]) for k in gradients if k.startswith('dW'))

        for i in range(1, self.num_layers + 1):
               

            if self.batch_norm:
            # Update Rule for last layer where Gamma and Beta are absent
                if i == self.num_layers:
                    # Fetching Variables
                    dW, db = gradients[f'dW{i}'], gradients[f'db{i}']
                    W, b = self.params[f'W{i}'], self.params[f'b{i}']
                    
                    # Applying Velocity for momentum
                    self.SdW[f'W{i}'] = self.beta2 * self.SdW[f'W{i}'] + (1 - self.beta2) * (dW)**2
                    self.Sdb[f'b{i}']  = self.beta2 * self.Sdb[f'b{i}'] + (1 - self.beta2) * (db)**2

                    # Update Rule
                    W -= self.lr * self.SdW[f'W{i}']
                    b -= self.lr * self.Sdb[f'b{i}']

                    # Restoring Variables
                    self.params[f'W{i}'], self.params[f'b{i}'] = W, b
                    break

                # Fetching Variables
                dW, dB, dG = gradients[f'dW{i}'], gradients[f'dB{i}'], gradients[f'dG{i}']
                W, B, G = self.params[f'W{i}'], self.params[f'B{i}'], self.params[f'G{i}']

                # Applying Velocity for momentum
                self.SdW[f'W{i}'] = self.beta2 * self.SdW[f'W{i}'] + (1 - self.beta2) * (dW)**2
                self.SdB[f'B{i}']  = self.beta2 * self.SdB[f'B{i}'] + (1 - self.beta2) * (dB)**2
                self.SdG[f'G{i}']  = self.beta2 * self.SdG[f'G{i}'] + (1 - self.beta2) * (dG)**2
    
                # Update rule
                W -= self.lr * dW / np.sqrt(self.SdW[f'W{i}'] + self.eps)
                B -= self.lr * dB / np.sqrt(self.SdB[f'B{i}'] + self.eps)
                G -= self.lr * dG / np.sqrt(self.SdG[f'G{i}'] + self.eps)

                # Restoring Variables
                self.params[f'W{i}'], self.params[f'G{i}'], self.params[f'B{i}'] = W, G, B

            else:
                dW, db = gradients[f'dW{i}'], gradients[f'db{i}']
                W, b = self.params[f'W{i}'], self.params[f'b{i}']   

                # Applying Velocity for momentum
                self.SdW[f'W{i}']  = self.beta2 * self.SdW[f'W{i}'] + (1 - self.beta2) * (dW)**2
                self.Sdb[f'b{i}']  = self.beta2 * self.Sdb[f'b{i}'] + (1 - self.beta2) * (db)**2
                
                # Update rule
                W -= self.lr * dW / np.sqrt(self.SdW[f'W{i}'] + self.eps)
                b -= self.lr * db / np.sqrt(self.Sdb[f'b{i}'] + self.eps)   

                # Restoring Variables
                self.params[f'W{i}'], self.params[f'b{i}'] = W, b

        return self.params
