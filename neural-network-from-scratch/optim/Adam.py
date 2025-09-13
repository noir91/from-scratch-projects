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
    def __init__(self, params, lr, beta = 0.9, beta2 = 0.99, eps = 1e-8, batch_norm = False): # putting params in step()
        self.lr = lr
        self.params = params
        self.beta = beta
        self.beta2 = beta2

        self.batch_norm = batch_norm
        self.eps = eps

        self.VdW, self.Vdb = {}, {}
        self.SdW, self.Sdb = {}, {}

        self.VdG, self.VdB = {}, {}
        self.SdG, self.SdB = {}, {}
        
        # time variable
        self.t = 0

        self.num_layers = max(int(k[1:]) for k in self.params if k.startswith('W'))
        # Initialize Velocity and history - Momentum            
        for i in range(1, self.num_layers + 1):  
    
            # Batch Normalization parameters velocity and history
            if self.batch_norm:
                # Update Rule for last layer where Gamma and Beta are absent
                if i == self.num_layers:
                    # Params
                    W = self.params[f'W{i}']
                    b = self.params[f'b{i}']

                    self.VdW[f'W{i}'] = np.zeros_like(W)
                    self.SdW[f'W{i}'] = np.zeros_like(W)
                    
                    self.Vdb[f'b{i}'] = np.zeros_like(b)
                    self.Sdb[f'b{i}'] = np.zeros_like(b)
                    break

                # Params
                W = self.params[f'W{i}']
                B = self.params[f'B{i}']
                G = self.params[f'G{i}']

                # Momentum
                self.VdW[f'W{i}'] = np.zeros_like(W)
                self.VdG[f'G{i}'] = np.zeros_like(G)
                self.VdB[f'B{i}'] = np.zeros_like(B)
                
                # RMSprop
                self.SdW[f'W{i}'] = np.zeros_like(W)
                self.SdG[f'G{i}'] = np.zeros_like(G)
                self.SdB[f'B{i}'] = np.zeros_like(B)
                
            else:
                # Params
                W = self.params[f'W{i}']
                b = self.params[f'b{i}']

                # Momentum
                self.VdW[f'W{i}'] = np.zeros_like(W)
                self.Vdb[f'b{i}'] = np.zeros_like(b)
                    
                # RMSprop
                self.SdW[f'W{i}'] = np.zeros_like(W)
                self.Sdb[f'b{i}'] = np.zeros_like(b)

    def step(self, gradients): 
        self.t += 1            
        self.num_layers = max(int(k[2:]) for k in gradients if k.startswith('dW'))

    
        for i in range(1, self.num_layers + 1):
            
            # Adam w/BatchNorm
            if self.batch_norm:
                # Update Rule for last layer where Gamma and Beta are absent
                if i == self.num_layers:
                    # Fetching Variables
                    W, b = self.params[f'W{i}'], self.params[f'b{i}']
                    
                    # Gradients
                    dW = gradients[f'dW{i}']
                    db = gradients[f'db{i}']

                    # Applying Velocity for momentum & RMSprop
                    self.VdW[f'W{i}']  = self.beta * self.VdW[f'W{i}']  + (1 - self.beta) * (dW)
                    self.SdW[f'W{i}']  = self.beta2 * self.SdW[f'W{i}']  + (1 - self.beta2) * dW**2

                    self.Vdb[f'b{i}']  = self.beta * self.Vdb[f'b{i}']  + (1 - self.beta) * (db)
                    self.Sdb[f'b{i}']  = self.beta2 * self.Sdb[f'b{i}']  + (1 - self.beta2) * db**2
                            
                    # Bias corrected momentum & RMSprop
                    self.VdW_corrected = self.VdW[f'W{i}'] /(1 - (self.beta ** self.t)) 
                    self.SdW_corrected = self.SdW[f'W{i}'] /(1 - (self.beta2 ** self.t))

                    self.Vdb_corrected = self.Vdb[f'b{i}'] /(1 - (self.beta ** self.t)) 
                    self.Sdb_corrected = self.Sdb[f'b{i}'] /(1 - (self.beta2 ** self.t))
                    
                    # Update rule 
                    W -= self.lr * self.VdW_corrected / np.sqrt(self.SdW_corrected + self.eps)
                    b -= self.lr * self.Vdb_corrected / np.sqrt(self.Sdb_corrected + self.eps)

                    # Restoring Variables
                    self.params[f'W{i}'], self.params[f'b{i}'] = W, b
                    break

                # Fetching Variables
                W, B, G = self.params[f'W{i}'], self.params[f'B{i}'], self.params[f'G{i}']
                
                # Gradients
                dW = gradients[f'dW{i}']
                dB = gradients[f'dB{i}']
                dG = gradients[f'dG{i}']

                # Applying Velocity for momentum & RMSprop
                self.VdW[f'W{i}'] = self.beta * self.VdW[f'W{i}'] + (1 - self.beta) * (dW)
                self.SdW[f'W{i}'] = self.beta2 * self.SdW[f'W{i}'] + (1 - self.beta2) * dW**2
    
                self.VdB[f'B{i}'] = self.beta * self.VdB[f'B{i}'] + (1 - self.beta) * (dB)
                self.SdB[f'B{i}'] = self.beta2 * self.SdB[f'B{i}'] + (1 - self.beta2) * dB**2
    
                self.VdG[f'G{i}'] = self.beta * self.VdG[f'G{i}'] + (1 - self.beta) * (dG)
                self.SdG[f'G{i}'] = self.beta2 * self.SdG[f'G{i}'] + (1 - self.beta2) * dG**2
                        
                # Bias corrected momentum & RMSprop
                self.VdW_corrected = self.VdW[f'W{i}'] / (1 - (self.beta ** self.t)) 
                self.SdW_corrected = self.SdW[f'W{i}'] / (1 - (self.beta2 ** self.t))
    
                self.VdB_corrected = self.VdB[f'B{i}'] / (1 - (self.beta ** self.t)) 
                self.SdB_corrected = self.SdB[f'B{i}'] / (1 - (self.beta2 ** self.t))
                
                self.VdG_corrected = self.VdG[f'G{i}'] / (1 - (self.beta ** self.t)) 
                self.SdG_corrected = self.SdG[f'G{i}'] / (1 - (self.beta2 ** self.t))
            
                # Update rule 
                W -= self.lr * self.VdW_corrected / np.sqrt(self.SdW_corrected + self.eps)
                B -= self.lr * self.VdB_corrected / np.sqrt(self.SdB_corrected + self.eps)
                G -= self.lr * self.VdG_corrected / np.sqrt(self.SdG_corrected + self.eps)
                
                # Restoring Variables
                self.params[f'W{i}'], self.params[f'G{i}'], self.params[f'B{i}'] = W, G, B
            # Adam  != w/BatchNorm
            else:
                # Fetching Variables
                W, b = self.params[f'W{i}'], self.params[f'b{i}']
                
                # Gradients
                dW = gradients[f'dW{i}']
                db = gradients[f'db{i}']

                # Applying Velocity for momentum & RMSprop
                self.VdW[f'W{i}']  = self.beta * self.VdW[f'W{i}']  + (1 - self.beta) * (dW)
                self.SdW[f'W{i}']  = self.beta2 * self.SdW[f'W{i}']  + (1 - self.beta2) * dW**2
    
                self.Vdb[f'b{i}']  = self.beta * self.Vdb[f'b{i}']  + (1 - self.beta) * (db)
                self.Sdb[f'b{i}']  = self.beta2 * self.Sdb[f'b{i}']  + (1 - self.beta2) * db**2
                        
                # Bias corrected momentum & RMSprop
                self.VdW_corrected = self.VdW[f'W{i}'] /(1 - (self.beta ** self.t)) 
                self.SdW_corrected = self.SdW[f'W{i}'] /(1 - (self.beta2 ** self.t))
    
                self.Vdb_corrected = self.Vdb[f'b{i}'] /(1 - (self.beta ** self.t)) 
                self.Sdb_corrected = self.Sdb[f'b{i}'] /(1 - (self.beta2 ** self.t))
                
                # Update rule 
                W -= self.lr * self.VdW_corrected / np.sqrt(self.SdW_corrected + self.eps)
                b -= self.lr * self.Vdb_corrected / np.sqrt(self.Sdb_corrected + self.eps)

                # Restoring Variables
                self.params[f'W{i}'], self.params[f'b{i}'] = W, b
        return self.params
