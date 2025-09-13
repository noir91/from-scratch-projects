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
    def __init__(self, params, lr, beta = 0.9, momentum = False, batch_norm = False):
        self.lr = lr
        self.params = params
        self.beta = beta
        self.momentum = momentum
        self.VdW = {}
        self.VdG = {}
        self.VdB = {}
        self.Vdb = {} 

        self.batch_norm = batch_norm
        self.num_layers = max(int(k[1:]) for k in self.params if k.startswith('W'))

        # Initialize Velocity and history
        for i in range(1, self.num_layers + 1):  
            #  Params
            W = self.params[f'W{i}']

            if self.batch_norm and self.momentum:
                
                B = self.params[f'B{i}']
                G = self.params[f'G{i}']

                self.VdW[f'W{i}'] = np.zeros_like(W)
                self.VdG[f'G{i}'] = np.zeros_like(G)
                self.VdB[f'B{i}'] = np.zeros_like(B)

                if i == self.num_layers:
                    b = self.params[f'b{i}']
                    self.Vdb[f'b{i}'] = np.zeros_like(b)

            elif not self.batch_norm and self.momentum:

                b = self.params[f'b{i}']  

                self.VdW[f'W{i}'] = np.zeros_like(W)
                self.Vdb[f'b{i}'] = np.zeros_like(b)

    def step(self, gradients): 
        self.num_layers = max(int(k[2:]) for k in gradients if k.startswith('dW'))

        # Updates Batch Norm w/Momentum
        if self.batch_norm:
            for i in range(1, self.num_layers + 1):
                if i == self.num_layers:
                    dW, db = gradients[f'dW{i}'], gradients[f'db{i}']
                    W = self.params[f'W{i}']
                    b = self.params[f'b{i}']
                    break
                
                else:
                    #  Gradients
                    dW, dB, dG = gradients[f'dW{i}'], gradients[f'dB{i}'], gradients[f'dG{i}']

                    #  Params
                    W = self.params[f'W{i}']
                    B = self.params[f'B{i}']
                    G = self.params[f'G{i}']

                # BatchNorm w/Momentum
                if self.momentum:

                    # Update Rule for last layer where Gamma and Beta are absent
                        if i == self.num_layers:
                            # Applying Velocity for momentum
                            self.VdW[f'W{i}'] = self.beta * self.VdW[f'W{i}'] + (1 - self.beta) * dW
                            self.Vdb[f'b{i}'] = self.beta * self.Vdb[f'b{i}'] + (1 - self.beta) * db

                            # Update Rule
                            W -= self.lr * self.Vdb[f'b{i}']
                            b -= self.lr * self.Vdb[f'b{i}']

                            # Restoring Variables
                            self.params[f'W{i}'], self.params[f'b{i}'] = W, b
                            break

                    # Applying Velocity for momentum
                        self.VdW[f'W{i}'] = self.beta * self.VdW[f'W{i}'] + (1 - self.beta) * dW
                        self.VdB[f'B{i}'] = self.beta * self.VdB[f'B{i}']  + (1 - self.beta) * dB
                        self.VdG[f'G{i}'] = self.beta * self.VdG[f'G{i}'] + (1 - self.beta) * dG

                        # Update Rule
                        W -= self.lr * self.VdW[f'W{i}']
                        B -= self.lr * self.VdB[f'B{i}']
                        G -= self.lr * self.VdG[f'G{i}']
                        

                        # Restoring Variables
                        self.params[f'W{i}'], self.params[f'G{i}'], self.params[f'B{i}'] = W, G, B

                # BatchNorm != w/Momentum                   
                else:
                    
                    # Update Rule for last layer where Gamma and Beta are absent
                        if i == self.num_layers:
                            # Update Rule
                            W -= self.lr * dW
                            b -= self.lr * db

                            # Restoring Variables
                            self.params[f'W{i}'], self.params[f'b{i}'] = W, b
                            break

                        # Update Rule
                        W -= self.lr * dW
                        G -= self.lr * dG
                        B -= self.lr * dB   

                        # Restoring Variables
                        self.params[f'W{i}'], self.params[f'G{i}'], self.params[f'B{i}'] = W, G, B

        # Updates vanilla SGD w/Momentum
        else:
        
            for i in range(1, self.num_layers + 1):
                #   Gradients
                dW = gradients[f'dW{i}']
                db = gradients[f'db{i}']

                #  Params       
                W = self.params[f'W{i}']
                b = self.params[f'b{i}']    

                if self.momentum:
                    # Applying Velocity for momentum
                    self.VdW[f'W{i}'] = self.beta * self.VdW[f'W{i}'] + (1 - self.beta) * dW
                    self.Vdb[f'b{i}'] = self.beta * self.Vdb[f'b{i}'] + (1 - self.beta) * db

                    # Restoring Variables
                    self.params[f'W{i}'], self.params[f'b{i}'] = W, b

                    # Update rule
                    W -= self.lr * self.VdW[f'W{i}']
                    b -= self.lr * self.Vdb[f'b{i}']
                else:

                    # Update rule
                    W -= self.lr * dW
                    b -= self.lr * db

                    # Restoring Variables
                    self.params[f'W{i}'], self.params[f'b{i}'] = W, b

        return self.params