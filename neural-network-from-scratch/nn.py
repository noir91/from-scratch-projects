import numpy as np
from activation_func.activations import relu, relu_derivative, sigmoid, softmax, sigmoid_derviative

class nn:
    def __init__(self, params):
        self.params = params
        self.Rmu = {}
        self.Rvar = {}
        
    def forward(self, X, activations = [], batch_norm = False, test = False, verbose = False):
        # Caching X to A0
        cache = {}
        cache['A0'] = X
        num_layers = len(self.params) // 4 if batch_norm else len(self.params) // 2
        count_i = 0
                    
        # Forward Propagation with Batch normalization eliminating beta parameter
        if batch_norm == True:
            epsilon = 1e-5
            ema_momentum = 0.9
            
            # Intializing Velocity
            if not self.Rmu and not self.Rvar:
                for i in range(1, num_layers+1):  
                    # (1, num_features)
                    self.Rmu[f'mu{i}'] = np.zeros((1, self.params[f'W{i}'].shape[1]))
                    self.Rvar[f'var{i}'] = np.zeros((1, self.params[f'W{i}'].shape[1]))
            
            # forward pass
            for i in range(1, num_layers+1):
                count_i +=1
                cache['Z' + str(i)] = np.dot(cache[f'A{i-1}'], self.params[f'W{i}'])
                
                # last layer
                if i == num_layers:
                    cache[f'Z{i}'] = cache[f'Z{i}'] + self.params[f'b{i}']
                    ZL = cache[f'Z{i}']
                    break
                    
                # Run time statistics loading
                if test:
                    mu = self.Rmu[f'mu{i}']
                    actual_var = self.Rvar[f'var{i}']
                    
                    std = np.sqrt(actual_var + epsilon)

                # Training on mu, var, std of Z ith and storing running mean and variances
                else:
                    if count_i < num_layers:
                        mu = np.mean(cache[f'Z{i}'], axis = 0, keepdims = True)
                        actual_var = np.var(cache[f'Z{i}'], axis = 0, keepdims = True)
        
                        std = np.sqrt(actual_var + epsilon)

                        # storing seperate running mean and variances using EWMA
                        self.Rmu[f'mu{i}'] = ema_momentum * self.Rmu[f'mu{i}'] + (1 - ema_momentum) * mu
                        self.Rvar[f'var{i}'] = ema_momentum * self.Rvar[f'var{i}'] + (1 - ema_momentum) * actual_var  
                        
                        # Pytorch convention, only use when the entire codebase follows such conventions for momentum
                        #self.Rmu[f'mu{i}'] = (1 - ema_momentum) * self.Rmu[f'mu{i}'] + ema_momentum * mu
                        #self.Rvar[f'var{i}'] = (1 - ema_momentum) * self.Rvar[f'var{i}'] + ema_momentum * actual_var
                        
                #if count_i < num_layers: 

                # Applying Batch Normalization 
                cache['Z_norm' + str(i)] = (cache[f'Z{i}'] - mu) / std
                cache['Z_tilde' + str(i)] = self.params[f'G{i}'] * cache[f'Z_norm{i}'] + self.params[f'B{i}']

                # Passing Mu, Var, Std to cache for use in backpropagation avoiding recomputation and batch statistics mismatch
                cache['Z_mu' + str(i)] = mu
                cache['Z_var' + str(i)] = actual_var
                cache['Z_std' + str(i)] = std
                
                cache['A' + str(i)] = activations[i-1](cache[f'Z_tilde{i}']) 
                if verbose:
                    mv = cache[f'Z_norm{i}'].mean(), cache[f'Z_norm{i}'].var()
                    print(f"Layer {i} | Mean(Z_norm) = {mv[0]:.6f} | Var(Z_norm) = {mv[1]:.6f}")
                    print(f"  Running mean: {self.Rmu[f'mu{i}'].mean():.6f} | Running var: {self.Rvar[f'var{i}'].mean():.6f}")
                    print(f"  Gamma[{i}] mean: {np.mean(self.params[f'G{i}']):.6f} | Beta[{i}] mean: {np.mean(self.params[f'B{i}']):.6f}")
            
                #else:
                    #cache['Z' + str(i)] = np.dot(cache[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
                        #continue
            ZL = cache[f'Z{num_layers}']
                
        else:
        # Standard Forward Propagation
            for i in range(1, num_layers+1):
                count_i +=1
                #changed the orientation from W.T+A to A.W
                cache['Z' + str(i)] = np.dot(cache[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}'] 
                
                if count_i > num_layers:
                    continue
                else:
                    cache['A' + str(i)] = activations[i-1](cache[f'Z{i}'])
                    
                    #--------------DEBUGGING-------------
                    #print(f'Z{i} : {cache['Z' + str(i)].shape}')
                    #print(f'A{i} : {cache['A' + str(i)].shape}')
                    #print(f'A{i}.max = {cache['A' + str(i)]}')
                    #--------------DEBUGGING-------------
            #changed y_pred from y_pred = cache[f'A{num_layers}'] to cache[f'Z{num_layers}'] to be used by cross_entropy_with_logits
            ZL = cache[f'Z{num_layers}'] 
        return ZL, cache

    # Backward propagation
    def backward(self, y_true, dZL, cache, activations = [], batch_norm = False):
        m = y_true.shape[0]
        L = len(self.params) // 4 if batch_norm else len(self.params) // 2
        epsilon = 1e-5
        
        if batch_norm == True:
            # Linear layer backprop (L layer)
            A_prev = cache['A' + str(L-1)]
            
            dWL = (1.0 / m) * np.dot(A_prev.T, dZL)
            dbL = (1.0 / m) * np.sum(dZL, axis = 0, keepdims = True)
            
            gradients = {'dZ' + str(L): dZL,
                       'dW' + str(L): dWL,
                       'db' + str(L): dbL,
                        }
            
            # Back propagating the rest of the network
            for l in reversed(range(1, L)):

                # parameters to be used # use cached fp to bp.
                var = cache[f'Z_var{l}']
                std = cache[f'Z_std{l}']
                
                gamma = self.params[f'G{l}']
                Z_tilde = cache[f'Z_tilde{l}']
                Z_norm = cache[f'Z_norm{l}']
    
                # Finding gradients
                gradients['dA' + str(l)] = np.dot(gradients[f'dZ{l+1}'], self.params[f'W{l+1}'].T)
    
                # Activations through backprop
                if activations[l-1] == sigmoid:
                  gradients['dZ' + str(l)] = gradients[f'dA{l}'] * sigmoid_derviative(Z_tilde)
                    
                else:
                  gradients['dZ' + str(l)] = gradients[f'dA{l}'] * relu_derivative(Z_tilde)
                gradients['dG' + str(l)] = np.sum(gradients[f'dZ{l}'] * cache[f'Z_norm{l}'], axis = 0, keepdims = True)
                gradients['dB' + str(l)] = np.sum(gradients[f'dZ{l}'], axis = 0, keepdims = True)
                
                dZl_tilde = (1.0 / m) * (1 / std) * ( 
                    (m * gradients[f'dZ{l}'] * gamma) 
                    - np.sum(gradients[f'dZ{l}']* gamma,  axis = 0, keepdims = True) 
                    - Z_norm * np.sum(gradients[f'dZ{l}'] * gamma * Z_norm, axis = 0, keepdims = True)
                )
                
                gradients['dW' + str(l)] = (1.0 / m) * np.dot(cache[f'A{l-1}'].T, dZl_tilde)  
            gradients = {k: v for k, v in gradients.items() if k.startswith(('dW','dB','dG','db'))}
              
        else:

            # Performing backpropogation for last layer since it does not follow a general format

            #AL = cache['A' + str(L)]
            #dZL =  AL - y_true
            A_prev = cache['A' + str(L-1)]
            
            dWL = (1.0 / m) * np.dot(A_prev.T, dZL)
            dbL = (1.0 / m) * np.sum(dZL, axis = 0, keepdims = True)
            gradients = {'dZ' + str(L): dZL,
                       'dW' + str(L): dWL,
                       'db' + str(L): dbL
                       }
            
            #--------------DEBUGGING-------------
            #print(f'dZ{L} : {gradients['dZ' + str(L)].shape}')
            #print(f'dW{L} : {gradients['dW' + str(L)].shape}')
            #print(f'db{L} : {gradients['db' + str(L)].shape}')
            #--------------DEBUGGING-------------
            
            # Calculating Back propagation on the rest of the network
            for l in reversed(range(1, L)):
                gradients['dA' + str(l)] = np.dot(gradients[f'dZ{l+1}'], self.params[f'W{l+1}'].T) # switched their positions
    
                # Activations for backprop
                if activations[l-1] == sigmoid:
                    gradients['dZ' + str(l)] = gradients[f'dA{l}'] * sigmoid_derviative(cache[f'Z{l}'])
                else:
                    gradients['dZ' + str(l)] = gradients[f'dA{l}'] * relu_derivative(cache[f'Z{l}'])
                gradients['dW' + str(l)] = (1.0 / m) * np.dot(cache[f'A{l-1}'].T, gradients[f'dZ{l}']) #switched their
                gradients['db' + str(l)] = (1.0 / m) * np.sum(gradients[f'dZ{l}'], axis = 0, keepdims = True) #changed axis 1 to 0
                
                   #--------------DEBUGGING-------------
                  #print(f'dA{l} : {gradients['dA' + str(l)].shape}')
                  #print(f'dZ{l} : {gradients['dZ' + str(l)].shape}')
                  #print(f'dW{l} : {gradients['dW' + str(l)].shape}')
                  #print(f'db{l} : {gradients['db' + str(l)].shape}')
                
                #--------------DEBUGGING-------------
            # Storing dW and db to gradients
            gradients = {k: v for k, v in gradients.items() if k.startswith(('dW','db'))}
              
        return gradients 
    
    def accuracy(self, X, y_true, activations = [], batch_norm = False):
        if batch_norm:
            y_pred, _ = self.forward(X, activations, batch_norm = True, test = True)
            
        else:
            y_pred, _ = self.forward(X, activations)
            
        preds = np.argmax(y_pred, axis=1)
        return np.mean(preds == y_true)
   

    
    
            