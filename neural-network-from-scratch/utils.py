import numpy as np
from activation_func.activations import relu, relu_derivative, sigmoid, softmax, sigmoid_derviative, cross_entropy_from_logits
from optim.SGD import SGD

def random_init(layers_dim, batch_norm = False):
    np.random.seed(0)
    params = {}
    
    L = len(layers_dim) - 1

    if batch_norm == False:
        for i in range(1, L+1):
            params["W" + str(i)]= np.random.randn(layers_dim[i-1], layers_dim[i]) * np.sqrt(2. / layers_dim[i-1]) # changing the shapes from i i-1 to i-1 to i
            params["b" + str(i)]= np.zeros((1,layers_dim[i]))
            #--------------DEBUGGING-------------
            #print(params['W' + str(i)].shape)
            #print(params['b' + str(i)].shape)
            #--------------DEBUGGING-------------
    else:
        for i in range(1, L+1):
            params['W' + str(i)] = np.random.randn(layers_dim[i-1], layers_dim[i]) * np.sqrt(2. / layers_dim[i-1])
            params['b' + str(i)] = np.zeros((1, layers_dim[i]))
            params['B' + str(i)] = np.zeros((1, layers_dim[i]))
            params['G' + str(i)] = np.ones((1, layers_dim[i]))
            
    return params

def forward(X, params, activations = [], batch_norm = False, test = False):
    # Caching X to A0
    cache = {'A0': X}
    num_layers = len(params) // 3 if batch_norm else len(params) // 2
    count_i = 0
                
    # Forward Propagation with Batch normalization eliminating beta parameter
    if batch_norm == True:
        if not Rmu and not Rvar:
            Rmu = {}
            Rvar = {}
        else:
            pass
        epsilon = 1e-9
        ema_momentum = 0.9
        # Intializing Velocity 
        if not Rmu and not Rvar:
            for i in range(1, num_layers):  
                # (1, num_features)
                Rmu[f'mu{i}'] = np.zeros((1, params[f'W{i}'].shape[1]))
                Rvar[f'var{i}'] = np.zeros((1, params[f'W{i}'].shape[1]))
        
        # forward pass
        for i in range(1, num_layers+1):
            count_i +=1
            cache['Z' + str(i)] = np.dot(cache[f'A{i-1}'], params[f'W{i}'])
            
            mu = np.mean(cache[f'Z{i}'], axis = 0, keepdims = True)
            var = np.sqrt(np.var(cache[f'Z{i}']) + epsilon, axis = 0, keepdims = True)
            
            # storing seperate running mean and variances using EWMA
            Rmu[f'mu{i}'] = ema_momentum * Rmu[f'mu{i-1}'] + (1 - ema_momentum) * mu
            Rvar[f'mu{i}'] = ema_momentum * Rvar[f'var{i-1}'] + (1 - ema_momentum) * var
            
            cache['Z_norm' + str(i)] = (cache[f'Z{i}'] - mu) / var
            cache['Z_tilde' + str(i)] = params[f'G{i}'] * cache[f'Z_norm{i}'] + params[f'B{i}']
            
            if count_i > 2:
                continue
            else:
                cache['A' + str(i)] = activations[i-1](cache[f'Z_tilde{i}'])
               
        ZL_tilde = cache[f'Z_tilde{num_layers}']
    else:
    # Standard Forward Propagation
        for i in range(1,num_layers+1):
            count_i +=1
            cache['Z' + str(i)] = np.dot(cache[f'A{i-1}'], params[f'W{i}']) + params[f'b{i}'] #changed the orientation from W.T+A to A.W
            if count_i > 2:
                continue
            else:
                cache['A' + str(i)] = activations[i-1](cache[f'Z{i}'])
                
                #--------------DEBUGGING-------------
                #print(f'Z{i} : {cache['Z' + str(i)].shape}')
                #print(f'A{i} : {cache['A' + str(i)].shape}')
                #print(f'A{i}.max = {cache['A' + str(i)]}')
                #--------------DEBUGGING-------------
        
        ZL = cache[f'Z{num_layers}'] #changed y_pred from y_pred = cache[f'A{num_layers}'] to cache[f'Z{num_layers}'] to be used by cross_entropy_with_logits
    return (ZL_tilde, cache) if batch_norm else (ZL, cache)

# Backward propagation
def backward(y_true, dZL, cache, params, activations = [], gradcheck = False, batch_norm = False):
    m = y_true.shape[0]
    L = len(params) // 3 if batch_norm else len(params) // 2
    epsilon = 1e-9
    
    if batch_norm == True:
        # Side note: dZL here isn't dZL infact, it is the last Z_tilde parameter from forward propagation
        A_prev = cache['A' + str(L-1)]
        
        # Batch norm dZL_tilde
        var = np.var(cache[f'Z{L}'])
        gamma = params[f'G{L}']
        ZL_norm = cache[f'Z_norm{L}']
        
        dZL_tilde = (1.0 / m) * 1 / np.sqrt(var + epsilon) * ( 
            (m * dZL * gamma) 
            - np.sum(dZL* gamma, axis = 0, keepdims = True) 
            - ZL_norm * np.sum(dZL * gamma * ZL_norm, axis = 0, keepdims = True)
        )
        
        dGL = np.sum(dZL_tilde * cache[f'Z_norm{L}'], axis= 0, keepdims = True)
        dBL = np.sum(dZL_tilde, axis = 0, keepdims = True)
        
        dWL = (1.0 / m) * np.dot(A_prev.T, dZL_tilde)
        
        gradients = {'dZ' + str(L): dZL_tilde,
                   'dW' + str(L): dWL,
                   'dB' + str(L): dBL,
                    'dG' + str(L): dGL
                    }
        
        # Back propagating the rest of the network
        for l in reversed(range(1, L)): 
            
            # parameters to be used
            var = np.var(cache[f'Z{l}'], axis = 0, keepdims = True)
            gamma = params[f'G{l}']
            Z_tilde = cache[f'Z_tilde{l}']
            Z_norm = cache[f'Z_norm{l}']

            # Finding gradients
            gradients['dA' + str(l)] = np.dot(gradients[f'dZ{l+1}'], params[f'W{l+1}'].T)

            # Activations through backprop
            if activations[l-1] == sigmoid:
              gradients['dZ' + str(l)] = gradients[f'dA{l}'] * sigmoid_derviative(Z_tilde)
            else:
              gradients['dZ' + str(l)] = gradients[f'dA{l}'] * relu_derivative(Z_tilde)
            gradients['dG' + str(l)] = np.sum(gradients[f'dZ{l}'] * cache[f'Z_norm{l}'], axis = 0, keepdims = True)
            gradients['dB' + str(l)] = np.sum(gradients[f'dZ{l}'], axis = 0, keepdims = True)
            
            dZl_tilde = (1.0 / m) * 1 / np.sqrt(var + epsilon) * ( 
                (m * gradients[f'dZ{l}'] * gamma) 
                - np.sum(gradients[f'dZ{l}']* gamma,  axis = 0, keepdims = True) 
                - Z_norm * np.sum(gradients[f'dZ{l}'] * gamma * Z_norm, axis = 0, keepdims = True)
            )
            
            gradients['dW' + str(l)] = (1.0 / m) * np.dot(cache[f'A{l-1}'].T, dZl_tilde)  
        gradients = {k: v for k, v in gradients.items() if k.startswith(('dW','dB','dG'))}
            
    # Performing backpropogation for last layer since it does not follow a general format
    else:
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
            gradients['dA' + str(l)] = np.dot(gradients[f'dZ{l+1}'], params[f'W{l+1}'].T) # switched their positions

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

# One-Hot Encoding
# I flipped the axis to row wise from column wise
def onehot(y, classes):

  one_hot_y = np.zeros((y.size, classes))
  one_hot_y[np.arange(y.size), y] = 1
  return one_hot_y

# Training
def train(lr, X, y_true, layers_dim, epochs):

  # Random Intialization of w, b
  params = random_init(layers_dim)

  # Mini batch
  batch_size = 64
  m = X.shape[1]
  num_indices = m // batch_size

  for epoch in range(epochs):
    # shuffle on each epoch
    perm = np.random.permutation(m)
    X_shuffled = X[:,perm]
    y_shuffled = y_true[:,perm]

    epoch_loss = 0.0
    for i in range(num_indices):
     start = i * batch_size
     end = start + batch_size
     X_batch = X_shuffled[:,start:end]
     y_batch = y_shuffled[:,start:end]

     # forward pass
     y_pred, cache = forward(X_batch, params, activations = [relu, relu, softmax])

     # computing loss
     loss = crossentropy(y_pred, y_batch)
     epoch_loss += loss

     # backward prop
     gradients = backward(y_batch, cache, params, activations = [relu, relu, softmax])

     # Optimizer
     optimizer = SGD(lr, params, gradients)
     # update weights
     params = optimizer.step()
     
    print(f"Epochs : {epoch+1}, Loss: {epoch_loss:.4f}")
  return params

# Accuracy Check
# changed axis to 1 , for row outputs. If i get a problem here in the future make the axis = 0
def accuracy(X, y_true, params, activations, batch_norm = False):
    if batch_norm:
        y_pred, _ = forward(X, params, activations, batch_norm = True)
        preds = np.argmax(y_pred, axis=1)
        return np.mean(preds == y_true)
    else:
        y_pred, _ = forward(X, params, activations)
        preds = np.argmax(y_pred, axis=1)
        return np.mean(preds == y_true)

# Update gradients
def update_weights(lr,  params, gradients):
  params["W1"] -= lr*gradients["dW1"]
  params["b1"] -= lr*gradients["db1"]

  params["W2"] -= lr*gradients["dW2"]
  params["b2"] -= lr*gradients["db2"]

  params["W3"] -= lr*gradients["dW3"]
  params["b3"] -= lr*gradients["db3"]
  
  return params

def compute_validation_loss(model, X_val, y_val, activations, batch_norm=False):
    # Forward pass on validation data
    Z_val, _ = model.forward(X_val, activations, batch_norm=batch_norm, test=True)
    
    # Compute cross-entropy loss
    loss, _ = cross_entropy_from_logits(Z_val, y_val)
    
    return loss