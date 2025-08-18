import numpy as np
from activation_func.activations import relu, relu_derivative, sigmoid, softmax, sigmoid_derviative
from optim.SGD import SGD
def random_init(layers_dim):
  np.random.seed(0)
  params = {}

  L = len(layers_dim) - 1

  for i in range(1, L+1):
     params["W" + str(i)]= np.random.randn(layers_dim[i], layers_dim[i-1]) * np.sqrt(2. / layers_dim[i-1])
     params["b" + str(i)]= np.zeros((layers_dim[i],1))
  return params

def forward(X, params, activations = []):
  # Caching X to A0
  cache = {'A0': X}
  num_layers = len(params) // 2

  # Forward Propagation
  for i in range(1,num_layers+1):
    cache['Z' + str(i)] = params[f'W{i}'] @ cache[f'A{i-1}'] + params[f'b{i}']
    cache['A' + str(i)] = activations[i-1](cache[f'Z{i}'])

  y_pred = cache[f'A{num_layers}']

  return y_pred, cache

# Cross Entropy Loss
def crossentropy(y_pred, y_true):
  sample =-np.sum(y_true * np.log(y_pred + 1e-8) , axis = 0)
  loss = np.mean(sample)
  return loss

# Backward propagation
def backward(y_true, cache, params, activations = []):
  m = y_true.shape[1]
  L = len(params) // 2

  # Performing backpropogation for last layer since it does not follow a general format
  AL = cache['A' + str(L)]
  dZL =  AL - y_true
  A_prev = np.dot(params[f'W{L}'].T, dZL)

  dWL = (1.0 / m) * np.dot(dZL, A_prev.T)
  dbL = (1.0 / m) * np.sum(dZL, axis = 1, keepdims = True)
  gradients = {'dZ' + str(L): dZL,
               'dW' + str(L): dWL,
               'db' + str(L): dbL
               }
  
  # Calculating Back propagation on the rest of the network
  for l in reversed(range(1, L)): 
    gradients['dA' + str(l)] = np.dot(params[f'W{l+1}'].T, gradients[f'dZ{l+1}'])
    # If Activation function is a sigmoid, use derivative of a sigmoid
    if activations[l-1] == sigmoid:
      gradients['dZ' + str(l)] = gradients[f'dA{l}'] * sigmoid_derviative(cache[f'Z{l}'])
    
    # Else use derivative of ReLU
    else:
      gradients['dZ' + str(l)] = gradients[f'dA{l}'] * relu_derivative(cache[f'Z{l}'])
      gradients['dW' + str(l)] = (1.0 / m) * np.dot(gradients[f'dZ{l}'], cache[f'A{l-1}'].T)
      gradients['db' + str(l)] = (1.0 / m) * np.sum(gradients[f'dZ{l}'], axis = 1, keepdims = True)

  # Storing dW and db to gradients
  gradients = {k: v for k, v in gradients.items() if k.startswith('dW') or k.startswith('db')}
  return gradients

# One-Hot Encoding
def onehot(y, classes):

  one_hot_y = np.zeros((classes, y.size))
  one_hot_y[y, np.arange(y.size)] = 1
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
def accuracy(X, y_true, params, activations):
  y_pred, _ = forward(X, params, activations)
  preds = np.argmax(y_pred, axis=0)
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