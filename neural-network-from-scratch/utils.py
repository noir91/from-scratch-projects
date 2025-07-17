import numpy as np

# Activation functions
def sigmoid(z):
  f = 1/(1+np.exp(-z))
  return f

def softmax(z):
  f = np.exp(z - np.max(z, axis = 0 , keepdims = True))
  return f / np.sum(f,axis = 0, keepdims = True)

def relu(z):
   return np.maximum(0,z)

def random_init(layers_dim):
  np.random.seed(0)
  params = {}

  L = len(layers_dim) - 1

  for i in range(1, L+1):
     params["W" + str(i)]= np.random.randn(layers_dim[i], layers_dim[i-1]) * np.sqrt(2. / layers_dim[i-1])
     params["b" + str(i)]= np.zeros((layers_dim[i],1))
  return params

def forward(X, params):

  z1 = params["W1"] @ X + params["b1"]
  a1 = relu(z1)

  z2 = params["W2"] @ a1 + params["b2"]
  a2 = relu(z2)

  z3 = params["W3"] @ a2 + params["b3"]
  a3 = softmax(z3)


  cache = {"X": X,
           "A1":a1, "A2":a2, "A3":a3,
           "Z1":z1, "Z2": z2, "Z3": z3
           }
  y_pred = a3
  return y_pred, cache

# Cross Entropy Loss
def crossentropy(y_pred, y_true):
   sample =-np.sum(y_true * np.log(y_pred + 1e-8) , axis = 0)
   loss = np.mean(sample)

   return loss

# ReLU activation derivative
def relu_derivative(Z):
  return Z > 0

# Backward propagation
def backward(y_true, cache, params):
  m = y_true.shape[1]
  dz3 = cache["A3"] - y_true
  dw3 = (1/m) * dz3 @ cache["A2"].T
  db3 = np.sum((1/m) * dz3, axis = 1, keepdims = True)

  da2 = params["W3"].T @ dz3
  dz2 = da2 * relu_derivative(cache["Z2"])
  dw2 = (1/m) * dz2 @ cache["A1"].T
  db2 = np.sum((1/m) * dz2, axis = 1, keepdims = True)

  da1 = params["W2"].T @ dz2
  dz1 = da1 * relu_derivative(cache["Z1"])
  dw1 = (1/m) * dz1 @ cache["X"].T
  db1 = np.sum((1/m) * dz1, axis = 1, keepdims = True)

  gradients = { "dW3":dw3, "db3":db3,
  "dW2":dw2, "db2":db2,
  "dW1":dw1, "db1":db1 }

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
     y_pred, cache = forward(X_batch, params)

     # computing loss
     loss = crossentropy(y_pred, y_batch)
     epoch_loss += loss
     #print(f"Epochs : {epoch+1}, Loss: {loss:.4f}")

     # backward prop
     gradients = backward(y_batch, cache, params)

     # update weights
     params["W1"] -= lr*gradients["dW1"]
     params["b1"] -= lr*gradients["db1"]

     params["W2"] -= lr*gradients["dW2"]
     params["b2"] -= lr*gradients["db2"]

     params["W3"] -= lr*gradients["dW3"]
     params["b3"] -= lr*gradients["db3"]
    print(f"Epochs : {epoch+1}, Loss: {epoch_loss:.4f}")
    return params

# Accuracy Check
def accuracy(X, y_true, params):
  y_pred, _ = forward(X, params)
  preds = np.argmax(y_pred, axis=0)
  return np.mean(preds == y_true)