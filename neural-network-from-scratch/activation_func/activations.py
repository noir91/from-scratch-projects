import numpy as np

def relu_derivative(Z):
  return (Z > 0).astype(float)

def sigmoid(z):
  f = 1/(1+np.exp(-z))
  return f

def softmax(z):
  f = z - np.max(z, axis = 1 , keepdims = True) # changed axis from 0 to 1
  expz = np.exp(f)
  return expz / np.sum(expz,axis = 1, keepdims = True) # changed axis from 0 to 1 

def relu(z):
   return np.maximum(0,z)

def sigmoid_derviative(z):
  return sigmoid(z) * (1 - sigmoid(z))

# ============ LOSS FUNCTIONS ===============
def cross_entropy_from_logits(Z, y_true):
  B = Z.shape[0]

  # Softmax with numerical stability
  Z_shift = Z -np.max(Z, axis =1, keepdims= True)
  expZ = np.exp(Z_shift)
  P = expZ / np.sum(expZ, axis =1, keepdims = True)

  # Loss
  correct_logprobs = -np.log(P[np.arange(B), y_true] + 1e-12) # numerical stability added make linkedin
  loss = np.mean(correct_logprobs)

  # Gradient
  dZ = P.copy()
  dZ[np.arange(B), y_true] -=1
  dZ /=B
  return loss, dZ

def crossentropy(y_pred, y_true):
    
  Z_shift = Z - np.max(Z, axis = 1, keepdims= True)
  logsumexp = np.log(np.sum(np.exp(Z_shift), axis = 1, keepdims= True))
  correct_logprobs = Z_shift[np.arange(len(y)), y] - logsumexp.squeeze()
  loss = -np.mean(correct_logprobs)
  return loss
  
  #eps = 1e-12
  #y_pred = np.clip(y_pred, eps, 1.0 - eps)
  #sample =-np.sum(y_true * np.log(y_pred) , axis = 1) # changed axis from 0 to 1
  #loss = np.mean(sample)
  #return loss

def softmax_crossentropy_backward(Z, y):
  """
  Gradients of softmax + cross_entropy wrt logits
  Z: (B, C) logits
  y: (B,) integer labels
  Returns: (B, C) dZ
  """
  B = Z.shape[0]
  P = softmax(Z)
  dZ = P.copy()
  dZ[np.arange(B), y] -= 1
  dZ /=B
  return dZ
