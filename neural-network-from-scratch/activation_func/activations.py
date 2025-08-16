import numpy as np

def relu_derivative(Z):
  return (Z > 0).astype(float)

def sigmoid(z):
  f = 1/(1+np.exp(-z))
  return f

def softmax(z):
  f = np.exp(z - np.max(z, axis = 0 , keepdims = True))
  return f / np.sum(f,axis = 0, keepdims = True)

def relu(z):
   return np.maximum(0,z)