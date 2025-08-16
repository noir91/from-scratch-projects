import numpy as np
from utils import forward, random_init, backward
from numpy import linalg as LA
from keras.datasets import mnist
from utils import crossentropy
import copy

class GradCheck:
    """
    gradient checker used for checking backpropagation gradients.  
    
    parameters:
        
        - X = Input X Array
        - Y = True labels
        - epsilon = margin of error, set to 1e-7 by default
        - grad = true backprop output values
        - parameters = cached parameters from the network i.e W1, b1, Z1, A1 ..... Wn, bn, Zn, An
        - layers_dim = Layers within the network

    performing numerical gradient approximation:
        formula: 
                gradient approx : (theta + epsilon) - (theta - epsilon) / 2 * epsilon
                checking grad diff : 
                        np.linalg.norm(grad, ord = 2) / np.linalg.norm(grad, ord = 2) + np.linalg.norm(grad_approx, ord = 2)
                
    If the gradient difference is lesser than 10e-7 GREAT!
    If the gradient difference is lesser than 10e-5, It's okay but should be better
    If the gradient difference is lesser than 10e-3, The backprop didn't run correctly,
        the gradients at disposal are wrongly caclulated.
    """

    def __init__(self, X, Y, layers_dim, epsilon):
        self.layers_dim = layers_dim
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.layers_dim = layers_dim

    def initialize_params(self):
        parameters = random_init(self.layers_dim)
        return parameters
    
    def dictionary_to_vectors(self, parameters):
        cache = []
        for _, parameter_values in parameters.items():
            flat_mat = parameter_values.flatten()
            cache.append(flat_mat)
        
        return np.concatenate(cache)
    
    def vectors_to_dictionary(self, vector):
        parameters = {}
        start = 0
        for l in range(1, len(self.layers_dim)):
            W_shape = (self.layers_dim[l], self.layers_dim[l-1])
            W_size = np.prod(W_shape)
            b_shape = (self.layers_dim[l], 1)
            b_size = np.prod(b_shape)

            parameters[f"W{l}"] = vector[start:start+W_size].reshape(W_shape)
            start += W_size
            parameters[f"b{l}"] = vector[start:start+b_size].reshape(b_shape)
            start += b_size

        return parameters

    def gradients_zero_like(self, grad_true, parameter_vectorized, epsilon, verbose= True):
        # compute numerical gradient for each parameter
        gradapprox = np.zeros_like(grad_true)
        total_params = parameter_vectorized.shape[0]
        original_parameters = copy.deepcopy(parameter_vectorized)
        count = 0
        count_down = max(1, grad_true.shape[0] // 100)

        for i in range(gradapprox.shape[0]):

            # theta nudged up by epsilon
            theta_plus = copy.deepcopy(original_parameters)

            theta_plus[i] += epsilon
            parameters_plus = self.vectors_to_dictionary(theta_plus)
            y_pred, _ = forward(self.X, parameters_plus)
            J_plus = crossentropy(y_pred = y_pred, y_true = self.Y) # obtaining J( theta + epsilon )
        
            # theta nudged down by epsilon
            theta_minus = copy.deepcopy(original_parameters)

            theta_minus[i] -= epsilon
            parameters_minus = self.vectors_to_dictionary(theta_minus)
            y_pred, _ = forward(self.X, parameters_minus)
            J_minus = crossentropy(y_pred = y_pred, y_true = self.Y) # obtaining J( theta - epsilon)

            # calculating numerical gradients for ith
            gradapprox[i] = (J_plus - J_minus) / (2*epsilon)

            # verbose
            if verbose == True:
                if i % 100 == 0:
                    count += 1
                    count_down -=1
                    grad_true_subset = grad_true[:i+1]
                    gradapprox_subset = gradapprox[:i+1]
                    numerator = LA.norm((gradapprox_subset - grad_true_subset), ord = 2 )
                    denominator = LA.norm(gradapprox_subset, ord = 2) + LA.norm(grad_true_subset, ord = 2)
                    diff = numerator/denominator
                    
                    print(f"Gradient difference at iteration {i}: {diff:.6e}")
                    #print(f"Grad diff without norm {grad_true_subset - gradapprox_subset}")
                    print(f"Processed {i} / {total_params} parameters")
                    print("#" * count + "_"*count_down)

        return gradapprox 
            
    def gradient_checker(self):

        # loading in randomly intialized parameters
        parameters = self.initialize_params()

        # converting dictionary to vectors
        parameters_vectorized = self.dictionary_to_vectors(parameters)
    
        # caching original forward prop parameters for backprop gradients at next step
        parameters_original = self.vectors_to_dictionary(parameters_vectorized)
        _, cache = forward(self.X, parameters_original)

        # caching backprop gradients 
        grad = backward(y_true = self.Y, cache = cache, params= parameters)
        grad_true = self.dictionary_to_vectors(grad)
        
        gradapprox = self.gradients_zero_like(grad_true= grad_true, parameter_vectorized= parameters_vectorized, epsilon= self.epsilon)
        #gradapprox = np.array([gradapprox]) # converted into a numpy array to solve shape problems during broadcasting
        
        # gradient numerical approximation
        numerator = LA.norm((gradapprox - grad_true), ord = 2 )
        denominator = LA.norm(gradapprox, ord = 2) + LA.norm(grad_true, ord = 2)
        diff = numerator/denominator
        
        return diff
