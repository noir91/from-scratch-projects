import numpy as np


# Batch Strategy function 
def batch_strategy(X, Y, batch_size):
    """

    This function is designed to control the training examples which will go
    into the training of the neural network.

    If batch_size == 'full':

        We will use the entire batch size. 
        Advantages:
        - Less stochastic and noisy
        - Converges to the global minima

        Disadvantages:
        - Slow convergence
        - Computationally expensive

    If batch_size > 2:

        We will use mini batches of training examples.
        Advantages:
        - Faster learning
        - Less computationally expensive, shows progress even before the entire batch
        is processed.

        Disadvantages: 
        - It is noisy in it's movement towards the descent, oscillations depend
        on the batch size.

        Note: Strongly advices to use 2**x.
    If batch_size == 1:

        We will use Stochastic gradient descent strategy, or online training, where
        a single example is processed to find the gradients towards the steepest descent.

        Advantages:
        - Can use a higher learning rate
        - Get's out of a local minima

        Disadvantages: 
        - Very stochastic, never converges to the global minima, but oscillates around it.

    """ 
    m = X.shape[1s]
    num_batches = m // batch_size
    indices = np.random.permutation(m)

    X_shuffled = X[:, indices]
    Y_shuffled = Y[:, indices]

    if batch_size == m:
        yield X_shuffled, Y_shuffled
    
    elif batch_size == 1:

        for i in range(m):
            X_sgd = X_shuffled[:, i:1+i]
            Y_sgd = Y_shuffled[:, i:1+i]
            yield X_sgd, Y_sgd        

    else:
        for i in range(num_batches):
            start = i * batch_size 
            end = start + batch_size

            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]
            yield X_batch, Y_batch
        

        if m % batch_size != 0:
            yield X_shuffled[:, num_batches*batch_size:], Y_shuffled[:, num_batches*batch_size:]
