import numpy as np
from utils import forward, backward, crossentropy, update_weights, random_init

class MiniBatchGD:

    def __init__(self, X, Y, lr, batch_size, layers_dim, epochs):
        self.lr = lr
        self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.layers_dim = layers_dim
        self.epochs = epochs

    def minibatch_gd(self):
        
        params = random_init(self.layers_dim)
        # Training examples 'm'
        m = self.X.shape[1]
        itr = m // self.batch_size
        epoch_loss_list = []
        for epoch in range(self.epochs):
    
            # np.random.permutation for unbiased estimate of the gradient estimate (random mini batches)   
            indices = np.random.permutation(m)

            X_shuffled = self.X[:, indices] 
            y_shuffled = self.Y[:, indices]

            epoch_loss = 0.0 
            epoch_loss_list.append(epoch_loss)
            # Mini-Batch Gradient Descent training loop
            for i in range(itr):
                start = i * self.batch_size
                end = start + self.batch_size            

                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]

                # forward propagation
                y_pred, cache = forward(X_batch, params)

                # compute loss
                loss = crossentropy(y_pred, y_batch)
                epoch_loss += loss

                # backward propagation
                gradients = backward(y_batch, cache, params)

                # update weights
                params = update_weights(self.lr, params, gradients)
            print(f" Epoch : {epoch+1}, Epoch Loss : {epoch_loss:.4f}")       
        return params, epoch_loss_list
    
