import numpy as np
from keras.datasets import mnist
from utils import onehot, accuracy, random_init, forward, backward, crossentropy
from dataloader.batch_strategy import batch_strategy
from activation_func.activations import relu, softmax
from optim.Adam import Adam

# Variables
batch_size = 128
epochs = 100
lr = 0.1
activations = [relu, relu, softmax]

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0
y_train_oh = onehot(y_train, 10)

# Parameters
layers_dim = [784, 128, 64, 10]
params = random_init(layers_dim)
optimizer = Adam(lr, params, gradients, 1e-8)

# Training loop 
for epoch in range(epochs):
    epoch_loss = 0.0
    for X_batch, y_batch in batch_strategy(X_train, y_train_oh, batch_size):
        # forward pass
        y_pred, cache = forward(X_batch, params, activations)

        # computing loss
        loss = crossentropy(y_pred, y_batch)
        epoch_loss += loss

        # backward prop
        gradients = backward(y_batch, cache, params, activations)

        # update weights
        params = optimizer.step()
     
    print(f"Epochs : {epoch+1}, Loss: {epoch_loss:.4f}")

# Evaluate
test_acc = accuracy(X_test, y_test, params, activations)
train_acc = accuracy(X_train, y_train, params, activations)

print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Train Accuracy: {train_acc * 100:.2f}%")