import numpy as np
from keras.datasets import mnist
from utils import onehot, train, accuracy

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0

# Parameters
layers_dim = [784, 128, 64, 10]
y_train_oh = onehot(y_train, 10)

# Train the model
trained_params = train(0.1, X_train, y_train_oh, layers_dim, 100)

# Evaluate
test_acc = accuracy(X_test, y_test, trained_params)
train_acc = accuracy(X_train, y_train, trained_params)

print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Train Accuracy: {train_acc * 100:.2f}%")