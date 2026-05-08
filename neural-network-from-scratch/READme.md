# From Scratch Projects

Version: v1.1.0

A from-scratch machine learning playground for neural networks, research papers, and core algorithms.

The goal is to keep the implementation explicit, inspectable, and small enough that every step in the pipeline can be traced by hand.

* * *

## Current State

Component Status
Neural network core In progress
Activation functions In place
Data loading In place
Optimizer layer In place
Gradient checking Available
Notebook experiments In progress
Documentation In progress

* * *

## System Pipeline

    MNIST / experimental input
          ↓
    dataloader.batch_strategy
          ↓
    activation_func.activations
          ↓
    nn.py / utils.py
          ↓
    optim/
          ↓
    main.py
          ↓
    gradient_checker.py
          ↓
    notebooks/

The repo is intentionally manual. No framework hides the math.

* * *

## What This Is

This project is for building neural network mechanics from scratch.

That means:
- parameter initialization
- batching
- forward propagation
- activation functions
- loss computation
- backpropagation
- optimizer updates
- gradient verification

The point is not convenience. The point is understanding.

* * *

## How To Run It

The training entrypoint lives inside:

    neural-network-from-scratch/main.py

The script expects to be run from inside that folder, because it uses local imports such as `utils`, `dataloader.batch_strategy`, `activation_func.activations`, and `optim.Adam`.

### Setup

Clone the repository:

    git clone https://github.com/noir91/from-scratch-projects.git
    cd from-scratch-projects/neural-network-from-scratch

Create a virtual environment:

    python -m venv .venv

Activate it:

    # macOS / Linux
    source .venv/bin/activate

    # Windows
    .venv\Scripts\activate

Install dependencies:

    pip install numpy tensorflow jupyter

If your environment uses standalone Keras instead of TensorFlow Keras:

    pip install keras

### Run

Start the training script:

    python main.py

Or open the notebooks:

    jupyter notebook

Then inspect:

    notebooks/neural-network_v1.ipynb
    notebooks/neural-network_v2.ipynb
    notebooks/tests.ipynb

* * *

## Architecture

The codebase is split into a few small layers.

Model Layer `nn.py`  
Contains the neural network class and the explicit forward / backward logic.

Utility Layer `utils.py`  
Holds helpers for one-hot encoding, accuracy, parameter initialization, forward propagation, backward propagation, and loss.

Activation Layer `activation_func/activations.py`  
Contains the activation functions and the cross-entropy-from-logits helper.

Batching Layer `dataloader/batch_strategy.py`  
Handles the batch iteration strategy used by training.

Optimizer Layer `optim/`  
Contains the parameter update rules, including Adam.

Verification Layer `gradient_checker.py`  
Used to compare analytical gradients against numerical gradients.

Experiment Layer `notebooks/`  
Used for quick iteration before moving logic into the main code path.

* * *

## Quick Reference

Function Module What it does
`onehot` `utils.py` Converts labels to one-hot vectors

`accuracy` `utils.py` Computes model accuracy

`random_init` `utils.py` Initializes weights and biases

`forward` `utils.py` Runs the forward pass

`backward` `utils.py` Computes gradients

`cross_entropy_from_logits` `activation_func/activations.py` Computes loss and gradient from logits

`batch_strategy` `dataloader/batch_strategy.py` Produces full, mini, or stochastic batches

`relu` `activation_func/activations.py` ReLU activation

`softmax` `activation_func/activations.py` Output activation

`Adam` `optim/Adam.py` Optimizer used for parameter updates

`gradient_checker` `gradient_checker.py` Gradient validation utility

* * *

## Training Flow

The current training loop is simple and direct.

    Load MNIST
         ↓
    Flatten and normalize images
         ↓
    One-hot encode labels
         ↓
    Initialize parameters
         ↓
    Slice data into batches
         ↓
    Forward pass
         ↓
    Compute cross-entropy loss
         ↓
    Backward pass
         ↓
    Update parameters
         ↓
    Evaluate train and test accuracy

The model currently uses a small fully connected network with a [784, 128, 64, 10] layout.

* * *

## Notes

I made neural networks from scratch with optimizers, batch normalization, deep layers, gradient checking for back propagation check, to externalize theoritcal knowledge learned in Deep Learning Specialization.

If you hit a runtime issue, check the main training loop and helper imports first. The repo is modular and structured for iteration.

* * *
