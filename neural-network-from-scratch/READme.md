# From Scratch Projects

Version: v1.1.0

A from-scratch machine learning playground for research papers, core algorithms, and neural network building blocks.

The goal is simple: build the machinery yourself, understand every moving part, and keep the implementation clean enough to inspect without guesswork.

* * *

## Current State

Component Status  
Neural network primitives In progress  
Activation functions In place  
Data loading utilities In place  
Optimizer layer In progress  
Gradient checking Available  
Notebook experiments In progress  
Project documentation Pending  

* * *

## System Pipeline

    Research paper / algorithm idea
          ↓
    neural-network-from-scratch/
          ↓
    activation_func/
          ↓
    dataloader/
          ↓
    optim/
          ↓
    nn.py
          ↓
    gradient_checker.py
          ↓
    notebooks/
          ↓
    main.py
          ↓
    utils.py

* * *

## Architecture

The repo is split into a few practical layers.

Core Layer `nn.py`  
Holds the model logic. Forward pass, backward pass, parameter flow, and the pieces needed to keep the network state explicit.

Activation Layer `activation_func/`  
Contains the non-linearities. Small, isolated, and easy to swap.

Data Layer `dataloader/`  
Handles data movement and batching. No hidden magic.

Optimization Layer `optim/`  
Keeps the update rules separate from the model itself.

Verification Layer `gradient_checker.py`  
Used to validate analytical gradients against numerical gradients before training gets trusted.

Experiment Layer `notebooks/`  
Where ideas get tested fast, then moved into the core implementation once they are worth keeping.

* * *

## Design Principles

This is not a framework.  
This is not a black box.  
This is a lab bench.

Every file exists to make the implementation easier to read, reason about, and extend.

* * *

## Project Structure

    from-scratch-projects/
    ├── neural-network-from-scratch/
    │   ├── activation_func/
    │   ├── dataloader/
    │   ├── notebooks/
    │   ├── optim/
    │   ├── .gitignore
    │   ├── gradient_checker.py
    │   ├── main.py
    │   ├── nn.py
    │   └── utils.py
    └── README.md

* * *

## What’s Next

- Expand the neural network implementation
- Add more algorithms from scratch
- Turn notebook experiments into reusable modules
- Add tests for core math and gradient flow
- Keep the codebase small, explicit, and easy to audit

* * *

## About

Here, I'll upload research papers and algorithms from scratch.
