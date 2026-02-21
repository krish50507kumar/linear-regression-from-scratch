# linear-regression-from-scratch
Multivariate linear regression implemented from scratch using vectorized gradient descent in NumPy. Includes manual cost computation, gradient derivation, feature scaling, and convergence analysis.


📈 Linear Regression from Scratch (NumPy Implementation)
Overview

This repository contains a from-scratch implementation of multivariate linear regression using:

Batch Gradient Descent

Mean Squared Error (MSE) loss

Analytical gradient derivation

Feature standardization

Log-scale convergence visualization

No ML libraries (e.g., scikit-learn) were used.
All computations are fully vectorized using NumPy.

Objective

We minimize the Mean Squared Error:

J(W) = (1/m) * Σ (y - ŷ)²

Where:

ŷ = XW

The goal is to learn optimal weights W using gradient descent.

Gradient

The analytical gradient of MSE with respect to weights:

dJ/dW = (2/m) * Xᵀ (XW - Y)

Update rule:

W := W - α * dJ/dW

Where:

α = learning rate

m = number of samples

Implementation Highlights

Fully vectorized (no loops over training samples)

Explicit bias term

Feature scaling (excluding bias column)

Log-scale cost visualization

Manual weight initialization

Feature Scaling

All features (except bias term) are standardized:

X_scaled = (X - mean) / std

This improves convergence speed and numerical stability.

Convergence Behavior

The cost curve is plotted using a logarithmic scale to reveal exponential decay in early training stages.

The rapid initial decrease occurs because gradients are large when weights are far from the optimum.

Tech Stack

Python 3

NumPy

Matplotlib

Why This Project?

This project was built to:

Understand gradient-based optimization at a mathematical level

Develop intuition for convergence dynamics

Avoid black-box ML abstractions

Strengthen linear algebra foundations

Possible Extensions

Early stopping

Learning rate scheduling

Mini-batch gradient descent

Normal equation comparison

L2 regularization (Ridge)

Author

Krish
Focused on first-principles machine learning and systems-level thinking.
