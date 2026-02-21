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

Minimize the Mean Squared Error:

𝐽
(
𝑊
)
=
1
𝑚
∑
𝑖
=
1
𝑚
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
J(W)=
m
1
	​

i=1
∑
m
	​

(y
i
	​

−
y
^
	​

i
	​

)
2

Where:

𝑦
^
=
𝑋
𝑊
y
^
	​

=XW

The goal is to learn optimal weights 
𝑊
W using gradient descent.

Gradient Derivation

The analytical gradient of MSE with respect to weights:

∂
𝐽
∂
𝑊
=
2
𝑚
𝑋
𝑇
(
𝑋
𝑊
−
𝑌
)
∂W
∂J
	​

=
m
2
	​

X
T
(XW−Y)

Update rule:

𝑊
:
=
𝑊
−
𝛼
∂
𝐽
∂
𝑊
W:=W−α
∂W
∂J
	​


Where:

𝛼
α = learning rate

𝑚
m = number of samples

Implementation Highlights

Fully vectorized (no explicit loops over samples)

Bias term handled explicitly

Feature standardization (excluding bias column)

Log-scale cost plot to analyze convergence behavior

Manual weight initialization

Feature Scaling

All features (except bias term) are standardized:

𝑋
𝑠
𝑐
𝑎
𝑙
𝑒
𝑑
=
𝑋
−
𝜇
𝜎
X
scaled
	​

=
σ
X−μ
	​


This improves convergence stability and prevents gradient explosion/divergence.

Convergence Behavior

The cost is plotted on a logarithmic scale to visualize exponential decay during early training phases.

Initial rapid decrease occurs due to large gradient magnitude when weights are far from the optimum.

Tech Stack

Python 3

NumPy

Matplotlib

Why This Project?

This implementation was built to:

Understand optimization mechanics at a mathematical level

Develop intuition for gradient-based learning

Avoid black-box ML abstractions

Strengthen linear algebra + numerical computation skills

Possible Extensions

Early stopping criteria

Learning rate scheduling

Mini-batch gradient descent

Normal equation comparison

Regularization (L2 / Ridge)

Author

Krish
Aspiring ML engineer focused on first-principles understanding of machine learning systems.
