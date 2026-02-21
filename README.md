# 📈 Linear Regression from Scratch (NumPy Implementation)

**Multivariate linear regression implemented entirely from scratch using vectorized gradient descent in NumPy. Includes manual cost computation, gradient derivation, feature scaling, and convergence analysis.**

## 📖 Overview

This repository contains a from-scratch implementation of multivariate linear regression. No high-level machine learning libraries (like scikit-learn) were used. All computations are fully vectorized using NumPy to ensure optimal performance and to demonstrate the underlying linear algebra of the algorithm.

**Key Features:**
* Batch Gradient Descent
* Mean Squared Error (MSE) loss
* Analytical gradient derivation
* Feature standardization
* Log-scale convergence visualization

---

## 🧮 Mathematical Formulation

### Objective
We minimize the Mean Squared Error (MSE) cost function:

$$J(W) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

Where the predicted values are computed as:

$$\hat{y} = XW$$

The goal is to learn the optimal weight vector $W$ using gradient descent.

### Gradient & Update Rule
The analytical gradient of the MSE with respect to the weights is derived as:

$$\frac{\partial J}{\partial W} = \frac{2}{m} X^T (XW - Y)$$

The weights are iteratively updated using the following rule:

$$W := W - \alpha \frac{\partial J}{\partial W}$$

Where:
* $\alpha$ = learning rate
* $m$ = number of training samples

### Feature Scaling
To improve convergence speed and ensure numerical stability, all features (excluding the bias term) are standardized:

$$X_{scaled} = \frac{X - \mu}{\sigma}$$

---

## ⚙️ Implementation Highlights

* **Fully Vectorized:** No `for` loops are used over the training samples, ensuring efficient matrix operations.
* **Explicit Bias Term:** The bias is manually appended as a column of ones to the feature matrix.
* **Feature Scaling:** Standardization is applied to prevent features with larger magnitudes from dominating the gradient.
* **Log-Scale Cost Visualization:** The cost curve is plotted using a logarithmic scale to clearly reveal the exponential decay in early training stages.
* **Manual Weight Initialization:** Weights are explicitly initialized before training begins.

## 📉 Convergence Behavior

* The rapid initial decrease in cost occurs because gradients are exceptionally large when the initialized weights are far from the global optimum.
* Visualizing the cost on a logarithmic scale provides a clearer picture of the model's convergence dynamics over time.

---

## 🛠️ Tech Stack

* **Python 3**
* **NumPy** (for all matrix and vector operations)
* **Matplotlib** (for convergence and data visualization)

---

## 🧠 Why This Project?

This project was built to step away from black-box ML abstractions and build a deep, foundational understanding of how these algorithms actually work. The primary goals were to:
* Understand gradient-based optimization at a strict mathematical level.
* Develop a strong intuition for convergence dynamics and hyperparameter tuning.
* Strengthen foundational linear algebra concepts by applying them in code.

---

## 🚀 Possible Extensions

- [ ] Early stopping implementation
- [ ] Learning rate scheduling / decay
- [ ] Mini-batch gradient descent
- [ ] Normal equation implementation for direct comparison
- [ ] L2 regularization (Ridge Regression)

---

## 👨‍💻 Author

**Krish** *Focused on first-principles machine learning, deep learning, and systems-level thinking.*
