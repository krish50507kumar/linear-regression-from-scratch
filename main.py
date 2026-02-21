import numpy as np
import matplotlib.pyplot as plt

def cost_function(Yr,Yp):
    """
    Computes Mean Squared Error (MSE).

    Optimization Objective:
        J(W) = (1/m) * Σ (Yr - Yp)^2

    This is a convex loss function for linear regression,
    guaranteeing a single global minimum.
    """
  
    cost=np.mean((Yr-Yp)**2)
    return cost
    
def update_W(Yr,Yp,L,W,X):
    m = len(Yr)

    # Analytical gradient of MSE with respect to weights:
    # ∂J/∂W = (2/m) * X^T (Yp - Yr)
    # Vectorized form avoids explicit loops for computational efficiency
  
    Wd = (2/m) * (X.T @ (Yp - Yr))

    # Gradient descent update rule:
    # W := W - α * ∂J/∂W
    W = W - L * Wd
  
    return W
    
def training(Yr,W,X,L,I):
    """
    Performs batch gradient descent.

    Iteratively updates weights to minimize MSE.
    Convergence behavior depends on learning rate (L)
    and feature scaling.
    """
  
    cost_history=[]
    for i in range(1,I+1):
        # Forward pass: compute predictions using current weights
        Yp = X @ W
        W=update_W(Yr,Yp,L,W,X)
        Yp_new = X @ W
        cost = cost_function(Yr, Yp_new)
        cost_history.append(cost)
        if i%10==0:
            print(f"log:{i} \t W:{W} \t cost:{cost}")
    return W,cost_history


W=np.array( [1,0.5,0.5,0.5] )
X=np.array(
    [
        [1, 230.1 ,37.8 ,69.1],
        [1,44.5 ,39.3 ,23.1],
        [1,17.2 ,45.9 ,34.7],
        [1,151.5 ,41.3 ,13.2]
    ]
)

sX=X.copy()
# Standardize features (excluding bias term at column 0)
# Improves convergence speed and prevents gradient instability
sX[:,1:]=(X[:,1:]-X[:,1:].mean(axis=0))/X[:,1:].std(axis=0)

Yr=np.array([22.1,10.4,18.3,18.5])

learning_rate = 0.05
iters=1000
W,Cost_history=training(Yr,W,sX,learning_rate,iters)

x = np.arange(1, len(Cost_history) + 1)

plt.figure(figsize=(10, 6))
plt.plot(x, Cost_history)
# Log scale reveals exponential decay pattern of gradient descent
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent Convergence')
plt.show()
