# PyTorch vs. TensorFlow vs. JAX: A Simple Model Comparison

Here is a comparison of PyTorch, TensorFlow, and JAX, complete with code implementations of a simple linear regression model to highlight their core differences.

### Core Differences at a Glance

| Feature | PyTorch | TensorFlow | JAX |
|---|---|---|---|
| **API Style** | Object-oriented, Pythonic | High-level (Keras) & low-level | Functional, NumPy-like |
| **Graph Building** | Dynamic (defined by execution) | Eager execution, with static graph compilation (`tf.function`) | Just-In-Time (JIT) compilation of functions (`@jit`) |
| **Mutability** | Tensors are mutable | Variables are mutable, Tensors are immutable | Arrays are immutable |
| **Primary Use Case**| Research and rapid prototyping | Production and scalability | High-performance research |
| **Ecosystem** | Mature, strong community support | Extensive, with tools for deployment and production | Growing, focused on research and high-performance computing |

---

### Simple Model: Linear Regression

We will implement a simple linear regression model, which aims to find the best-fit line (`y = w * x + b`) for a given set of data points. This will involve:
1.  Defining the model.
2.  Defining a loss function (Mean Squared Error).
3.  Calculating gradients.
4.  Updating the model's parameters (weights `w` and bias `b`).

#### Synthetic Data Setup

```python
import numpy as np

# Generate some noisy data
X_raw = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float32)
y_raw = np.array([3.5, 6.3, 8.9, 12.1, 15.2, 18.4], dtype=np.float32)

# Normalize the data for better training
X = (X_raw - X_raw.mean()) / X_raw.std()
y = (y_raw - y_raw.mean()) / y_raw.std()
```

---

### PyTorch Implementation

PyTorch is known for its imperative style and object-oriented approach, making it feel very "Pythonic".

```python
import torch
import torch.nn as nn

# Convert data to PyTorch Tensors
X_torch = torch.from_numpy(X).view(-1, 1)
y_torch = torch.from_numpy(y).view(-1, 1)

# 1. Define the model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# 2. Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(X_torch)

    # Calculate loss
    loss = loss_fn(y_pred, y_torch)

    # Backward pass and optimization
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()      # Compute gradients
    optimizer.step()     # Update weights

# Get the learned parameters
w_pytorch, b_pytorch = model.parameters()
print(f"PyTorch -- w: {w_pytorch.item():.4f}, b: {b_pytorch.item():.4f}")
```

#### Explanation of Core Differences in PyTorch:

*   **Object-Oriented:** The model is defined as a class that inherits from `nn.Module`. This class holds the model's layers and its state (parameters).
*   **Mutable Parameters:** The model's parameters (`model.parameters()`) are mutable and are updated in-place by the `optimizer.step()` call.
*   **Dynamic Computation Graph:** The forward pass (`model(X_torch)`) dynamically builds a computation graph. The call to `loss.backward()` then traverses this graph backward to compute gradients.
*   **Integrated Autograd and Optimizers:** PyTorch's `autograd` engine tracks operations to compute gradients, and the `torch.optim` module provides a wide range of optimizers that handle parameter updates.

---

### TensorFlow Implementation

TensorFlow, through its high-level API Keras, provides a similar object-oriented feel to PyTorch, but with a strong emphasis on building scalable and deployable models.

```python
import tensorflow as tf

# Data is already in NumPy format, which TensorFlow can use directly

# 1. Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# 2. Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='mean_squared_error')

# 3. Training
model.fit(X, y, epochs=100, verbose=0)

# Get the learned parameters
w_tf, b_tf = model.layers[0].get_weights()
print(f"TensorFlow -- w: {w_tf[0][0]:.4f}, b: {b_tf[0]:.4f}")
```

#### Explanation of Core Differences in TensorFlow:

*   **High-Level API (Keras):** The most common way to build models in TensorFlow is with Keras, which provides a very user-friendly and abstracted experience. The `model.compile()` and `model.fit()` methods abstract away the training loop.
*   **Static Graph Optimization:** While TensorFlow 2.x uses eager execution by default, calling `model.fit()` or wrapping code in `tf.function` allows TensorFlow to create and optimize a static computation graph, which can lead to better performance.
*   **Ecosystem:** TensorFlow offers a comprehensive ecosystem with tools like TensorBoard for visualization and TensorFlow Serving for easy deployment, making it a strong choice for production environments.

---

### JAX Implementation

JAX's approach is functional, closely resembling NumPy. It encourages writing pure functions that are then transformed by JAX's capabilities like automatic differentiation and JIT compilation.

```python
import jax
import jax.numpy as jnp

# Convert data to JAX arrays
X_jax = jnp.array(X).reshape(-1, 1)
y_jax = jnp.array(y).reshape(-1, 1)

# 1. Define the model as a function
def model(params, x):
    return x @ params['w'] + params['b']

# 2. Define the loss function as a function
def loss_fn(params, x, y):
    y_pred = model(params, x)
    return jnp.mean((y_pred - y)**2)

# 3. Define the update step
@jax.jit
def update_step(params, x, y, lr):
    grads = jax.grad(loss_fn)(params, x, y)
    return {p: params[p] - lr * grads[p] for p in params}

# Initialize parameters
key = jax.random.PRNGKey(0)
params = {
    'w': jax.random.normal(key, (1, 1)),
    'b': jnp.zeros((1,))
}

# Training loop
for epoch in range(100):
    params = update_step(params, X_jax, y_jax, lr=0.01)

# Get the learned parameters
print(f"JAX -- w: {params['w'][0][0]:.4f}, b: {params['b'][0]:.4f}")
```

#### Explanation of Core Differences in JAX:

*   **Functional Programming:** The model, loss, and update logic are all defined as pure functions. There are no classes or internal states.
*   **Immutable Data Structures:** JAX arrays and parameters are immutable. The `update_step` function does not modify the existing parameters; it returns a new set of updated parameters.
*   **Explicit State Management:** The parameters (`params`) are explicitly passed into and returned from functions. This is a core concept in functional programming.
*   **Function Transformations:** JAX's power comes from its function transformations:
    *   `jax.grad()`: Transforms a function into another function that computes its gradients.
    *   `jax.jit()`: Just-In-Time compiles a function for high-performance execution on accelerators like GPUs and TPUs.