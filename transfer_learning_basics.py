"""
Transfer Learning - Basic Python Demonstration
==============================================
Matches the matrix worked example exactly.
Source task : Dublin temperature data  (pre-trained weights)
Target task : Cork temperature data    (fine-tuning)

Network architecture:
  Input  (2,) --> Hidden layer (2 neurons) --> Output (1,)
  W1: 2x2  <-- FROZEN during transfer
  W2: 1x2  <-- RETRAINED on Cork data
"""

import numpy as np

# ─────────────────────────────────────────────
# STEP 1 : Pre-trained weights (learned from Dublin)
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1 : Pre-trained weights (from Dublin)")
print("=" * 55)

W1 = np.array([[ 0.8,  0.2],   # Hidden layer weights  (FROZEN)
               [-0.5,  0.6]])

W2 = np.array([[0.7,  0.4]])    # Output layer weights  (will be retrained)

print(f"W1 (frozen):\n{W1}")
print(f"W2 (initial):\n{W2}\n")


# ─────────────────────────────────────────────
# STEP 2 : Cork input data  (normalised 0-1)
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 2 : Cork input x")
print("=" * 55)

x = np.array([[0.6],   # x1 = today 9am temperature
              [0.3]])  # x2 = today 3pm temperature

y_true = np.array([[0.5]])   # actual Cork tomorrow temperature (normalised)

print(f"x  = {x.T}  (normalised temperatures)")
print(f"y  = {y_true}  (actual Cork tomorrow)\n")


# ─────────────────────────────────────────────
# STEP 3 : Forward pass - hidden layer
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 3 : Forward pass  z1 = W1 · x")
print("=" * 55)

z1 = W1 @ x   # matrix multiply: (2x2) @ (2x1) = (2x1)

print(f"z1[1] = (0.8 × 0.6) + (0.2 × 0.3) = {z1[0][0]:.4f}")
print(f"z1[2] = (-0.5× 0.6) + (0.6 × 0.3) = {z1[1][0]:.4f}")
print(f"z1 = {z1.T}\n")


# ─────────────────────────────────────────────
# STEP 4 : ReLU activation
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 4 : ReLU activation  a1 = ReLU(z1)")
print("=" * 55)

def relu(z):
    return np.maximum(0, z)

a1 = relu(z1)
print(f"ReLU({z1[0][0]:.4f}) = {a1[0][0]:.4f}")
print(f"ReLU({z1[1][0]:.4f}) = {a1[1][0]:.4f}  <-- negative set to 0")
print(f"a1 = {a1.T}\n")


# ─────────────────────────────────────────────
# STEP 5 : Forward pass - output layer
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 5 : Output  y_hat = W2 · a1")
print("=" * 55)

y_hat = W2 @ a1   # (1x2) @ (2x1) = (1x1)

print(f"y_hat = (0.7 × {a1[0][0]}) + (0.4 × {a1[1][0]})")
print(f"y_hat = {y_hat[0][0]:.4f}")
print(f"In real units (0-20°C range): {y_hat[0][0] * 20:.2f}°C\n")


# ─────────────────────────────────────────────
# STEP 6 : Compute loss (MSE)
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 6 : Loss = (y_hat - y)²")
print("=" * 55)

loss = (y_hat - y_true) ** 2
print(f"Loss = ({y_hat[0][0]:.4f} - {y_true[0][0]})²  =  {loss[0][0]:.4f}\n")


# ─────────────────────────────────────────────
# STEP 7 : Update only W2  (transfer learning step)
# W1 stays FROZEN -- this is the core of transfer learning
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 7 : Update W2 only  (W1 stays frozen!)")
print("=" * 55)

eta = 0.1   # learning rate

# Gradient of loss w.r.t. W2
grad_W2 = 2 * (y_hat - y_true) * a1.T   # shape (1x2)

W2_new = W2 - eta * grad_W2

print(f"Gradient ∂Loss/∂W2 = {grad_W2}")
print(f"\nW2 before : {W2}")
print(f"W2 after  : {W2_new}")
print(f"\nW1 unchanged (frozen): \n{W1}\n")


# ─────────────────────────────────────────────
# STEP 8 : Run 100 Cork training steps to show convergence
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 8 : Fine-tune W2 over 100 Cork samples")
print("=" * 55)

# Reset W2 to pre-trained Dublin values
W2 = np.array([[0.7, 0.4]])

# Simulate 100 Cork training examples
np.random.seed(42)
cork_X = np.random.rand(100, 2, 1)   # 100 input pairs
cork_Y = np.random.rand(100, 1, 1)   # 100 target temperatures

losses = []

for i, (xi, yi) in enumerate(zip(cork_X, cork_Y)):
    # Forward pass (W1 stays frozen throughout)
    z1_i  = W1 @ xi
    a1_i  = relu(z1_i)
    yh_i  = W2 @ a1_i

    # Loss
    loss_i = float(((yh_i - yi) ** 2).sum())
    losses.append(loss_i)

    # Update ONLY W2
    grad = 2 * (yh_i - yi) * a1_i.T
    W2 = W2 - eta * grad

    if (i + 1) % 25 == 0:
        print(f"  Sample {i+1:3d} | Loss = {loss_i:.4f} | W2 = {W2.round(4)}")

print(f"\nFinal W2 (adapted to Cork): {W2.round(4)}")
print(f"W1 still frozen:            \n{W1}")

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print("  W1  -->  FROZEN  (general weather pattern knowledge)")
print("  W2  -->  UPDATED (adapted to Cork's local patterns)")
print("  This is Transfer Learning.")
print("=" * 55)