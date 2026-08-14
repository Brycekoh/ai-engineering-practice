"""Phase 1 - Lesson 04: Calculus for ML

Lesson notes: phases/01-math-foundations/04-calculus-for-ml/docs/en.md

Type the code out yourself - don't paste it. Typing is what makes it stick.
Run this file with:  python practice.py
"""

import math
import random


# ---------------------------------------------------------------------------
# Step 1 - Numerical derivative from scratch
#
# The derivative measures: if I nudge x by a tiny amount, how much does f(x)
# change? Use the central difference formula:
#   f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h)
# ---------------------------------------------------------------------------

def numerical_derivative(f, x, h=1e-7):
    # Central difference: (f(x+h) - f(x-h)) / (2h)
    return (f(x + h) - f(x - h) )/ (2 * h)

# ---------------------------------------------------------------------------
# Step 2 - Partial derivatives and gradient
#
# For a function with multiple inputs, the gradient is a vector of partial
# derivatives — one per input. Nudge each input independently.
# ---------------------------------------------------------------------------

def numerical_gradient(f, point, h=1e-7):
    # For each dimension i:
    #   make a copy of point, nudge index i by +h and -h
    #   partial = (f(point_plus) - f(point_minus)) / (2h)
    #   collect all partials into a list
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient


# ---------------------------------------------------------------------------
# Step 3 - Test derivatives
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Numerical vs Analytical Derivatives ===")

    def f(x):
        return x ** 2

    for x in [-2, -1, 0, 1, 2]:
        numerical = numerical_derivative(f, x)
        analytical = 2 * x
        print(f"x={x:2d}  numerical={numerical:.6f}  analytical={analytical:.1f}")
    # numerical should match analytical closely

    # -----------------------------------------------------------------------
    # Step 4 - Gradient of a multivariable function
    # -----------------------------------------------------------------------
    print()
    print("=== Gradient ===")

    def f_multi(point):
        x, y = point
        return x**2 + 3*x*y + y**2

    grad = numerical_gradient(f_multi, [1.0, 2.0])
    print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
    print(f"Analytical: [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
    # expect [8.0, 7.0]

    # -----------------------------------------------------------------------
    # Step 5 - Gradient descent in 1D: find the minimum of f(x) = x²
    #
    # Start at x=5. Each step: x = x - lr * gradient
    # Should converge to x=0 (the bottom of the bowl)
    # -----------------------------------------------------------------------
    print()
    print("=== Gradient Descent 1D ===")

    x = 5.0
    lr = 0.1
    for step in range(20):
        grad = 2 * x
        x = x - lr * grad
        if step % 4 == 0 or step == 19:
            print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
    # x should approach 0

    # -----------------------------------------------------------------------
    # Step 6 - Gradient descent in 2D
    #
    # Minimize f(x,y) = x² + y². Start at (4, 3).
    # Should converge to (0, 0).
    # -----------------------------------------------------------------------
    print()
    print("=== Gradient Descent 2D ===")

    def f_2d(point):
        x, y = point
        return x**2 + y**2

    point = [4.0, 3.0]
    lr = 0.1
    for step in range(30):
        grad = numerical_gradient(f_2d, point)
        point = [p - lr * g for p, g in zip(point, grad)]
        loss = f_2d(point)
        if step % 5 == 0 or step == 29:
            print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
    # should converge to (0, 0)

    # -----------------------------------------------------------------------
    # Step 7 - TRAIN A REAL MODEL: linear regression
    #
    # Data: y = 2x + 1 (but the model doesn't know that)
    # Model: prediction = w * x + b
    # Loss: mean squared error = mean((pred - actual)²)
    # Gradients: dL/dw = mean(2 * error * x), dL/db = mean(2 * error)
    #
    # Fill in the gradient computation and weight update.
    # -----------------------------------------------------------------------
    print()
    print("=== Training Linear Regression ===")

    random.seed(42)
    w = random.gauss(0, 1)
    b = random.gauss(0, 1)
    lr = 0.01

    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [3.0, 5.0, 7.0, 9.0, 11.0]

    for epoch in range(200):
        total_loss = 0
        dw = 0
        db = 0
        for x, y in zip(xs, ys):
            pred = w * x + b
            error = pred - y
            total_loss += error ** 2
            dw += 2 * error * x
            db += 2 * error
            # dw += ___  (gradient of loss w.r.t. w = 2 * error * x)
            # db += ___  (gradient of loss w.r.t. b = 2 * error)
            pass
        dw /= len(xs)
        db /= len(xs)
        total_loss /= len(xs)
        w -= lr * dw
        b -= lr * db
        # w -= ___  (update w using learning rate and gradient)
        # b -= ___  (update b using learning rate and gradient)
        pass
        if epoch % 40 == 0 or epoch == 199:
            print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

    print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
    print(f"Actual:  y = 2x + 1")
    # w should converge to ~2.0, b to ~1.0

    # -----------------------------------------------------------------------
    # Step 8 - NumPy comparison (same thing, fewer lines)
    # -----------------------------------------------------------------------
    print()
    print("=== NumPy (same thing, fewer lines) ===")
    import numpy as np

    x_np = np.array([1, 2, 3, 4, 5], dtype=float)
    y_np = np.array([3, 5, 7, 9, 11], dtype=float)

    w, b = np.random.randn(), np.random.randn()
    lr = 0.01

    for epoch in range(200):
        pred = w * x_np + b
        error = pred - y_np
        loss = np.mean(error ** 2)
        dw = np.mean(2 * error * x_np)
        db = np.mean(2 * error)
        w -= lr * dw
        b -= lr * db

    print(f"Learned: y = {w:.2f}x + {b:.2f}")

    # -----------------------------------------------------------------------
    # Exercises
    # -----------------------------------------------------------------------
    print()
    print("=== Exercises ===")

    # Ex 1: Use gradient descent to find the minimum of
    #        f(x, y) = (x - 3)² + (y + 1)²
    #        Start from (0, 0). Should converge to (3, -1).

    def f_ex1(point):
        x, y = point
        return (x - 3) ** 2 + (y + 1) ** 2

    point = [0.0, 0.0]
    lr = 0.1
    for step in range(30):
        grad = numerical_gradient(f_ex1, point)
        point = [p - lr * g for p, g in zip(point, grad)]
        if step % 5 == 0 or step == 29:
            print(f"step {step:2d}  point=({point[0]:.4f}, {point[1]:.4f})  f={f_ex1(point):.6f}")

    # Ex 2: Try different learning rates (0.001, 0.1, 0.5, 1.0) on the 1D
    #        gradient descent. Print how many steps each takes to reach
    #        f(x) < 0.0001. What happens with lr=1.0?
    for test_lr in [0.001, 0.1, 0.5, 1.0]:
        x = 5.0
        for step in range(1000):
            grad = 2 * x
            x = x - test_lr * grad
            if x ** 2 < 0.0001:
                print(f"lr={test_lr:<5}  converged in {step} steps")
                break
        else:
            print(f"lr={test_lr:<5}  did NOT converge after 1000 steps, x={x:.4f}")