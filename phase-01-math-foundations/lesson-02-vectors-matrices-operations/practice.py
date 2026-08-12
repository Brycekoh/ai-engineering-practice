"""Phase 1 - Lesson 02: Vectors, Matrices & Operations

Lesson notes: phases/01-math-foundations/02-vectors-matrices-operations/docs/en.md

Type the code out yourself - don't paste it. Typing is what makes it stick.
Run this file with:  python practice.py
"""

import random


# ---------------------------------------------------------------------------
# Step 1 - Vector (you already know this from Lesson 1 — quick rebuild)
# ---------------------------------------------------------------------------

class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        # zip + add matching pairs
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        # scale every element by `scalar`
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5

# ---------------------------------------------------------------------------
# Step 2 - Matrix class with ALL operations
#
# This is the big one. Fill in each method. Expected outputs are below.
# ---------------------------------------------------------------------------

class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        # Add matching positions: self.data[i][j] + other.data[i][j]
        return Matrix([self.data[   ]])


    def __sub__(self, other):
        pass

    def scalar_multiply(self, scalar):
        # Multiply every element by scalar
        pass

    def element_wise_multiply(self, other):
        # Multiply matching positions (NOT matrix multiply)
        #   | 1  2 |   | 5  6 |   |  5  12 |
        #   | 3  4 | * | 7  8 | = | 21  32 |
        pass

    def matmul(self, other):
        # Matrix multiplication: each cell = dot product of row from self
        # with column from other.
        #   | 1  2 |   | 5  6 |   | 19  22 |
        #   | 3  4 | @ | 7  8 | = | 43  50 |
        #
        # Cell (i,j) = sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
        pass

    def transpose(self):
        # Flip rows and columns: row i becomes column i.
        # A (2x3) becomes a (3x2).
        pass

    def determinant(self):
        # 2x2: ad - bc
        # For larger matrices: expand along first row using minors (recursive).
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            # return a*d - b*c
            pass
        # General case (3x3 and up): cofactor expansion along row 0
        # det = 0
        # for j in range(self.cols):
        #     minor = Matrix with row 0 and column j removed
        #     det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        # return det
        pass

    def inverse_2x2(self):
        # Only for 2x2. Formula:
        #   1/det * | d  -b |
        #           | -c  a |
        # Raise ValueError if det == 0 (singular).
        pass

    @staticmethod
    def identity(n):
        # The "do-nothing" matrix. 1s on the diagonal, 0s elsewhere.
        # identity(3) = | 1  0  0 |
        #               | 0  1  0 |
        #               | 0  0  1 |
        pass


# ---------------------------------------------------------------------------
# Step 3 - See it work
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])

    print("=== Core operations ===")
    print(f"A + B           = {(A + B).data}")              # [[6, 8], [10, 12]]
    print(f"A * B (elem)    = {A.element_wise_multiply(B).data}")  # [[5, 12], [21, 32]]
    print(f"A @ B (matmul)  = {A.matmul(B).data}")          # [[19, 22], [43, 50]]
    print(f"A^T             = {A.transpose().data}")         # [[1, 3], [2, 4]]
    print(f"det(A)          = {A.determinant()}")            # -2
    print(f"A^-1            = {A.inverse_2x2().data}")       # [[-2.0, 1.0], [1.5, -0.5]]

    print()
    print("=== Identity check ===")
    I = Matrix.identity(2)
    print(f"I               = {I.data}")                     # [[1, 0], [0, 1]]
    result = A.matmul(A.inverse_2x2())
    print(f"A @ A^-1        = {result.data}")                # [[1.0, 0.0], [0.0, 1.0]]

    print()
    print("=== Shape rule ===")
    C = Matrix([[1, 2], [3, 4], [5, 6]])   # (3 x 2)
    D = Matrix([[7, 8, 9, 10], [11, 12, 13, 14]])  # (2 x 4)
    print(f"C shape: {C.shape}, D shape: {D.shape}")
    print(f"C @ D shape:    = {C.matmul(D).shape}")          # (3, 4)
    print(f"C @ D           = {C.matmul(D).data}")

    # -----------------------------------------------------------------------
    # Step 4 - A REAL NEURAL NETWORK LAYER
    #
    # output = relu(weights @ input + bias)
    #
    # That single line is the entire forward pass. You've now built every
    # piece of it from scratch.
    # -----------------------------------------------------------------------
    print()
    print("=== Neural network layer: relu(W @ x + b) ===")

    random.seed(42)
    inputs = Matrix([[0.5], [0.8], [0.2]])              # 3 features, 1 sample
    weights = Matrix([
        [random.uniform(-1, 1) for _ in range(3)]
        for _ in range(2)
    ])                                                   # 2 neurons, 3 inputs
    bias = Matrix([[0.1], [0.1]])                        # 1 bias per neuron

    def relu_matrix(m):
        return Matrix([[max(0, val) for val in row] for row in m.data])

    pre_activation = weights.matmul(inputs)              # W @ x
    with_bias = pre_activation + bias                    # + b
    output = relu_matrix(with_bias)                      # relu(...)

    print(f"Input shape:    {inputs.shape}")              # (3, 1)
    print(f"Weight shape:   {weights.shape}")             # (2, 3)
    print(f"Output shape:   {output.shape}")              # (2, 1)
    print(f"Output:         {output.data}")
    print()
    print("That's it. output = relu(W @ x + b). A dense layer.")

    # -----------------------------------------------------------------------
    # Step 5 - NumPy comparison (same math, one-liners)
    # -----------------------------------------------------------------------
    print()
    print("=== NumPy (same thing, fewer lines) ===")
    import numpy as np

    A_np = np.array([[1, 2], [3, 4]])
    B_np = np.array([[5, 6], [7, 8]])

    print(f"A + B           =\n{A_np + B_np}")
    print(f"A * B (elem)    =\n{A_np * B_np}")
    print(f"A @ B (matmul)  =\n{A_np @ B_np}")
    print(f"A^T             =\n{A_np.T}")
    print(f"det(A)          = {np.linalg.det(A_np):.1f}")
    print(f"A^-1            =\n{np.linalg.inv(A_np)}")

    # Broadcasting demo
    print()
    print("=== Broadcasting ===")
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    bias_vec = np.array([10, 20, 30])
    print(f"matrix + bias   =\n{matrix + bias_vec}")     # bias stretches across rows

    # -----------------------------------------------------------------------
    # Exercises
    # -----------------------------------------------------------------------
    print()
    print("=== Exercises ===")

    # Ex 1: Verify the inverse. A @ A^-1 should give identity.
    #        Try 3 different 2x2 matrices. What happens when det = 0?

    # Ex 2: Build a TWO-layer network (no NumPy, just your Matrix class):
    #        input (3) -> hidden (4) -> output (2)
    #        Random weights, run a forward pass, verify all shapes.
    #        Hint: layer1 weights are (4 x 3), layer2 weights are (2 x 4)

    print("Exercises: uncomment and implement when ready.")
