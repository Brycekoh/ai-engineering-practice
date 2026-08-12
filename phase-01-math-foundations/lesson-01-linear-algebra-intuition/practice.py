"""Phase 1 - Lesson 01: Linear Algebra Intuition

Lesson notes: phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md

Type the code out yourself - don't paste it. Typing is what makes it stick.
Run this file with:  python practice.py
"""


# ---------------------------------------------------------------------------
# Step 1 - The Vector class
#
# Fill in each method. Delete the `pass` and write the real line.
# Hints are in the comments; the answers are in the lesson notes if stuck.
# ---------------------------------------------------------------------------
import math
import random

import numpy as np

# Flip to True when you want to work the exercises at the bottom.
DO_EXERCISES = False


class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        # Pair up matching positions with zip(), add each pair.
        # return Vector([...])
        return Vector([a+b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        # Same as __add__, but subtract.
        return Vector([a-b for a, b in zip(self.components, other.components)])
    def dot(self, other):
        # "Multiply matching pairs, then add them up."
        # sum(a * b for a, b in zip(...))
        return sum(a * b for a, b in zip(self.components, other.components))
    def magnitude(self):
        # Pythagoras: square each part, sum, then ** 0.5 (square root).
        return sum(x ** 2 for x in self.components) ** 0.5
    def normalize(self):
        # Divide every component by the magnitude -> length becomes 1.
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])
    def cosine_similarity(self, other):
        # dot product divided by (both magnitudes multiplied)
        return self.dot(other) / (self.magnitude() * other.magnitude())
    def __repr__(self):
        return f"Vector({self.components})"

a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")           # expect Vector([5, 7, 9])
print(f"a . b = {a.dot(b)}")        # expect 32
print(f"|a| = {a.magnitude()}")   # expect ~3.7417
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")  # expect ~0.9746

# ---------------------------------------------------------------------------
# Step 2 - The Matrix class (a machine that moves points)
# ---------------------------------------------------------------------------

class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        # Makes the @ operator work:  rotation @ point
        # Each output component = one row of this matrix, dotted with `other`.
        if isinstance(other, Vector):
            # return Vector([...])
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                # dot product of row i of this matrix with column j of other
                dot_product = sum(self.rows[i][k] * other.rows[k][j] for k in range(self.shape[1]))
                row.append(dot_product)
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])
    def __repr__(self):
        return f"Matrix({self.rows})"

rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print (f"original = {point}, rotated = {rotated}")  # expect Vector([-1, 3])
# ---------------------------------------------------------------------------
# Step 3 - Projection (the "shadow" of a onto b)
# ---------------------------------------------------------------------------

def project(a, b):
    # scalar = (a . b) / (b . b)
    # return Vector([scalar * x for x in b.components])
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


# ---------------------------------------------------------------------------
# Exercise helpers - fill these in
# ---------------------------------------------------------------------------

def angle_between(a, b):
    """Ex 1: angle between two vectors, in degrees."""
    # cos_theta = a.cosine_similarity(b)
    # return math.degrees(math.acos(cos_theta))
    pass


def most_similar_pair(count, dim):
    """Ex 3: make `count` random vectors of `dim` numbers, return the two
    most similar (highest cosine similarity) as (index_a, index_b, score)."""
    # vectors = [Vector([random.gauss(0, 1) for _ in range(dim)]) for _ in range(count)]
    # then compare every pair i < j, keep the best score
    pass


def gram_schmidt(vectors):
    """Ex 4: turn independent vectors into an orthonormal basis.

    For each vector: subtract its projection onto every basis vector found
    so far, then normalize what's left.
    """
    orthonormal = []
    for v in vectors:
        w = v
        # for u in orthonormal:
        #     w = w - project(w, u)
        # orthonormal.append(w.normalize())
        pass
    return orthonormal


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])

    print("=== Vectors ===")
    print(f"a + b            = {a + b}")           # expect Vector([5, 7, 9])
    print(f"a . b            = {a.dot(b)}")        # expect 32
    print(f"|a|              = {a.magnitude()}")   # expect ~3.7417
    print(f"cosine(a, b)     = {a.cosine_similarity(b)}")

    print()
    print("=== Matrix: rotate 90 degrees ===")
    rotation_90 = Matrix([[0, -1], [1, 0]])
    point = Vector([3, 1])
    print(f"original         = {point}")
    print(f"rotated          = {rotation_90 @ point}")   # expect Vector([-1, 3])

    print()
    print("=== Projection ===")
    print(f"proj of [3,4] onto [1,0] = {project(Vector([3, 4]), Vector([1, 0]))}")
    # expect Vector([3.0, 0.0])

    # -----------------------------------------------------------------------
    # Step 4 - USE IT: the same thing in NumPy
    #
    # Compare these numbers to yours above. They should match exactly.
    # NumPy is doing the identical arithmetic - just in optimized C.
    # -----------------------------------------------------------------------
    print()
    print("=== NumPy (same math, one line each) ===")

    na = np.array([1, 2, 3], dtype=float)
    nb = np.array([4, 5, 6], dtype=float)

    print(f"a + b            = {na + nb}")
    print(f"a . b            = {np.dot(na, nb)}")
    print(f"|a|              = {np.linalg.norm(na):.4f}")
    print(f"cosine(a, b)     = {np.dot(na, nb) / (np.linalg.norm(na) * np.linalg.norm(nb)):.4f}")

    # Rotation matrix, same as yours. @ is the same operator you implemented.
    nrot = np.array([[0, -1], [1, 0]])
    npoint = np.array([3, 1])
    print(f"rotated          = {nrot @ npoint}")

    # Projection of [3,4] onto [1,0] - the shadow formula, unchanged.
    pa, pb = np.array([3, 4]), np.array([1, 0])
    print(f"projection       = {(np.dot(pa, pb) / np.dot(pb, pb)) * pb}")

    # -----------------------------------------------------------------------
    # Step 5 - A neural network layer is just matrix @ vector
    #
    # W holds the "weights". Training a model = adjusting these numbers.
    # -----------------------------------------------------------------------
    print()
    print("=== One neural network layer ===")

    rng = np.random.default_rng(42)
    W = rng.standard_normal((2, 3)) * 0.1   # 2x3 weight matrix
    x = np.array([1.0, 0.5, -0.3])          # 3D input

    print(f"input  (3 numbers) = {x}")
    print(f"output (2 numbers) = {W @ x}")
    print("That's it. A layer takes a point in 3D and moves it into 2D.")

    # -----------------------------------------------------------------------
    # Step 6 - Rank: how many independent directions?
    # -----------------------------------------------------------------------
    print()
    print("=== Rank ===")
    print(f"rank [[1,2],[2,4]] = {np.linalg.matrix_rank(np.array([[1, 2], [2, 4]]))}")  # 1
    print(f"rank [[1,0],[0,1]] = {np.linalg.matrix_rank(np.array([[1, 0], [0, 1]]))}")  # 2

    # -----------------------------------------------------------------------
    # Step 7 - PyTorch: the same dot product, but it tracks gradients
    #
    # This is the bridge to training. PyTorch remembers every operation so it
    # can work out how to adjust the numbers later. More in Phase 3.
    # -----------------------------------------------------------------------
    print()
    print("=== PyTorch ===")
    try:
        import torch

        tx = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
        ty = torch.tensor([1.0, 0.0, 0.0])

        similarity = torch.dot(tx, ty)
        similarity.backward()          # work backwards through the maths

        print(f"x            = {tx.data}")
        print(f"y            = {ty.data}")
        print(f"dot product  = {similarity.item():.4f}")
        print(f"d(dot)/dx    = {tx.grad}")
        print("The gradient of a dot product w.r.t. x is just y. PyTorch found that itself.")
        print(f"GPU available: {torch.cuda.is_available()}")
    except ImportError:
        print("torch not installed in this environment - skipping.")

    # -----------------------------------------------------------------------
    # EXERCISES - work these yourself. Expected answers are in the comments.
    # -----------------------------------------------------------------------
    print()
    if not DO_EXERCISES:
        print("=== Exercises skipped (set DO_EXERCISES = True to work them) ===")
        raise SystemExit(0)

    print("=== Exercises ===")

    # Ex 1: angle between two vectors, in degrees.
    #       cosine_similarity gives you cos(angle). Undo it with math.acos,
    #       then convert radians -> degrees with math.degrees.
    print(f"1. angle between [1,0] and [0,1] = {angle_between(Vector([1, 0]), Vector([0, 1]))}")
    # expect 90.0  (perpendicular)

    # Ex 2: a scaling matrix that doubles x and triples y, applied to [1, 1].
    #       Hint: what 2x2 matrix sends [x, y] -> [2x, 3y]?
    scale = Matrix([[0, 0], [0, 0]])          # <- fix these numbers
    print(f"2. scaled [1,1] = {scale @ Vector([1, 1])}")
    # expect Vector([2, 3])

    # Ex 3: 5 random 50-dimensional "word" vectors - find the two most similar.
    print(f"3. most similar pair = {most_similar_pair(5, 50)}")

    # Ex 4: verify a Gram-Schmidt result really is orthonormal.
    basis = gram_schmidt([Vector([3, 0]), Vector([2, 2])])
    print(f"4. basis = {basis}")
    print(f"   u1 . u2 = {basis[0].dot(basis[1]):.6f}")   # expect ~0 (perpendicular)
    print(f"   |u1| = {basis[0].magnitude():.6f}")         # expect 1.0
    print(f"   |u2| = {basis[1].magnitude():.6f}")         # expect 1.0

    # Ex 5: build a 3x3 matrix whose rank is 2, and check it.
    rank2 = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]])              # <- change so rank is 2 a different way
    print(f"5. rank = {np.linalg.matrix_rank(rank2)}")     # expect 2

    # Ex 6: project [1,2,3] onto [1,1,1]. What does the answer represent?
    print(f"6. projection = {project(Vector([1, 2, 3]), Vector([1, 1, 1]))}")
    # expect Vector([2.0, 2.0, 2.0]) - look at the number 2 and the input. Notice anything?
