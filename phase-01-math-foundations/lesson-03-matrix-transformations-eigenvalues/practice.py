"""Phase 1 - Lesson 03: Matrix Transformations

Lesson notes: phases/01-math-foundations/03-matrix-transformations/docs/en.md

Type the code out yourself - don't paste it. Typing is what makes it stick.
Run this file with:  python practice.py
"""

import math
import random


# ---------------------------------------------------------------------------
# Step 1 - Transformation matrices from scratch
#
# Each function returns a 2x2 list-of-lists matrix.
# ---------------------------------------------------------------------------

def rotation_2d(theta):
    # Returns the 2x2 rotation matrix for angle theta (radians).
    # [[cos θ, -sin θ],
    #  [sin θ,  cos θ]]
    c, s = math.cos(theta), math.sin(theta)
    return [[c , -s], [s, c]]

def scaling_2d(sx, sy):
    # Returns a 2x2 scaling matrix.
    # [[sx,  0],
    #  [ 0, sy]]
    return [[sx,0],[0,sy]]

def shearing_2d(kx, ky):
    # Returns a 2x2 shearing matrix.
    # [[1, kx],
    #  [ky, 1]]
    return [[1,kx], [ky, 1]]    

def reflection_x():
    # Reflect across x-axis: y flips sign.
    # [[1,  0],
    #  [0, -1]]
    return [[1,0],[0,-1]]

def reflection_y():
    # Reflect across y-axis: x flips sign.
    # [[-1, 0],
    #  [ 0, 1]]
    return [[-1,0], [0,1]]


# ---------------------------------------------------------------------------
# Helper: matrix-vector multiply and matrix-matrix multiply
# (You built these in Lesson 2 — rebuild them as standalone functions)
# ---------------------------------------------------------------------------

def mat_vec_mul(matrix, vector):
    # Each output element = dot product of one row with the vector.
    # return [sum(matrix[i][j] * vector[j] for j in ...) for i in ...]
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    # Matrix multiply: cell (i,j) = dot product of row i of a with col j of b.
    # return [[sum(a[i][k] * b[k][j] ...) for j in ...] for i in ...]
    rows_a, cols_b = len(a) , len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]


def det_2x2(matrix):
    # Determinant of 2x2: ad - bc
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


# ---------------------------------------------------------------------------
# Step 2 - Test the transformations
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    point = [1.0, 0.0]
    angle = math.pi / 4  # 45 degrees

    print("=== Transformations ===")
    rotated = mat_vec_mul(rotation_2d(angle), point)
    print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")
    # expect (0.7071, 0.7071)

    scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
    print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")
    # expect (2.0, 3.0)

    sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
    print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")
    # expect (2.0, 1.0)

    reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
    print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")
    # expect (-2.0, 1.0)

    # -----------------------------------------------------------------------
    # Step 3 - Composition: order matters!
    #
    # rotate-then-scale vs scale-then-rotate give DIFFERENT results.
    # "B @ A" means: apply A first, then B.
    # -----------------------------------------------------------------------
    print()
    print("=== Composition ===")

    R = rotation_2d(math.pi / 2)   # 90 degrees
    S = scaling_2d(2, 0.5)

    rotate_then_scale = mat_mul(S, R)   # S applied after R
    scale_then_rotate = mat_mul(R, S)   # R applied after S

    p = [1.0, 0.0]
    r1 = mat_vec_mul(rotate_then_scale, p)
    r2 = mat_vec_mul(scale_then_rotate, p)

    print(f"Rotate 90 then scale: ({r1[0]:.2f}, {r1[1]:.2f})")
    # expect (0.00, 0.50)
    print(f"Scale then rotate 90: ({r2[0]:.2f}, {r2[1]:.2f})")
    # expect (0.00, 2.00)
    print(f"Same? {r1 == r2}")
    # expect False

    # -----------------------------------------------------------------------
    # Step 4 - Determinant as area scaling factor
    # -----------------------------------------------------------------------
    print()
    print("=== Determinants ===")

    print(f"det(rotation 45)  = {det_2x2(rotation_2d(math.pi/4)):.4f}")  # 1.0
    print(f"det(scale 2,3)    = {det_2x2(scaling_2d(2, 3)):.1f}")         # 6.0
    print(f"det(shear kx=1)   = {det_2x2(shearing_2d(1, 0)):.1f}")       # 1.0
    print(f"det(reflect y)    = {det_2x2(reflection_y()):.1f}")           # -1.0

    # -----------------------------------------------------------------------
    # Step 5 - Eigenvalues & eigenvectors from scratch (2x2)
    #
    # Eigenvalue equation: A @ v = lambda * v
    # Characteristic equation: lambda^2 - trace*lambda + det = 0
    # -----------------------------------------------------------------------
    print()
    print("=== Eigenvalues (from scratch) ===")

    def eigenvalues_2x2(matrix):
        a, b = matrix[0]
        c, d = matrix[1]
        trace = a + d
        det = a * d - b * c
        discriminant = trace ** 2 - 4 * det
        if discriminant < 0:
            real = trace / 2
            imag = (-discriminant) ** 0.5 / 2
            return (complex(real, imag), complex(real, -imag))
        sqrt_disc = discriminant ** 0.5
        return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

    def eigenvector_2x2(matrix, eigenvalue):
        a, b = matrix[0]
        c, d = matrix[1]
        if abs(b) > 1e-10:
            v = [b, eigenvalue - a]
        elif abs(c) > 1e-10:
            v = [eigenvalue - d, c]
        else:
            if abs(a - eigenvalue) < 1e-10:
                v = [1, 0]
            else:
                v = [0, 1]
        mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
        return [v[0] / mag, v[1] / mag]

    A = [[2, 1], [1, 2]]
    vals = eigenvalues_2x2(A)
    print(f"Matrix: {A}")
    print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")
    # expect 3.0 and 1.0

    for val in vals:
        vec = eigenvector_2x2(A, val)
        result = mat_vec_mul(A, vec)
        scaled = [val * vec[0], val * vec[1]]
        print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
        print(f"    A@v = {[round(x,4) for x in result]}")
        print(f"    l*v = {[round(x,4) for x in scaled]}")
        # A@v and l*v should match — that's the whole point

    # -----------------------------------------------------------------------
    # Step 6 - NumPy comparison
    # -----------------------------------------------------------------------
    print()
    print("=== NumPy (same thing, fewer lines) ===")
    import numpy as np

    theta = np.pi / 4
    R_np = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
    print(f"Rotate (1,0) by 45 deg: {R_np @ np.array([1.0, 0.0])}")

    A_np = np.array([[2, 1], [1, 2]], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(A_np)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors (columns):\n{eigenvectors}")

    # Eigendecomposition: A = V @ D @ V^-1
    V = eigenvectors
    D = np.diag(eigenvalues)
    reconstructed = V @ D @ np.linalg.inv(V)
    print(f"A = V @ D @ V^-1:\n{reconstructed}")
    # should match A_np exactly

    # -----------------------------------------------------------------------
    # Exercises
    # -----------------------------------------------------------------------
    print()
    print("=== Exercises ===")

    # Ex 1: Apply rotation, scaling, and shearing to a unit square
    #        (corners [0,0], [1,0], [1,1], [0,1]).
    #        Print transformed corners. Verify rotation preserves distances.

    # Ex 2: Find eigenvalues of [[4, 2], [1, 3]] by hand using the
    #        characteristic equation, then verify with your function and NumPy.

    # Ex 3: Compose 3 transformations: rotate 30 deg, scale (1.5, 0.8),
    #        shear kx=0.3. Apply to 8 points on a circle. Verify that
    #        det(composed) = det(R) * det(S) * det(Sh).

    print("Exercises: implement when ready.")
