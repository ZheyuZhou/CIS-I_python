import numpy as np
import scipy

A = np.array([
    [-1, -2],
    [-2, -1]
])

print(scipy.linalg.eig(A, left=True))