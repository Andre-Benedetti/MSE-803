import numpy as np
from scipy.signal import correlate2d

# 1. Define Matrix A (Input) and Matrix B (Filter/Kernel)
A = np.array([
    [1, 2, 3],
    [5, 6, 7],
    [10, 0, 11]
])

B = np.array([
    [5, 3],
    [9, 1]
])

# 2. Perform the convolution operation (sliding window cross-correlation)
output = correlate2d(A, B, mode='valid')

# 3. Display the matrices and the final calculated outcome
print("Input Matrix A:")
print(A)
print("\nFilter B (Kernel):")
print(B)
print("\nConvolution Operation Outcome:")
print(output)