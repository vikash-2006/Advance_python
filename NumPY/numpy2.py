import numpy as np


# =============================================================================
# SECTION 3 — eye()
# =============================================================================
#
# np.eye(n) :
#   Creates an Identity Matrix of size n x n.
#   - Diagonal elements are set to 1
#   - All other elements are set to 0
#   - Commonly used in linear algebra operations
# =============================================================================

a = np.eye(4)
print("Identity Matrix (4x4) :\n", a)
# Output:
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]


# =============================================================================
# SECTION 4 — diag()
# =============================================================================
#
# np.diag(array) :
#   Takes a 1D array and places its values on the diagonal of a 2D matrix.
#   All other positions are filled with 0.
#   - Input  : 1D array of n elements
#   - Output : n x n matrix with given values on the diagonal
# =============================================================================

a = np.array([1, 45, 78, 90])
print("\n1D Array          :", a)           # Output: [ 1 45 78 90]

b = np.diag(a)
print("Diagonal Matrix :\n", b)
# Output:
# [[ 1  0  0  0]
#  [ 0 45  0  0]
#  [ 0  0 78  0]
#  [ 0  0  0 90]]


# =============================================================================
# SECTION 5 — Random Module
# =============================================================================
#
# NumPy provides a random module (np.random) to generate random numbers.
# It is useful for simulations, testing, and machine learning.
# =============================================================================


# -----------------------------------------------------------------------------
# 5a. randint()
# -----------------------------------------------------------------------------
# np.random.randint(low, high, size) :
#   Generates random integers between low (inclusive) and high (exclusive).
#   - low  : Minimum value (included)
#   - high : Maximum value (excluded)
#   - size : Total number of random integers to generate
# -----------------------------------------------------------------------------

a = np.random.randint(1, 10, 3)
print("\nrandom.randint(1, 10, 3) :", a)    # Output: 3 random integers between 1-9


# -----------------------------------------------------------------------------
# 5b. rand()
# -----------------------------------------------------------------------------
# np.random.rand(n) :
#   Generates n random float values between 0 (inclusive) and 1 (exclusive).
#   - Values are uniformly distributed in [0.0, 1.0)
#   - Useful for probability and statistics problems
# -----------------------------------------------------------------------------

a = np.random.rand(5)
print("random.rand(5)           :", a)      # Output: 5 random floats between 0 and 1


# -----------------------------------------------------------------------------
# 5c. seed()
# -----------------------------------------------------------------------------
# np.random.seed(n) :
#   Fixes the random number generator so results are REPRODUCIBLE.
#   - Using the same seed always produces the same random output.
#   - Commonly used in data science to get consistent results.
#   - n can be any integer value (acts as the starting point / key)
# -----------------------------------------------------------------------------

np.random.seed(3)
a = np.random.randint(1, 10, 3)
print("random with seed(3)      :", a)      # Output: Always [9 4 9] for seed=3


# =============================================================================
# SECTION 6 — View vs Copy
# =============================================================================
#
# View :
#   - A view does NOT create a new array. It references the original data.
#   - Any changes made through the view WILL affect the original array.
#   - Created by simple slicing -> b = a[2:5]
#
# Copy :
#   - A copy creates a completely independent duplicate of the array.
#   - Changes made to the copy do NOT affect the original array.
#   - Created using .copy() -> b = a[2:5].copy()
# =============================================================================

# --- VIEW Example ---
a = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print("\nOriginal Array    :", a)

a[3:6] = 0                                  # Modifies original directly
print("After a[3:6] = 0  :", a)             # Output: [10 20 30  0  0  0 70 80]

# --- COPY Example ---
a = np.array([10, 20, 30, 40, 50, 60, 70, 80])
b = a[3:6].copy()                           # Creates an independent copy
b[:] = 0                                    # Only 'b' changes, 'a' stays the same
print("Original (a)      :", a)             # Output: [10 20 30 40 50 60 70 80]
print("Copy (b) after b[:]=0 :", b)         # Output: [0 0 0]


# =============================================================================
# SECTION 7 — Operations on Arrays
# =============================================================================


# -----------------------------------------------------------------------------
# 7a. arange()
# -----------------------------------------------------------------------------
# np.arange(start, stop, step) :
#   Works like Python's range() but returns a NumPy array.
#   - start : Starting value (included)
#   - stop  : Ending value   (excluded)
#   - step  : Increment between values
# -----------------------------------------------------------------------------

a = np.arange(1, 16, 1)
print("\narange(1, 16, 1)  :", a)           # Output: [ 1  2  3 ... 15]

# Boolean Condition on Array
print("a > 10            :", a > 10)        # Returns True/False for each element

# Boolean Indexing — Filter even numbers
b = a % 2 == 0
print("Even numbers in a :", a[b])          # Output: [ 2  4  6  8 10 12 14]


# -----------------------------------------------------------------------------
# 7b. reshape()
# -----------------------------------------------------------------------------
# array.reshape(rows, columns) :
#   Changes the shape of an array without changing its data.
#
# IMPORTANT RULE :
#   rows * columns = total number of elements
#   Example : 12 elements -> valid shapes: (1,12), (2,6), (3,4), (4,3), (6,2), (12,1)
# -----------------------------------------------------------------------------

a = np.random.randint(1, 50, 12)
print("\nFlat Array (12 elements)  :", a)

print("Reshaped to (2, 6) :\n", a.reshape(2, 6))
print("Reshaped to (3, 4) :\n", a.reshape(3, 4))


# -----------------------------------------------------------------------------
# 7c. Array Arithmetic Operations
# -----------------------------------------------------------------------------
# NumPy supports element-wise arithmetic between arrays of the same shape.
#   a + b  -> Element-wise Addition
#   a - b  -> Element-wise Subtraction
#   a * b  -> Element-wise Multiplication  (NOT matrix multiplication)
#   a.dot(b) -> Matrix Multiplication      (dot product)
# -----------------------------------------------------------------------------

a = np.arange(1, 5).reshape(2, 2)
b = np.arange(5, 9).reshape(2, 2)

print("\nArray a :\n", a)
# [[1 2]
#  [3 4]]

print("Array b :\n", b)
# [[5 6]
#  [7 8]]

print("a + b (Addition)              :\n", a + b)
# [[ 6  8]
#  [10 12]]

print("a - b (Subtraction)           :\n", a - b)
# [[-4 -4]
#  [-4 -4]]

print("a * b (Element-wise Multiply) :\n", a * b)
# [[ 5 12]
#  [21 32]]

print("a.dot(b) (Matrix Multiply)    :\n", a.dot(b))
# [[19 22]
#  [43 50]]