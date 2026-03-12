import numpy as np      # 'np' is the standard alias used for numpy


# =============================================================================
# SECTION 8 — Mathematical Operations on Arrays
# =============================================================================
#
# NumPy supports a wide range of built-in mathematical functions that apply
# ELEMENT-WISE on every element of the array automatically.
# No need for loops — NumPy handles it internally at high speed.
# =============================================================================

a = np.arange(1, 5)
print("Original Array    :", a)             # Output: [1 2 3 4]


# -----------------------------------------------------------------------------
# 8a. Power Operator (**)
# -----------------------------------------------------------------------------
# a ** n :
#   Raises each element of the array to the power of n.
#   - a**2  -> squares every element
#   - a**3  -> cubes every element
# -----------------------------------------------------------------------------

print("a ** 2 (squares)  :", a ** 2)        # Output: [ 1  4  9 16]
print("a ** 3 (cubes)    :", a ** 3)        # Output: [ 1  8 27 64]


# -----------------------------------------------------------------------------
# 8b. np.sqrt()
# -----------------------------------------------------------------------------
# np.sqrt(array) :
#   Returns the square root of each element in the array.
#   - Output is always in float (decimal) values
#   - Equivalent to a ** 0.5
# -----------------------------------------------------------------------------

print("np.sqrt(a)        :", np.sqrt(a))
# Output: [1.         1.41421356 1.73205081 2.        ]


# -----------------------------------------------------------------------------
# 8c. np.sin() and np.cos()
# -----------------------------------------------------------------------------
# np.sin(array) :
#   Computes the Sine of each element (values in radians, not degrees).
#
# np.cos(array) :
#   Computes the Cosine of each element (values in radians, not degrees).
#
# Note : Results are float values between -1 and 1.
# -----------------------------------------------------------------------------

print("np.sin(a)         :", np.sin(a))
# Output: [ 0.84147098  0.90929743  0.14112001 -0.7568025 ]

print("np.cos(a)         :", np.cos(a))
# Output: [ 0.54030231 -0.41614684 -0.9899925  -0.65364362]


# =============================================================================
# SECTION 9 — linspace()
# =============================================================================
#
# np.linspace(start, stop, num) :
#   Returns 'num' evenly spaced values between start and stop (both inclusive).
#
# KEY DIFFERENCE from arange() :
#   - arange() -> you control the STEP SIZE
#   - linspace() -> you control the TOTAL NUMBER of values
#
# Syntax : np.linspace(start, stop, num_of_values)
#   - start : Starting value (included)
#   - stop  : Ending value   (included)
#   - num   : How many equally spaced points to generate
# =============================================================================

a = np.linspace(1, 5, 4)
print("\nlinspace(1, 5, 4) :", a)
# Output: [1.         2.33333333 3.66666667 5.        ]
# Explanation: 4 values equally spaced between 1 and 5


# =============================================================================
# SECTION 10 — unique()
# =============================================================================
#
# np.unique(array, return_index, return_counts) :
#   Returns unique elements from the array along with optional extra info.
#
# Parameters :
#   - arr            : The input array
#   - return_index   : If True -> returns the INDEX of first occurrence of each unique value
#   - return_counts  : If True -> returns the FREQUENCY (count) of each unique value
#
# Output (3 arrays when both True) :
#   1. Unique values (sorted)
#   2. Index of first occurrence of each unique value
#   3. Count/frequency of each unique value
# =============================================================================

a = np.array([1, 2, 3, 2, 2, 2, 3, 3, 2, 4, 5])
print("\nOriginal Array    :", a)

unique_vals, indices, counts = np.unique(a, return_index=True, return_counts=True)

print("Unique Values     :", unique_vals)    # Output: [1 2 3 4 5]
print("First Index       :", indices)        # Output: [ 0  1  2  9 10]
print("Frequency Count   :", counts)         # Output: [1 5 3 1 1]

# Reading the result:
#   Value 1 -> appears 1 time, first seen at index 0
#   Value 2 -> appears 5 times, first seen at index 1
#   Value 3 -> appears 3 times, first seen at index 2
#   Value 4 -> appears 1 time, first seen at index 9
#   Value 5 -> appears 1 time, first seen at index 10


# =============================================================================
# SECTION 11 — hstack() and vstack()
# =============================================================================
#
# These functions are used to JOIN / COMBINE two or more arrays together.
#
# hstack() — Horizontal Stack :
#   Joins arrays SIDE BY SIDE (column-wise / along axis=1).
#   - Number of ROWS must be equal in all arrays
#   - Result has MORE COLUMNS
#
# vstack() — Vertical Stack :
#   Joins arrays ON TOP OF EACH OTHER (row-wise / along axis=0).
#   - Number of COLUMNS must be equal in all arrays
#   - Result has MORE ROWS
# =============================================================================

a = np.arange(1, 5).reshape(2, 2)
b = np.arange(5, 9).reshape(2, 2)
c = np.arange(9, 13).reshape(2, 2)

print("\nArray a :\n", a)
# [[1 2]
#  [3 4]]

print("Array b :\n", b)
# [[5 6]
#  [7 8]]

print("Array c :\n", c)
# [[ 9 10]
#  [11 12]]

# --- hstack : joins left to right ---
print("hstack(a, b, c) :\n", np.hstack((a, b, c)))
# Output:
# [[ 1  2  5  6  9 10]
#  [ 3  4  7  8 11 12]]

# --- vstack : joins top to bottom ---
print("vstack(a, b, c) :\n", np.vstack((a, b, c)))
# Output:
# [[ 1  2]
#  [ 3  4]
#  [ 5  6]
#  [ 7  8]
#  [ 9 10]
#  [11 12]]


# -----------------------------------------------------------------------------
# hstack() on 1D Arrays
# -----------------------------------------------------------------------------
# hstack on 1D arrays simply CONCATENATES them into one longer 1D array.
# -----------------------------------------------------------------------------

a = np.arange(6, 10)                        # [6 7 8 9]
b = np.arange(11, 15)                       # [11 12 13 14]

print("\nhstack on 1D arrays :", np.hstack((a, b)))
# Output: [ 6  7  8  9 11 12 13 14]


# -----------------------------------------------------------------------------
# Bonus — reshape with random arrays before stacking
# -----------------------------------------------------------------------------
# Rule : Total elements = rows * columns
#   36 elements -> valid shapes: (2,18), (4,9), (6,6), (9,4), (18,2), (3,12) etc.
# -----------------------------------------------------------------------------

np.random.seed(3)
a = np.random.randint(30, 80, 36).reshape(2, 18)    # shape -> (2 rows, 18 cols)
b = np.random.randint(30, 80, 36).reshape(4, 9)     # shape -> (4 rows,  9 cols)
c = np.random.randint(30, 80, 36).reshape(6, 6)     # shape -> (6 rows,  6 cols)

print("\na (2x18) :\n", a)
print("b (4x9)  :\n", b)
print("c (6x6)  :\n", c)