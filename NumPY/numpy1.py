# =============================================================================
#                        INTRODUCTION TO NUMPY
# =============================================================================
#
# What is NumPy?
# --------------
# NumPy stands for Numerical Python. It is an open-source Python library
# used for scientific and mathematical computations. It provides support for
# large, multi-dimensional arrays and matrices, along with a collection of
# high-level mathematical functions to operate on them.
#
# Why NumPy over Python Lists?
# -----------------------------
# - Python List  : Heterogeneous  -> can store mixed data types (int, str, float)
# - NumPy Array  : Homogeneous    -> stores only one data type (faster & efficient)
# - NumPy arrays consume less memory and execute operations much faster than lists.
# =============================================================================

import numpy as np      # 'np' is the standard alias used for numpy


# =============================================================================
# SECTION 1 — List vs NumPy Array (Multiplication Difference)
# =============================================================================
#
# In a regular Python list, the * operator REPEATS the list elements.
# In a NumPy array, the * operator performs ELEMENT-WISE multiplication.
# =============================================================================

a = [1, 2, 3]
print("List * 3        :", a * 3)          # Output: [1, 2, 3, 1, 2, 3, 1, 2, 3]

a = np.array([1, 2, 3])
print("NumPy Array * 3 :", a * 3)          # Output: [3, 6, 9]


# =============================================================================
# SECTION 2 — Converting a Python List to a NumPy Array
# =============================================================================
#
# np.array(object) :
#   Converts any list or sequence into a NumPy ndarray (N-dimensional array).
#   'ndarray' stands for N-Dimensional Array.
# =============================================================================

a = [1, 2, 3]                              # Python List
b = np.array(a)                            # NumPy Array
print("\nOriginal List  :", a)
print("NumPy Array    :", b)

print("\nType of a (list)  :", type(a))    # <class 'list'>
print("Type of b (array) :", type(b))      # <class 'numpy.ndarray'>


# =============================================================================
# SECTION 3 — 2D NumPy Array (Matrix)
# =============================================================================
#
# A 2D array is essentially a Matrix — a table with Rows and Columns.
# We pass a list of lists to np.array() to create a 2D array.
#
# Example Matrix (3x3):
#
#         c_1  c_2  c_3
#   r_1 [  1    2    3  ]
#   r_2 [  4    5    6  ]
#   r_3 [  7    8    9  ]
#
# Rows    : r_1=[1,2,3]  r_2=[4,5,6]  r_3=[7,8,9]
# Columns : c_1=[1,4,7]  c_2=[2,5,8]  c_3=[3,6,9]
# =============================================================================

a = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

b = np.array(a)
print("\n2D NumPy Array (Matrix):\n", b)

# .shape  -> Returns a tuple (rows, columns) — tells the size of each dimension
# .size   -> Returns total number of elements (rows * columns)
# .ndim   -> Returns the number of dimensions (1D=1, 2D=2, 3D=3, ...)

print("\nTotal Shape      :", b.shape)     # (3, 3)  -> 3 rows, 3 columns
print("Total Elements   :", b.size)       # 9       -> 3 * 3
print("Total Dimensions :", b.ndim)       # 2       -> it is a 2D array


# =============================================================================
# SECTION 4 — Pre-Built Functions of NumPy
# =============================================================================


# -----------------------------------------------------------------------------
# 4.1  np.zeros(shape)
# -----------------------------------------------------------------------------
# Creates an array filled entirely with 0.0 (float by default).
# Useful for initializing arrays before filling them with actual data.
#
# Syntax:
#   np.zeros(n)          -> 1D array of n zeros
#   np.zeros((r, c))     -> 2D array with r rows and c columns, all zeros
# -----------------------------------------------------------------------------

print("\n--- np.zeros() ---")

a = np.zeros(3)                            # 1D array
print("1D zeros :", a)                     # [0. 0. 0.]

a = np.zeros((3, 4))                       # 2D array — 3 rows, 4 columns
print("2D zeros :\n", a)


# -----------------------------------------------------------------------------
# 4.2  np.ones(shape)
# -----------------------------------------------------------------------------
# Creates an array filled entirely with 1.0 (float by default).
# Useful for creating identity-like arrays or initializing weights.
#
# Syntax:
#   np.ones(n)           -> 1D array of n ones
#   np.ones((r, c))      -> 2D array with r rows and c columns, all ones
# -----------------------------------------------------------------------------

print("\n--- np.ones() ---")

a = np.ones(4)                             # 1D array
print("1D ones :", a)                      # [1. 1. 1. 1.]

b = np.ones((2, 3))                        # 2D array — 2 rows, 3 columns
print("2D ones :\n", b)


# -----------------------------------------------------------------------------
# Practice Problem:
# Create a 5x6 array where all elements are 1, then display:
#   -> the array
#   -> its shape
#   -> its number of dimensions
#   -> its total number of elements
# -----------------------------------------------------------------------------

print("\n--- Practice: 5x6 Array of Ones ---")

a = np.ones((5, 6))
print(a)
print("Shape          :", a.shape)         # (5, 6)
print("Dimensions     :", a.ndim)          # 2
print("Total Elements :", a.size)          # 30  -> 5 * 6