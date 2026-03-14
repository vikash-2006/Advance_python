# =============================================================================
#  PANDAS FUNDAMENTALS — PLACEMENT, TIPS & COVID DATASETS
#  File   : pandas4.py
#  Topics : DataFrame Properties, Label Encoding, EDA
# =============================================================================

import pandas as pd

# =============================================================================
#  SECTION 1 : PLACEMENT DATASET
#  Columns    : cgpa, resume_score, placed
# =============================================================================

# -----------------------------------------------------------------------------
# 1. pd.read_csv()
#    Reads a CSV file from disk and loads it into a pandas DataFrame.
#    A DataFrame is a 2D table — like an Excel sheet — with rows and columns.
#    Always the first step when working with CSV data.
# -----------------------------------------------------------------------------
df = pd.read_csv("/Users/apple/data science/Advance_Python/Pandas/Day_4_Pandas/placement - placement.csv")

# -----------------------------------------------------------------------------
# 2. df  (just typing the variable)
#    Displays the ENTIRE DataFrame in a notebook / interactive shell.
#    In a .py script, wrap it in print() to see output in the terminal.
# -----------------------------------------------------------------------------
print("=" * 55)
print("FULL DATAFRAME:")
print("=" * 55)
print(df)

# -----------------------------------------------------------------------------
# 3. df.head(n)
#    Returns the FIRST n rows of the DataFrame.
#    Default is 5 rows if no argument is passed.
#    Used to get a quick preview of what the data looks like.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("HEAD — First 5 Rows:")
print("=" * 55)
print(df.head())

# -----------------------------------------------------------------------------
# 4. df.tail(n)
#    Returns the LAST n rows of the DataFrame.
#    Default is 5 rows if no argument is passed.
#    Useful to verify the data ends correctly (no extra blank rows, etc.).
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("TAIL — Last 5 Rows:")
print("=" * 55)
print(df.tail())

# -----------------------------------------------------------------------------
# 5. df.info()
#    Prints a concise summary of the DataFrame including:
#      - Number of rows and columns
#      - Column names and their data types (int, float, object, etc.)
#      - Count of non-null (non-missing) values per column
#      - Memory usage
#    Great for quickly spotting missing data or wrong data types.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("INFO — DataFrame Summary:")
print("=" * 55)
df.info()

# -----------------------------------------------------------------------------
# 6. df.describe()
#    Generates DESCRIPTIVE STATISTICS for all numeric columns:
#      count  — number of non-null values
#      mean   — average value
#      std    — standard deviation (spread of data)
#      min    — minimum value
#      25%    — first quartile  (25% of data is below this)
#      50%    — median          (middle value)
#      75%    — third quartile  (75% of data is below this)
#      max    — maximum value
#    Helps understand the distribution and scale of numeric features.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("DESCRIBE — Statistical Summary:")
print("=" * 55)
print(df.describe())

# -----------------------------------------------------------------------------
# 7. df.iloc[row_range, col_range]
#    Selects data using INTEGER-based index positions (like list slicing).
#      df.iloc[:4, :3]  →  rows 0,1,2,3  and  columns 0,1,2
#    Think of it as: "give me rows at positions __ and columns at positions __"
#    The endpoint is EXCLUSIVE (just like Python slicing).
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("ILOC — Top-Left Corner (rows 0-3, cols 0-2):")
print("=" * 55)
top_left_corner_df = df.iloc[:4, :3]
print(top_left_corner_df)

# -----------------------------------------------------------------------------
# 8. df.loc[row_range, column_names]
#    Selects data using LABEL-based indexing.
#      df.loc[1:3, ['cgpa', 'placed']]  →  rows with index 1,2,3
#                                           and only the named columns
#    Unlike iloc, the endpoint is INCLUSIVE (row 3 is included).
#    Use when you know the column NAMES you want to select.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("LOC — Rows 1 to 3, Columns: cgpa & placed:")
print("=" * 55)
print(df.loc[1:3, ['cgpa', 'placed']])

# -----------------------------------------------------------------------------
# 9. df.axes
#    Returns a list of two objects:
#      [0] → RangeIndex  : the row labels (e.g. 0, 1, 2, ... 99)
#      [1] → Index       : the column labels (e.g. 'cgpa', 'resume_score')
#    Useful to quickly see the index and column structure of a DataFrame.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("AXES — Row and Column Labels:")
print("=" * 55)
s = df.axes
print(s)

# -----------------------------------------------------------------------------
# 10. df.dtypes
#     Returns the DATA TYPE of every column in the DataFrame.
#     Common types:
#       int64   → whole numbers
#       float64 → decimal numbers
#       object  → strings / mixed types
#       bool    → True / False
#     Knowing dtypes helps you spot columns that need type conversion.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("DTYPES — Data Types of All Columns:")
print("=" * 55)
p = df.dtypes
print(p)

# -----------------------------------------------------------------------------
# 11. df['column'].dtypes  (dtype of a single column / Series)
#     Same as df.dtypes but for one specific column.
#     Useful when you only want to check a single feature's type.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("DTYPE — Single Column 'cgpa':")
print("=" * 55)
print(df['cgpa'].dtypes)

# -----------------------------------------------------------------------------
# 12. df.empty
#     Returns True  → if the DataFrame has NO rows OR NO columns at all.
#     Returns False → if it has at least one row and one column.
#     ⚠️  COMMON MISTAKE: This does NOT check for NaN/missing values.
#         To check for missing values, use df.isnull().sum() instead.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("EMPTY — Is the DataFrame completely empty?")
print("=" * 55)
f = df.empty
print(f)  # Expect: False (since data was loaded successfully)

# -----------------------------------------------------------------------------
# 13. df.isnull().sum()
#     df.isnull() → creates a boolean DataFrame: True where value is NaN
#     .sum()      → counts the True values per column
#     Result      → number of MISSING VALUES in each column
#     Use this after loading data to decide if cleaning is needed.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("ISNULL — Missing Values Per Column:")
print("=" * 55)
print(df.isnull().sum())

# -----------------------------------------------------------------------------
# 14. df.dropna()
#     Removes all ROWS that contain at least one NaN (missing) value.
#     Returns a NEW DataFrame — the original is not modified unless
#     you reassign: df = df.dropna()
#
#     ⚠️  BUG FIX:
#         df = df.dropna     ← WRONG  (just stores the method, does nothing)
#         df = df.dropna()   ← CORRECT (calls the method with parentheses)
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("DROPNA — Remove Rows with Missing Values:")
print("=" * 55)
df = df.dropna()   # ✅ Fixed: added ()
print(df)

# -----------------------------------------------------------------------------
# 15. df.ndim
#     Returns the NUMBER OF DIMENSIONS (axes) of the DataFrame.
#     A DataFrame is always 2D → ndim = 2  (rows + columns)
#     A Series  is always 1D → ndim = 1
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("NDIM — Number of Dimensions:")
print("=" * 55)
i = df.ndim
print(i)  # Output: 2

# -----------------------------------------------------------------------------
# 16. df.shape
#     Returns a TUPLE (number_of_rows, number_of_columns).
#     df.shape[0] → total number of rows
#     df.shape[1] → total number of columns
#     One of the most commonly used properties in data science.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("SHAPE — Rows and Columns:")
print("=" * 55)
t = df.shape
print("Shape (rows, columns):", t)
print("Total Rows           :", df.shape[0])
print("Total Columns        :", df.shape[1])

# -----------------------------------------------------------------------------
# 17. df.size
#     Returns the TOTAL NUMBER OF ELEMENTS in the DataFrame.
#     Calculated as: number_of_rows × number_of_columns
#     Example: shape (100, 3) → size = 300
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("SIZE — Total Number of Elements (rows × cols):")
print("=" * 55)
d = df.size
print(d)

# -----------------------------------------------------------------------------
# 18. df.values
#     Returns the DataFrame as a RAW NUMPY ARRAY.
#     All column labels and index are stripped — only the data remains.
#     Useful when passing data to machine learning libraries (scikit-learn,
#     TensorFlow, etc.) that expect NumPy arrays as input.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("VALUES — NumPy Array Representation:")
print("=" * 55)
a = df.values
print(a)

# -----------------------------------------------------------------------------
# 19. df.copy()
#     Creates a DEEP COPY of the DataFrame.
#     Changes made to the copy do NOT affect the original.
#     Always use .copy() when you want to modify a subset or version of
#     your data without disturbing the source.
# -----------------------------------------------------------------------------
df = df.copy()

# -----------------------------------------------------------------------------
# 20. df.sort_values(by='column')
#     Sorts the DataFrame by the values in the specified column.
#     Default order is ASCENDING (lowest → highest).
#     Use ascending=False for descending order.
#     The original DataFrame is NOT modified — result is returned.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("SORT_VALUES — Sorted by 'resume_score' (Ascending):")
print("=" * 55)
p = df.sort_values(by='resume_score')
print(p)


# =============================================================================
#  SECTION 2 : TIPS DATASET — LABEL ENCODING
#  Columns    : total_bill, tip, sex, smoker, day, time, size
# =============================================================================

# -----------------------------------------------------------------------------
# 21. pd.read_csv() — load Tips dataset
# -----------------------------------------------------------------------------
df = pd.read_csv("/Users/apple/data science/Advance_Python/Pandas/Day_4_Pandas/tips - tips.csv")
print("\n" + "=" * 55)
print("TIPS DATASET — Original:")
print("=" * 55)
print(df.head())

# -----------------------------------------------------------------------------
# 22. Series.map(dictionary)  — LABEL ENCODING
#     Replaces each value in a column using a key → value mapping dict.
#     This converts CATEGORICAL TEXT values into NUMBERS so that
#     machine learning models (which need numeric input) can process them.
#
#     Encoding 'sex' column:
#       "Female" → 1
#       "Male"   → 2
# -----------------------------------------------------------------------------
df['sex'] = df['sex'].map({"Female": 1, "Male": 2})
print("\n" + "=" * 55)
print("ENCODING — 'sex' column encoded:")
print("=" * 55)
print(df.head())

# -----------------------------------------------------------------------------
# 23. Encoding 'smoker' column:
#       "Yes" → 1
#       "No"  → 0
# -----------------------------------------------------------------------------
df['smoker'] = df['smoker'].map({"Yes": 1, "No": 0})
print("\n" + "=" * 55)
print("ENCODING — 'smoker' column encoded:")
print("=" * 55)
print(df.head())

# -----------------------------------------------------------------------------
# 24. Encoding 'day' column:
#       "Sun"  → 1
#       "Sat"  → 2
#       "Thur" → 3
#       "Fri"  → 4
# -----------------------------------------------------------------------------
df['day'] = df['day'].map({"Sun": 1, "Sat": 2, "Thur": 3, "Fri": 4})
print("\n" + "=" * 55)
print("ENCODING — 'day' column encoded:")
print("=" * 55)
print(df.head())

# -----------------------------------------------------------------------------
# 25. Encoding 'time' column:
#       "Dinner" → 1
#       "Lunch"  → 0
# -----------------------------------------------------------------------------
df['time'] = df['time'].map({"Dinner": 1, "Lunch": 0})
print("\n" + "=" * 55)
print("ENCODING — 'time' column encoded:")
print("=" * 55)
print(df.head())

# -----------------------------------------------------------------------------
# 26. df.shape — verify dataset dimensions after encoding
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("SHAPE — Tips Dataset Dimensions:")
print("=" * 55)
print(df.shape)

# -----------------------------------------------------------------------------
# 27. df.iloc[:4, :3] — top-left corner of Tips dataset
#     Selects first 4 rows and first 3 columns using integer positions.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("ILOC — Top-Left Corner of Tips (rows 0-3, cols 0-2):")
print("=" * 55)
top_left_corner_df = df.iloc[:4, :3]
print(top_left_corner_df)

# -----------------------------------------------------------------------------
# 28. df.loc[1:3, ['total_bill', 'tip']]
#     Selects rows 1 to 3 (inclusive) and only the columns
#     'total_bill' and 'tip' by their label names.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("LOC — Rows 1 to 3, Columns: total_bill & tip:")
print("=" * 55)
print(df.loc[1:3, ['total_bill', 'tip']])

# -----------------------------------------------------------------------------
# 29. df.sort_values(by='total_bill')
#     Sorts the Tips dataset by 'total_bill' in ascending order.
#     Useful for identifying lowest and highest spending customers.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("SORT_VALUES — Sorted by 'total_bill' (Ascending):")
print("=" * 55)
p = df.sort_values(by='total_bill')
print(p)