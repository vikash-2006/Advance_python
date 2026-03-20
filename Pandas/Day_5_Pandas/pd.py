# =============================================================================
#  PANDAS FUNDAMENTALS — PLACEMENT & TIPS DATASETS
#  File   : pandas_fundamentals.py
#  Topics : Type Conversion, Arithmetic, Filtering, Renaming,
#           Conditional Columns, Export, EDA, Data Types Theory
# =============================================================================

import pandas as pd
import numpy as np

# =============================================================================
#  SECTION 1 : PLACEMENT DATASET
#  Columns    : cgpa, resume_score, placed
# =============================================================================

# -----------------------------------------------------------------------------
# 1. pd.read_csv()
#    Reads a CSV file from disk and loads it into a pandas DataFrame.
#    A DataFrame is a 2D labeled table — like an Excel sheet — with rows
#    and columns. This is always the first step when working with CSV data.
# -----------------------------------------------------------------------------
df = pd.read_csv("/Users/apple/data science/Advance_Python/Pandas/Day_5_Pandas/placement - placement.csv")

print("=" * 55)
print("FULL DATAFRAME — PLACEMENT:")
print("=" * 55)
print(df)

# -----------------------------------------------------------------------------
# 2. df.astype()
#    Converts the data type of a column to a specified type.
#    Common types : int, float, str, bool
#    Here we convert 'cgpa' from float → int  (e.g. 8.7 becomes 8).
#    NOTE: This truncates (cuts off) the decimal part — it does NOT round.
# -----------------------------------------------------------------------------
df['cgpa'] = df['cgpa'].astype(int)

print("\n" + "=" * 55)
print("AFTER astype(int) — cgpa column converted to integer:")
print("=" * 55)
print(df)

# -----------------------------------------------------------------------------
# 3. Series.add()
#    Adds a scalar (single number) or another Series to every element
#    in a column. Equivalent to df['cgpa'] + 10.
#    Useful because it supports extra options like fill_value for NaNs.
# -----------------------------------------------------------------------------
df['cgpa'] = df['cgpa'].add(10)

print("\n" + "=" * 55)
print("AFTER add(10) — 10 added to every value in 'cgpa':")
print("=" * 55)
print(df)

# -----------------------------------------------------------------------------
# 4. Series.mean()
#    Returns the arithmetic average of all values in a column.
#    Formula : sum of all values ÷ total number of values
# -----------------------------------------------------------------------------
r = df['cgpa'].mean()
print("\n" + "=" * 55)
print(f"MEAN  of 'cgpa'  : {r}")
print("=" * 55)

# -----------------------------------------------------------------------------
# 5. Series.median()
#    Returns the middle value when all values are sorted in order.
#    Less affected by outliers than mean.
#    If even count → average of the two middle values.
# -----------------------------------------------------------------------------
s = df['cgpa'].median()
print(f"MEDIAN of 'cgpa' : {s}")

# -----------------------------------------------------------------------------
# 6. df.filter(items=[...])
#    Returns a DataFrame containing ONLY the specified column names.
#    Useful when you want to extract a subset of columns by name.
#    Alternative : df[['col1', 'col2']] — both do the same thing.
# -----------------------------------------------------------------------------
g = df.filter(items=['cgpa', 'placed'])

print("\n" + "=" * 55)
print("FILTER — only 'cgpa' and 'placed' columns:")
print("=" * 55)
print(g)

# Double-bracket indexing achieves the same result:
print("\nSame result using double-bracket indexing:")
print(df[['cgpa', 'placed']])

# -----------------------------------------------------------------------------
# 7. df.filter(items=[...], axis=0)
#    When axis=0, filter works on ROW INDEX labels (not columns).
#    items=[5, 6] returns rows whose index label is 5 or 6.
#    axis=0 → row-wise  |  axis=1 → column-wise (default)
# -----------------------------------------------------------------------------
s = df.filter(items=[5, 6], axis=0)

print("\n" + "=" * 55)
print("FILTER axis=0 — rows with index labels 5 and 6:")
print("=" * 55)
print(s)

# -----------------------------------------------------------------------------
# 8. df.to_string()
#    Converts the entire DataFrame into a single plain-text string.
#    Useful for logging, saving to .txt files, or printing without truncation.
#    Does NOT save to disk — it just returns a string object in memory.
# -----------------------------------------------------------------------------
v = df.to_string()
print("\n" + "=" * 55)
print("to_string() — type of result:", type(v))
print("=" * 55)

# df.to_csv("path/output.csv")  →  saves the DataFrame to a CSV file on disk

# -----------------------------------------------------------------------------
# 9. df.columns
#    Returns an Index object containing all column labels of the DataFrame.
#    Useful for inspecting structure or programmatically referencing columns.
# -----------------------------------------------------------------------------
idx = df.columns
print("\n" + "=" * 55)
print("COLUMN INDEX OBJECT:", idx)
print("=" * 55)

# First column label
label = df.columns[0]
print(f"First column label : {label}")

# Convert to a Python list
p = df.columns.tolist()
print(f"Columns as list    : {p}")

# Convert to a NumPy array
q = df.columns.values
print(f"Columns as array   : {q}")

# -----------------------------------------------------------------------------
# 10. df.rename(columns={...})
#     Renames one or more columns using a dictionary mapping.
#     Format : { 'old_name' : 'new_name' }
#     Returns a NEW DataFrame — does NOT modify the original unless
#     you pass inplace=True or reassign the result.
# -----------------------------------------------------------------------------
p = df.rename(columns={'cgpa': 'updated_cgpa', 'resume_score': 'semester_marks'})

print("\n" + "=" * 55)
print("RENAMED COLUMNS — cgpa → updated_cgpa, resume_score → semester_marks:")
print("=" * 55)
print(p)

# -----------------------------------------------------------------------------
# 11. Series.where(condition, other=...)
#     Keeps the original value where the condition is TRUE.
#     Replaces with `other` wherever the condition is FALSE.
#     Here : keep cgpa value if cgpa > 17, else put 0.
#     Creates a new column 'half' without modifying 'cgpa'.
# -----------------------------------------------------------------------------
df['half'] = df['cgpa'].where(df['cgpa'] > 17, other=0)

print("\n" + "=" * 55)
print("WHERE — new 'half' column (cgpa if > 17, else 0) — first 10 rows:")
print("=" * 55)
print(df.head(10))


# =============================================================================
#  SECTION 2 : TIPS DATASET
#  Columns    : total_bill, tip, sex, smoker, day, time, size
# =============================================================================

print("\n\n" + "=" * 55)
print("SECTION 2 — TIPS DATASET")
print("=" * 55)

df = pd.read_csv("/Users/apple/data science/Advance_Python/Pandas/Day_5_Pandas/tips - tips.csv")
print(df)

# -----------------------------------------------------------------------------
# 12. df.iloc[start:stop]
#     Integer-Location based indexing — selects rows by POSITION (0-based).
#     iloc[11:14] returns rows at positions 11, 12, 13  (stop is exclusive).
#     Use iloc when you know the row number, not the label.
# -----------------------------------------------------------------------------
data = df.iloc[11:14]

print("\n" + "=" * 55)
print("ILOC — rows 11 to 13 (positions):")
print("=" * 55)
print(data)

# -----------------------------------------------------------------------------
# 13. df.rename() — renaming specific columns
#     'sex'  → 'Gender'
#     'size' → 'Total_size'
# -----------------------------------------------------------------------------
r = df.rename(columns={'sex': 'Gender', 'size': 'Total_size'})

print("\n" + "=" * 55)
print("RENAMED — sex → Gender, size → Total_size:")
print("=" * 55)
print(r)

# -----------------------------------------------------------------------------
# 14. astype(int) on 'total_bill'
#     Converts the float column to integer (removes decimal portion).
#     e.g. 16.99 → 16
# -----------------------------------------------------------------------------
df['total_bill'] = df['total_bill'].astype(int)

print("\n" + "=" * 55)
print("AFTER astype(int) on 'total_bill':")
print("=" * 55)
print(df)

# -----------------------------------------------------------------------------
# 15. df.dtypes
#     Returns the data type of EACH column in the DataFrame.
#     Common dtypes :
#       int64   → whole numbers
#       float64 → decimal numbers
#       object  → strings / mixed types
#       bool    → True / False
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("DTYPES — data type of each column:")
print("=" * 55)
print(df.dtypes)

# -----------------------------------------------------------------------------
# 16. df.head(n)
#     Returns the FIRST n rows of the DataFrame (default n=5).
#     Quick way to preview data without printing everything.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("HEAD(6) — first 6 rows:")
print("=" * 55)
print(df.head(6))

# -----------------------------------------------------------------------------
# 17. df.to_csv() / df.to_excel()
#     Exports the DataFrame to a file on disk.
#     index=False → do not write the row numbers (0, 1, 2…) as a column.
#     to_excel() requires the 'openpyxl' library: pip install openpyxl
# -----------------------------------------------------------------------------
df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx", index=False)
print("\nFiles saved: output.csv  |  output.xlsx")


# =============================================================================
#  SECTION 3 : DATA TYPES — THEORY & CLASSIFICATION
# =============================================================================
#
#  Understanding data types is the FOUNDATION of data science. Every column
#  in a dataset belongs to one of these two broad categories:
#
#  ┌────────────────────────────────────────────────────────────────────────┐
#  │  1. NUMERICAL (Quantitative) Data                                      │
#  │     → Data that represents measurable quantities (numbers).            │
#  │                                                                        │
#  │     a) CONTINUOUS  — can take any value within a range, including      │
#  │                      decimals. Can be divided further.                 │
#  │        Examples : salary (₹20,500.60), weight (70.54 kg),              │
#  │                   distance (270.78 km), height, age (as float)         │
#  │                                                                        │
#  │     b) DISCRETE    — only specific, countable values (often integers). │
#  │                      Cannot be divided meaningfully.                   │
#  │        Examples : number of employees, number of children              │
#  │                                                                        │
#  ├────────────────────────────────────────────────────────────────────────┤
#  │  2. CATEGORICAL (Qualitative) Data                                     │
#  │     → Data that represents groups or labels (no math meaning).         │
#  │                                                                        │
#  │     a) NOMINAL  — categories with NO natural order.                    │
#  │        Examples : gender (M/F), marital status (married/unmarried),    │
#  │                   has_covid (Yes/No)                                   │
#  │                                                                        │
#  │     b) ORDINAL  — categories WITH a meaningful order/ranking.          │
#  │        Examples : education level (UG < PG < PhD),                     │
#  │                   customer rating (1★ < 2★ < 3★)                     │
#  └────────────────────────────────────────────────────────────────────────┘
#
#  WHY IT MATTERS IN PANDAS / ML:
#  → ML algorithms need numbers. Categorical columns must be ENCODED first.
#
#  LABEL ENCODING (map categorical → integer):
#  -------------------------------------------
#  has_covid column : 'Yes' → 1 , 'No' → 0
#
#  Method 1 — Using .map()  (manual, readable, full control)
#  Method 2 — Using pd.get_dummies()  (one-hot encoding for multiple labels)
# =============================================================================

# Example : Label Encoding with .map()
covid_data = pd.DataFrame({
    'patient_id' : [101, 102, 103, 104, 105],
    'has_covid'  : ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'bill_paid'  : [452.78, 310.50, 789.00, 123.45, 560.20]
})

print("\n" + "=" * 55)
print("SECTION 3 — LABEL ENCODING EXAMPLE")
print("=" * 55)
print("Before encoding:")
print(covid_data)

# .map() replaces each value using the dictionary provided.
# 'Yes' → 1  |  'No' → 0
covid_data['has_covid'] = covid_data['has_covid'].map({'Yes': 1, 'No': 0})

print("\nAfter Label Encoding  ( Yes→1 , No→0 ):")
print(covid_data)
print("\nData types after encoding:")
print(covid_data.dtypes)

# =============================================================================
#  END OF FILE
# =============================================================================