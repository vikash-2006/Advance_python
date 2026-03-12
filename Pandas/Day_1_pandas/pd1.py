# =============================================================================
#  PANDAS - Data Manipulation Library
# =============================================================================
#  Definition:
#  Pandas is an open-source Python library used for data manipulation and
#  analysis. It provides powerful data structures to work with structured
#  (tabular) data easily and efficiently.
#
#  Install: pip install pandas openpyxl
# =============================================================================

import pandas as pd   # 'pd' is the standard alias used by everyone


# =============================================================================
#  DATA STRUCTURES IN PANDAS
# =============================================================================

# -----------------------------------------------------------------------------
#  (1) SERIES  — 1-Dimensional Data Structure
# -----------------------------------------------------------------------------
#  Definition:
#  A Series is a one-dimensional array-like object that can hold any data type
#  (integers, strings, floats, etc.). It has an index (0, 1, 2...) but NO
#  column name. Think of it as a single column of a spreadsheet.
#
#  Syntax: pd.Series([values])
# -----------------------------------------------------------------------------

series_example = pd.Series([1, 2, 3, 4, 5])

print("---- SERIES ----")
print(series_example)
print("Type:", type(series_example))   # <class 'pandas.core.series.Series'>
print()


# -----------------------------------------------------------------------------
#  (2) DATAFRAME  — 2-Dimensional Data Structure
# -----------------------------------------------------------------------------
#  Definition:
#  A DataFrame is a two-dimensional table with rows and columns — just like
#  an Excel spreadsheet or a SQL table. Each column in a DataFrame is a Series.
#  It stores both VALUES and COLUMN NAMES.
#
#  Syntax: pd.DataFrame(dictionary)
# -----------------------------------------------------------------------------

# Step 1: Create a dictionary (raw data)
employee_data = {
    "EmpId"      : [1, 2, 3, 4, 5],
    "Name"       : ["Sam", "Raj", "Rahul", "Nikhil", "Mohit"],
    "Department" : ["HR", "IT", "SALES", "IT", "HR"]
}

# Step 2: Convert dictionary into a DataFrame
df = pd.DataFrame(employee_data)

print("---- DATAFRAME ----")
print(df)
print()


# =============================================================================
#  EXPORTING DATA  —  Save DataFrame to a File
# =============================================================================

# -----------------------------------------------------------------------------
#  Export to CSV
# -----------------------------------------------------------------------------
#  Definition:
#  CSV (Comma-Separated Values) is a plain text file format where each row is
#  a line and values are separated by commas. Use index=False to avoid saving
#  the row numbers (0,1,2...) as an extra column.
#
#  Syntax: df.to_csv("filename.csv", index=False)
# -----------------------------------------------------------------------------

df.to_csv("Emp_data.csv", index=False)
print("✔ Data exported to Emp_data.csv")


# -----------------------------------------------------------------------------
#  Export to Excel
# -----------------------------------------------------------------------------
#  Definition:
#  Saves the DataFrame as a .xlsx (Microsoft Excel) file. Requires the
#  'openpyxl' library. Install it with: pip install openpyxl
#
#  Syntax: df.to_excel("filename.xlsx", index=False)
# -----------------------------------------------------------------------------

df.to_excel("Emp_data.xlsx", index=False)
print("✔ Data exported to Emp_data.xlsx")
print()


# =============================================================================
#  IMPORTING DATA  —  Load a File into a DataFrame
# =============================================================================

# -----------------------------------------------------------------------------
#  Read from Excel
# -----------------------------------------------------------------------------
#  Definition:
#  Reads an Excel (.xlsx) file and loads it into a DataFrame for analysis.
#  Change the file path to match where your file is saved on your computer.
#
#  Syntax: pd.read_excel("filepath")
# -----------------------------------------------------------------------------

excel_df = pd.read_excel("Emp_data.xlsx")   # <-- Change path if needed

print("---- DATA READ FROM EXCEL ----")
print(excel_df)
print()


# -----------------------------------------------------------------------------
#  Read from CSV
# -----------------------------------------------------------------------------
#  Definition:
#  Reads a CSV file and loads it into a DataFrame. This is the most common
#  way to load datasets (e.g., downloaded from Kaggle, Google Sheets, etc.)
#
#  Syntax: pd.read_csv("filepath")
#
#  NOTE: Replace the path below with the actual path to your CSV file.
#        Example for Windows : "C:/Users/YourName/Downloads/flipkart.csv"
#        Example for Mac/Linux: "/home/yourname/Downloads/flipkart.csv"
# -----------------------------------------------------------------------------

# csv_df = pd.read_csv("flipkart.csv")   # <-- Uncomment and set your path
# print("---- FLIPKART CSV DATA ----")
# print(csv_df)


# =============================================================================
#  QUICK REFERENCE
# =============================================================================
#
#  pd.Series([...])             → Create a 1D array
#  pd.DataFrame({...})          → Create a 2D table from a dictionary
#  df.to_csv("file.csv")        → Save DataFrame as CSV
#  df.to_excel("file.xlsx")     → Save DataFrame as Excel
#  pd.read_csv("file.csv")      → Load CSV into DataFrame
#  pd.read_excel("file.xlsx")   → Load Excel into DataFrame
#  df.head()                    → Show first 5 rows
#  df.tail()                    → Show last 5 rows
#  df.shape                     → Show (rows, columns) count
#  df.info()                    → Show column types and null counts
#  df.describe()                → Show statistics (mean, min, max, etc.)
#
# =============================================================================