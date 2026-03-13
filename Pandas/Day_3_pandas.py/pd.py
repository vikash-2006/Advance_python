# =============================================================================
#  COVID DATA - EXPLORATORY DATA ANALYSIS (EDA)
#  File   : covid_eda.py
#  Dataset: covid_toy.csv
#  Columns: age, gender, fever, cough, city, has_covid
# =============================================================================

import pandas as pd

# -----------------------------------------------------------------------------
# 1. LOAD DATASET
#    pd.read_csv() reads a CSV file and converts it into a DataFrame.
#    A DataFrame is a 2D table with rows and columns, similar to an Excel sheet.
# -----------------------------------------------------------------------------
df = pd.read_csv("/Users/apple/data science/Advance_Python/Pandas/Day_3_pandas.py/covid_toy - covid_toy.csv")


# -----------------------------------------------------------------------------
# 2. df.head(n)
#    Returns the FIRST n rows of the DataFrame.
#    Default is 5 rows if no argument is passed.
#    Used to get a quick preview of the dataset.
# -----------------------------------------------------------------------------
print("=" * 55)
print("HEAD — First 2 Rows:")
print("=" * 55)
print(df.head(2))


# -----------------------------------------------------------------------------
# 3. df.tail(n)
#    Returns the LAST n rows of the DataFrame.
#    Default is 5 rows if no argument is passed.
#    Useful to check if the data ends correctly (no extra blank rows, etc.).
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("TAIL — Last 3 Rows:")
print("=" * 55)
print(df.tail(3))


# -----------------------------------------------------------------------------
# 4. df.sample(n)
#    Returns n RANDOMLY selected rows from the DataFrame.
#    Default is 1 row. Each run may give different rows.
#    Useful for spot-checking data without bias.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("SAMPLE — 1 Random Row:")
print("=" * 55)
print(df.sample())


# -----------------------------------------------------------------------------
# 5. df.dtypes
#    Shows the DATA TYPE of each column:
#      int64   → Integer numbers (e.g., age)
#      float64 → Decimal numbers (e.g., fever: 98.5)
#      object  → Text / String (e.g., gender, city)
#    Helps identify if any column needs type conversion.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("DTYPES — Data Type of Each Column:")
print("=" * 55)
print(df.dtypes)


# -----------------------------------------------------------------------------
# 6. df.info()
#    Prints a SUMMARY of the entire DataFrame including:
#      - Total rows and columns
#      - Column names and their data types
#      - Count of Non-Null (non-missing) values per column
#      - Memory usage
#    Great for detecting missing values at a glance.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("INFO — DataFrame Overview:")
print("=" * 55)
df.info()


# -----------------------------------------------------------------------------
# 7. df.describe()
#    Returns STATISTICAL SUMMARY for all numeric columns:
#      count  → Number of non-null values
#      mean   → Average value
#      std    → Standard Deviation (spread of data)
#      min    → Minimum value
#      25%    → 1st Quartile (25% of data is below this)
#      50%    → Median (middle value)
#      75%    → 3rd Quartile (75% of data is below this)
#      max    → Maximum value
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("DESCRIBE — Statistical Summary:")
print("=" * 55)
print(df.describe())


# -----------------------------------------------------------------------------
# 8. df['column'].value_counts()
#    Counts the FREQUENCY of each unique value in a categorical column.
#    Results are sorted from most frequent to least frequent.
#    Useful for understanding the distribution of categories.
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("VALUE COUNTS — Gender Column:")
print("=" * 55)
print(df['gender'].value_counts())

print("\n" + "=" * 55)
print("VALUE COUNTS — Cough Column:")
print("=" * 55)
print(df['cough'].value_counts())


# -----------------------------------------------------------------------------
# 9. ENCODING — df['column'].map({})
#    Machine Learning models only understand NUMBERS, not text.
#    .map() replaces each category label with a numeric value.
#
#    gender : Female → 1  |  Male   → 0
#    cough  : Mild   → 0  |  Strong → 1
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("ENCODING — Converting Categories to Numbers:")
print("=" * 55)

df['gender'] = df['gender'].map({"Female": 1, "Male": 0})
df['cough']  = df['cough'].map({"Mild": 0, "Strong": 1})

print("gender → Female=1, Male=0")
print("cough  → Mild=0,   Strong=1")
print("\nDataFrame after Encoding (first 5 rows):")
print(df.head())



df['city'] = df['city'].map({"Kolkata": 1, "Delhi": 2, "Mumbai" : 3, "Bangalore" : 4})
print("\nDataFrame after Encoding (first 5 rows):")
print(df.head())


df ['has_covid'] = df['has_covid'].map({"Yes" : 1, "No" : 0.})

print("\nDataFrame after Encoding (first 5 rows):")
print(df.head())

# =============================================================================
#  END OF FILE
# =============================================================================