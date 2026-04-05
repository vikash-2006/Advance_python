# =============================================================================
# INSURANCE DATA PREPROCESSING & TRAIN-TEST SPLIT
# Author  : Vikash Kumawat
# Topic   : Drop Columns + Label Encoding + Scaling + Train-Test Split
# Dataset : insurance.csv
# Goal    : Predict whether a person is a SMOKER or NOT (Classification)
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 1 — IMPORTS
# -----------------------------------------------------------------------------

import numpy as np          # Numerical operations — arrays, math, rounding
import pandas as pd         # DataFrame operations — load, clean, transform CSV

from sklearn.model_selection import train_test_split   # Split data into train/test
from sklearn.preprocessing   import StandardScaler     # Scale features to mean=0, std=1


# -----------------------------------------------------------------------------
# SECTION 2 — LOAD DATASET
# -----------------------------------------------------------------------------

# pd.read_csv() → reads a CSV file from disk → returns a Pandas DataFrame
df = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-3/insurance - insurance.csv")
# .head() → shows first 5 rows — use this to quickly inspect column names & data
print("=== First 5 Rows ===")
print(df.head())

# Dataset columns:
# age      → integer  : age of the insured person
# sex      → text     : 'male' or 'female'
# bmi      → float    : Body Mass Index (weight / height²)
# children → integer  : number of dependent children
# smoker   → text     : 'yes' or 'no' ← THIS IS OUR TARGET
# region   → text     : geographic region (northeast, northwest, etc.)
# charges  → float    : medical insurance charges (in USD)


# -----------------------------------------------------------------------------
# SECTION 3 — CHECK MISSING VALUES
# -----------------------------------------------------------------------------

# .isnull()     → returns True/False for each cell (True = missing)
# .sum()        → counts True values per column → gives count of NaN per column
print("\n=== Missing Values Per Column ===")
print(df.isnull().sum())

# If all values are 0 → no missing data → no fillna() needed for this dataset


# -----------------------------------------------------------------------------
# SECTION 4 — DROP IRRELEVANT COLUMN
# -----------------------------------------------------------------------------

# DEFINITION: Not all columns are useful for prediction.
# 'region' is a nominal text column with 4 unique values.
# For simplicity, we drop it here (in advanced projects: use One-Hot Encoding).

# df.drop(columns=[...]) → removes the listed columns and returns new DataFrame
# inplace=False by default → must reassign to df
df = df.drop(columns=['region'])

print("\n=== After Dropping 'region' Column ===")
print(df.head())


# -----------------------------------------------------------------------------
# SECTION 5 — EXPLORE CATEGORICAL COLUMNS
# -----------------------------------------------------------------------------

# .value_counts() → count of each unique value in a Series
# Tells us how many male vs female rows exist before encoding
print("\n=== Sex Value Counts ===")
print(df['sex'].value_counts())


# -----------------------------------------------------------------------------
# SECTION 6 — LABEL ENCODING USING .map()
# -----------------------------------------------------------------------------

# DEFINITION: ML models only understand numbers, NOT text.
# Label Encoding converts text categories into integer values.
# We use .map(dict) to manually assign numbers — gives us full control.

# --- 6.1 Encode 'sex' → female=0, male=1
df['sex'] = df['sex'].map({'female': 0, 'male': 1})

# --- 6.2 Encode 'smoker' → yes=1, no=0  ← our TARGET column
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

print("\n=== After Label Encoding ===")
print(df.head())


# -----------------------------------------------------------------------------
# SECTION 7 — DEFINE FEATURES (X) AND TARGET (y)
# -----------------------------------------------------------------------------

# DEFINITION:
# X (Features/Input)  → the columns the model LEARNS from
# y (Target/Output)   → the column the model tries to PREDICT

# In this project: we predict whether someone is a smoker
# → y = 'smoker'   (binary: 0 or 1)
# → X = everything else (age, sex, bmi, children, charges)

# x → drop the target column 'smoker' from the DataFrame
x = df.drop(columns=['smoker'])

# y → select ONLY the target column
y = df['smoker']

print("\n=== Feature Matrix X (first 5 rows) ===")
print(x.head())

print("\n=== Target Vector y (first 5 values) ===")
print(y.head())

print("\n=== Shapes ===")
print(f"X shape : {x.shape}")   # (rows, columns) → e.g., (1338, 5)
print(f"y shape : {y.shape}")   # (rows,)         → e.g., (1338,)


# -----------------------------------------------------------------------------
# SECTION 8 — TRAIN-TEST SPLIT
# -----------------------------------------------------------------------------

# DEFINITION:
# We split the data into:
# → Training Set (80%) : model LEARNS patterns from this
# → Test Set     (20%) : we EVALUATE the model on unseen data

# Why split? → To check if the model generalizes to NEW data
# (not just memorized the training data — called "overfitting")

# Parameters:
# test_size=0.2    → 20% rows go to test, 80% to train
# random_state=42  → fixed seed so you get the same split every run

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

print("\n=== Split Sizes ===")
print(f"x_train : {x_train.shape}")   # 80% of total rows
print(f"x_test  : {x_test.shape}")    # 20% of total rows
print(f"y_train : {y_train.shape}")
print(f"y_test  : {y_test.shape}")

# Quick summary stats on training features (BEFORE scaling)
print("\n=== x_train Describe (Before Scaling) ===")
print(np.round(x_train.describe(), 2))

# Look at the 'mean' row:
# → age mean ≈ 39   → bmi mean ≈ 30   → charges mean ≈ 13000+
# These are on very different scales → StandardScaler will fix this


# -----------------------------------------------------------------------------
# SECTION 9 — STANDARD SCALING (Feature Normalization)
# -----------------------------------------------------------------------------

# DEFINITION: StandardScaler standardizes features using Z-score formula:
#
#             z = (x - mean) / standard_deviation
#
# After scaling:
#   → mean  ≈ 0.00  for every column
#   → std   ≈ 1.00  for every column
#
# WHY SCALE?
# → 'charges' ranges from 1000–65000
# → 'children' ranges from 0–5
# Without scaling, 'charges' dominates the model unfairly.
# Scaling puts all features on the same level playing field.

# --- 9.1 Create StandardScaler object
sc = StandardScaler()

# --- 9.2 fit_transform on TRAINING data
# fit      → LEARN mean & std from x_train data
# transform→ APPLY z-score using those learned values
# ⚠️ ALWAYS fit ONLY on training data — NEVER on test data
x_train_sc = sc.fit_transform(x_train)
# x_train_sc is a NumPy array (no column names yet)
print("\n=== Scaled Array (x_train_sc) — first 3 rows ===")
print(np.round(x_train_sc[:3], 4))

# --- 9.3 Convert scaled NumPy array back to DataFrame
# We pass columns=x_train.columns to restore column names
x_train_new = pd.DataFrame(x_train_sc, columns=x_train.columns)

print("\n=== x_train After Scaling (DataFrame) ===")
print(np.round(x_train_new, 2).head())

# --- 9.4 IMPORTANT: Scale test data using the SAME scaler
# Use .transform() only — do NOT call .fit_transform() on test data
# That would cause DATA LEAKAGE (model learns from test statistics)
x_test_sc  = sc.transform(x_test)
x_test_new = pd.DataFrame(x_test_sc, columns=x_test.columns)

# --- 9.5 Verify: check stats after scaling
print("\n=== x_train Stats After Scaling ===")
print(np.round(x_train_new.describe(), 2))
# Expected after scaling:
# mean ≈ 0.00 for all columns ✅
# std  ≈ 1.00 for all columns ✅

print("\n=== Preprocessing Complete! ===")
print("Next → Train a classifier on (x_train_new, y_train)")
print("Models to try: LogisticRegression | DecisionTreeClassifier | RandomForestClassifier")
