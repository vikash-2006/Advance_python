# =============================================================================
# COVID-19 DATA PREPROCESSING & TRAIN-TEST SPLIT
# Author  : Vikash Kumawat
# Topic   : Data Encoding + Missing Value Handling + Scaling + Train-Test Split
# Dataset : covid_toy.csv
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 1 — IMPORTS
# -----------------------------------------------------------------------------

import numpy as np          # Numerical operations & array handling
import pandas as pd         # DataFrame operations (rows/columns)
from sklearn.model_selection import train_test_split   # Split data → train & test
from sklearn.preprocessing import StandardScaler       # Feature scaling (Z-score)


# -----------------------------------------------------------------------------
# SECTION 2 — LOAD DATASET
# -----------------------------------------------------------------------------

# pd.read_csv() → reads a CSV file and returns it as a DataFrame
df = pd.read_csv("Day-2/covid_toy - covid_toy.csv")

# .head() → shows first 5 rows by default — good for quick inspection
print("=== First 5 Rows ===")
print(df.head())


# -----------------------------------------------------------------------------
# SECTION 3 — EXPLORE CATEGORICAL COLUMNS
# -----------------------------------------------------------------------------

# .value_counts() → counts frequency of each unique value in a column
# Useful to understand distribution of categories before encoding

print("\n=== Cough Value Counts ===")
print(df['cough'].value_counts())

print("\n=== City Value Counts ===")
print(df['city'].value_counts())


# -----------------------------------------------------------------------------
# SECTION 4 — LABEL ENCODING (Manual using .map())
# -----------------------------------------------------------------------------

# DEFINITION: Label Encoding converts text categories into numbers.
# ML models cannot understand text — they only understand numbers.
# We use .map() to manually assign integer values to each category.

# ✅ Why manual map (not LabelEncoder)?
# → Manual map gives YOU control over which label gets which number.
# → LabelEncoder assigns numbers alphabetically — may not make sense logically.

# --- 4.1 Encode 'gender' → Male=0, Female=1
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# --- 4.2 Encode 'cough' → Mild=0, Strong=1
df['cough'] = df['cough'].map({'Mild': 0, 'Strong': 1})

# --- 4.3 Encode 'has_covid' (Target Column) → NO=0, Yes=1
df['has_covid'] = df['has_covid'].map({'NO': 0, 'Yes': 1})

# --- 4.4 Encode 'city' → Kolkata=0, Delhi=1, Bangalore=2, Mumbai=3
df['city'] = df['city'].map({'Kolkata': 0, 'Delhi': 1, 'Bangalore': 2, 'Mumbai': 3})

print("\n=== After Label Encoding ===")
print(df.head())


# -----------------------------------------------------------------------------
# SECTION 5 — HANDLE MISSING VALUES
# -----------------------------------------------------------------------------

# DEFINITION: Missing values (NaN) cause errors in ML models.
# Common strategy for numerical columns → fill with MEAN.
# This keeps the column's overall average unchanged.

# --- 5.1 Fill missing 'fever' values with the column mean
df['fever'] = df['fever'].fillna(df['fever'].mean())

# df['fever'].mean() → calculates average of non-null values
# .fillna()          → replaces all NaN with that value

print("\n=== Missing Values After Fix ===")
print(df.isnull().sum())   # Should show 0 for 'fever' now


# -----------------------------------------------------------------------------
# SECTION 6 — FEATURE MATRIX (X) & TARGET VECTOR (y)
# -----------------------------------------------------------------------------

# DEFINITION:
# X (Features) → the input columns the model learns FROM
# y (Target)   → the output column the model tries to PREDICT

# --- 6.1 X → drop target column 'has_covid'
x = df.drop(columns=['has_covid'])
print("\n=== Feature Matrix (X) ===")
print(x.head())

# --- 6.2 y → only the target column
y = df['has_covid']
print("\n=== Target Vector (y) ===")
print(y.head())


# -----------------------------------------------------------------------------
# SECTION 7 — TRAIN-TEST SPLIT
# -----------------------------------------------------------------------------

# DEFINITION: We split data into two sets:
# → Training Set : model LEARNS from this (80%)
# → Test Set     : model is EVALUATED on this (20%) — never seen before!

# Why split? → To check if the model generalizes to NEW data (not just memorizes)

# Parameters:
# test_size=0.2    → 20% goes to test, 80% to train
# random_state=42  → seed for reproducibility (same split every time you run)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

print("\n=== Split Sizes ===")
print(f"x_train : {x_train.shape}")   # Rows × Columns
print(f"x_test  : {x_test.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_test  : {y_test.shape}")

# Quick stats on training data (BEFORE scaling)
print("\n=== x_train Stats (Before Scaling) ===")
print(np.round(x_train.describe(), 2))


# -----------------------------------------------------------------------------
# SECTION 8 — STANDARD SCALING (Feature Normalization)
# -----------------------------------------------------------------------------

# DEFINITION: StandardScaler standardizes features using Z-score formula:
#             z = (x - mean) / std_dev
#
# → After scaling: mean ≈ 0, std_dev ≈ 1 for every column
# → This prevents columns with large numbers (e.g., fever=102) from
#   dominating columns with small numbers (e.g., cough=0 or 1)

# IMPORTANT RULE:
# → .fit_transform() on TRAIN data   → learns mean/std, then applies scaling
# → .transform()     on TEST data    → applies the SAME mean/std (no re-learning!)
# ⚠️ NEVER fit on test data → that would cause "data leakage"

# --- 8.1 Create the scaler object
sc = StandardScaler()

# --- 8.2 Fit on train data → learn parameters (mean, std) and transform
x_train_sc = sc.fit_transform(x_train)
# Returns a NumPy array → need to convert back to DataFrame

# --- 8.3 Convert scaled array back to DataFrame (to keep column names)
x_train_new = pd.DataFrame(x_train_sc, columns=x_train.columns)

print("\n=== x_train Stats (After Scaling) ===")
print(np.round(x_train_new.describe(), 2))

# ✅ After scaling:
# mean  → ~0.00 for all columns
# std   → ~1.00 for all columns
# This confirms StandardScaler worked correctly!

print("\n=== Pipeline Complete! ===")
print("Next Step → Train a model (e.g., Logistic Regression, Decision Tree)")
