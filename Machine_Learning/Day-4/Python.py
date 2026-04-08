# =============================================================================
# DAY - 3 : NORMALIZATION USING MinMaxScaler
# Author  : Vikash Kumawat
# Topic   : MinMaxScaling | dropna() | Two Dataset Practice
# Datasets: covid_toy.csv  &  placement.csv
# =============================================================================

# =============================================================================
# CONCEPT : WHAT IS NORMALIZATION ?
# =============================================================================
#
# Normalization = scaling all values in a column to a FIXED RANGE → [0, 1]
#
# REAL-WORLD EXAMPLES (why we need it):
#
#   Play Store App Downloads:
#       App A  →  650 downloads
#       App B  →  4,200 downloads
#       App C  →  2,300,000 downloads
#   If you run any ML model directly, the 2.3 million value DOMINATES.
#   Visualization becomes a zig-zag mess because of the extreme range.
#
#   Employee Salary:
#       Junior  →  5,000 Rs
#       Mid     →  50,000 Rs
#       Senior  →  2,00,000 Rs
#       Manager →  10,00,000 Rs
#   Without scaling, salary will dominate all other features (age, exp, etc.)
#
# SOLUTION → MinMaxScaler
#   Formula :  x_scaled = (x - x_min) / (x_max - x_min)
#   Result  :  Every value lands between 0.0 and 1.0
#   min → becomes 0.0   |   max → becomes 1.0   |   rest → proportional
#
# DIFFERENCE from StandardScaler:
#   StandardScaler  → mean=0, std=1  (no fixed range, can go negative)
#   MinMaxScaler    → min=0, max=1   (strict bounded range, always positive)
#   Use MinMaxScaler when: data has no strong outliers & you need 0-1 range
#   Use StandardScaler  when: data has outliers or follows normal distribution
#
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# DATASET 1 — COVID TOY DATASET
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

print("=" * 60)
print("  DATASET 1 — COVID TOY")
print("=" * 60)

# -----------------------------------------------------------------------------
# STEP 0 — LOAD DATASET
# -----------------------------------------------------------------------------

df = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-2/covid_toy - covid_toy.csv")

# .head(5) → shows first 5 rows
print("\n--- First 5 Rows ---")
print(df.head(5))


# -----------------------------------------------------------------------------
# STEP 1 — CHECK AND HANDLE MISSING VALUES
# -----------------------------------------------------------------------------

# DEFINITION: .isnull().sum() → counts NaN in each column
print("\n--- Missing Values ---")
print(df.isnull().sum())

# DEFINITION: .dropna() → removes ALL rows that have at least one NaN value
# Used here instead of .fillna() because we want to keep only clean rows.
# Difference:
#   .fillna(value) → replaces NaN with a value (keeps all rows)
#   .dropna()      → deletes rows that contain NaN (loses some data)
df = df.dropna()

print(f"\nRows after dropna(): {df.shape[0]}")


# -----------------------------------------------------------------------------
# STEP 2 — DROP IRRELEVANT COLUMN
# -----------------------------------------------------------------------------

# 'city' is a nominal text column with multiple categories.
# Dropped here for simplicity (in real projects → use One-Hot Encoding)
df = df.drop(columns=['city'])

print("\n--- After Dropping 'city' ---")
print(df.head(3))


# -----------------------------------------------------------------------------
# STEP 3 — LABEL ENCODING (Text → Numbers)
# -----------------------------------------------------------------------------

# DEFINITION: ML models only understand numbers.
# .map(dict) → converts text labels to integers manually.

df['gender']    = df['gender'].map({"Male": 0, "Female": 1})
df['cough']     = df['cough'].map({"Mild": 0, "Strong": 1})
df['has_covid'] = df['has_covid'].map({"Yes": 1, "No": 0})
# Note: has_covid uses "Yes"/"No" (capital N) — different from Day-1 "NO"

print("\n--- After Label Encoding ---")
print(df.head(2))


# -----------------------------------------------------------------------------
# STEP 4 — SPLIT INTO FEATURES (X) AND TARGET (y)
# -----------------------------------------------------------------------------

# X → input columns the model learns from (all except target)
# y → output column the model predicts
x = df.drop(columns=['has_covid'])   # Features
y = df['has_covid']                   # Target: has COVID? (0 or 1)

print(f"\nX shape: {x.shape}  |  y shape: {y.shape}")


# -----------------------------------------------------------------------------
# STEP 5 — TRAIN-TEST SPLIT
# -----------------------------------------------------------------------------

# 80% training, 20% testing | random_state=42 for reproducibility
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print("\n--- x_train Stats BEFORE MinMaxScaling ---")
print(np.round(x_train.describe(), 2))
# Look at 'min' and 'max' rows — values are NOT in 0-1 range yet


# -----------------------------------------------------------------------------
# STEP 6 — APPLY MinMaxScaler
# -----------------------------------------------------------------------------

# DEFINITION: MinMaxScaler scales each feature to range [0, 1]
# Formula: x_scaled = (x - x_min) / (x_max - x_min)
#
# .fit_transform(x_train) →
#   fit      : learns x_min and x_max from TRAINING data
#   transform: applies formula to produce 0-1 values
#
# IMPORTANT RULE (same as StandardScaler):
#   fit_transform() → ONLY on x_train
#   transform()     → ONLY on x_test  (use same min/max learned from train)

mn = MinMaxScaler()                        # Create scaler object
x_train_mn  = mn.fit_transform(x_train)   # Learn + scale training data
x_train_new = pd.DataFrame(             # Convert array back to DataFrame
    x_train_mn,
    columns=x_train.columns
)

print("\n--- x_train Stats AFTER MinMaxScaling ---")
print(np.round(x_train_new.describe(), 2))
# After scaling:
# min row → 0.00 for ALL columns ✅
# max row → 1.00 for ALL columns ✅
# mean    → somewhere between 0 and 1 ✅

# Scale test data using SAME scaler (DO NOT fit again)
x_test_mn  = mn.transform(x_test)
x_test_new = pd.DataFrame(x_test_mn, columns=x_test.columns)

print("\n✅ Dataset 1 (COVID) — MinMaxScaling Done!")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# DATASET 2 — PLACEMENT DATASET
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

print("\n" + "=" * 60)
print("  DATASET 2 — PLACEMENT")
print("=" * 60)

# Why a second dataset?
# Placement data has columns like cgpa (0-10), iq (80-140), etc.
# These are on very different scales → perfect use case for MinMaxScaler

# -----------------------------------------------------------------------------
# STEP 1 — LOAD DATASET
# -----------------------------------------------------------------------------

df2 = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-1/placement - placement.csv")

print("\n--- First 4 Rows ---")
print(df2.head(4))

# Columns expected: cgpa, iq, placed
# 'placed' = target (1 = got placed, 0 = not placed)


# -----------------------------------------------------------------------------
# STEP 2 — SPLIT INTO X AND y
# -----------------------------------------------------------------------------

# No missing values or encoding needed for this dataset (already numeric)
x2 = df2.drop(columns=['placed'])    # Features: cgpa, iq
y2 = df2['placed']                    # Target: placed (0 or 1)

print(f"\nX shape: {x2.shape}  |  y shape: {y2.shape}")


# -----------------------------------------------------------------------------
# STEP 3 — TRAIN-TEST SPLIT
# -----------------------------------------------------------------------------

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, test_size=0.2, random_state=42
)

print("\n--- x2_train Stats BEFORE MinMaxScaling ---")
print(np.round(x2_train.describe(), 2))
# cgpa range  → 0 to 10  (smaller scale)
# iq range    → 80 to 140 (larger scale)
# Without scaling, iq dominates cgpa in distance-based models (KNN, SVM)


# -----------------------------------------------------------------------------
# STEP 4 — APPLY MinMaxScaler
# -----------------------------------------------------------------------------

# Fresh scaler object for this dataset
mn2 = MinMaxScaler()
x2_train_mn  = mn2.fit_transform(x2_train)
x2_train_new = pd.DataFrame(x2_train_mn, columns=x2_train.columns)

print("\n--- x2_train Stats AFTER MinMaxScaling ---")
print(np.round(x2_train_new.describe(), 2))
# Both cgpa and iq now scaled to 0-1 range ✅
# Model will treat them with equal importance ✅

x2_test_mn  = mn2.transform(x2_test)
x2_test_new = pd.DataFrame(x2_test_mn, columns=x2_test.columns)

print("\n✅ Dataset 2 (Placement) — MinMaxScaling Done!")

print("\n" + "=" * 60)
print("  PREPROCESSING COMPLETE — READY TO TRAIN MODELS")
print("  Next → LogisticRegression / KNN / DecisionTree")
print("=" * 60)
