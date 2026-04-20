# =============================================================================
# DAY - 9 : COFFEE SALES DATASET — ENCODING + SCALING + PANDAS OPERATIONS
# Author  : Vikash Kumawat
# Topic   : LabelEncoder | OrdinalEncoder | StandardScaler | MinMaxScaler
#           + iloc[] filtering + df.rename()
# Dataset : Coffe_sales.csv
# =============================================================================

# =============================================================================
# DATASET OVERVIEW — COFFEE SALES
# =============================================================================
# Rows    : Coffee shop transaction records
# Target  : money (price paid) → REGRESSION task
#
# Columns:
#   Date         → transaction date         → DROP (not useful as-is)
#   Time         → transaction time         → DROP (not useful as-is)
#   cash_type    → card / cash              → ENCODE (categorical)
#   coffee_name  → type of coffee ordered   → ENCODE (categorical)
#   money        → amount paid              → TARGET (float)
#   hour_of_day  → hour of purchase         → numerical
#   Weekday      → Mon/Tue/Wed...           → ENCODE (categorical)
#   Weekdaysort  → numeric sort for weekday → DROP (redundant)
#   Month_name   → Jan/Feb/Mar...           → ENCODE (categorical)
#   Monthsort    → numeric sort for month   → DROP (redundant)
#   Time_of_Day  → Morning/Afternoon/Night  → ENCODE (ordinal!)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing   import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

CSV_PATH = "Coffe_sales - Coffe_sales.csv"


# =============================================================================
# ── SECTION 1: LOAD DATASET & EXPLORE
# =============================================================================

print("=" * 60)
print("  SECTION 1 — LOAD & EXPLORE")
print("=" * 60)

# pd.read_csv() → reads CSV file into DataFrame
df = pd.read_csv(CSV_PATH)

print("\n--- First 5 Rows ---")
print(df.head())

# .isnull().sum() → count NaN values per column
print("\n--- Missing Values ---")
print(df.isnull().sum())

# .value_counts() → frequency of each unique value
print("\n--- cash_type Distribution ---")
print(df['cash_type'].value_counts())


# =============================================================================
# ── SECTION 2: LABEL ENCODING
# =============================================================================

print("\n" + "=" * 60)
print("  SECTION 2 — LABEL ENCODING")
print("=" * 60)

# DEFINITION: LabelEncoder converts text categories → integers ALPHABETICALLY
# Each unique value gets a unique number (0, 1, 2, ...)
# Reuse same lb object — it relearns for each column

lb = LabelEncoder()

# cash_type  : card / cash → card=0, cash=1  (alphabetical)
df['cash_type'] = lb.fit_transform(df['cash_type'])

# coffee_name: multiple coffee types → 0,1,2... alphabetically
df['coffee_name'] = lb.fit_transform(df['coffee_name'])

# Time_of_Day: Afternoon / Morning / Night → 0,1,2  (alphabetical)
# NOTE: This has MEANINGFUL ORDER (Morning < Afternoon < Night)
# OrdinalEncoder would be better — but LabelEncoder shown here for practice
df['Time_of_Day'] = lb.fit_transform(df['Time_of_Day'])

# Weekday    : Mon,Tue,Wed... → alphabetical encoding
df['Weekday'] = lb.fit_transform(df['Weekday'])

# Month_name : Apr,Aug,Dec... → alphabetical (not calendar order!)
# NOTE: Month has a natural calendar order — OrdinalEncoder preferred
df['Month_name'] = lb.fit_transform(df['Month_name'])

print("\n--- After LabelEncoder ---")
print(df.head())
print("\n✅ Section 2 (LabelEncoder) Done!")


# =============================================================================
# ── SECTION 3: ORDINAL ENCODING
# =============================================================================

print("\n" + "=" * 60)
print("  SECTION 3 — ORDINAL ENCODING")
print("=" * 60)

# DEFINITION: OrdinalEncoder — YOU define the order for each column
# Great for columns like months (Jan=0, Feb=1...) or time of day
# Works on MULTIPLE columns at once

# Load fresh dataset for clean encoding
df1 = pd.read_csv(CSV_PATH)

print("\nAll columns:", df1.columns.tolist())
print(df1.head())

# Drop columns we don't need for this encoding demo
# Keeping only: cash_type, Time_of_Day, Month_name
df1 = df1.drop(columns=[
    'hour_of_day', 'money', 'Weekdaysort', 'Monthsort',
    'Date', 'Time', 'coffee_name', 'Weekday'
])

print("\n--- Remaining Columns ---")
print(df1.columns.tolist())

# Explore distributions before encoding
print("\ncash_type:", df1['cash_type'].value_counts().to_dict())
print("Time_of_Day:", df1['Time_of_Day'].value_counts().to_dict())
print("Month_name:", df1['Month_name'].value_counts().to_dict())

# DEFINE CATEGORY ORDER:
# categories_order MUST match column order in df1
# df1 columns: cash_type, Time_of_Day, Month_name
#
# cash_type  : only 'card' here (cash already filtered?) → [['card']]
#              NOTE: if 'cash' also exists, add it: [['card','cash']]
# Time_of_Day: Afternoon=0, Morning=1, Night=2 (your defined order)
# Month_name : specific business order (by sales volume or calendar)

categories_order = [
    ['card'],                                            # cash_type
    ['Afternoon', 'Morning', 'Night'],                   # Time_of_Day
    ['Mar','Oct','Feb','Sep','Aug',                      # Month_name
     'Dec','Nov','May','Jul','Jun','Jan','Apr']
]

oe = OrdinalEncoder(categories=categories_order)

# fit_transform → learn categories + encode → returns 2D NumPy array
encoded_array = oe.fit_transform(df1)

print("\n--- Encoded Array (first 5 rows) ---")
print(encoded_array[:5])

# Convert back to DataFrame with column names
df1_encoded = pd.DataFrame(encoded_array, columns=df1.columns)
print("\n--- df1 After OrdinalEncoder ---")
print(df1_encoded.head())
print("\n✅ Section 3 (OrdinalEncoder) Done!")


# =============================================================================
# ── SECTION 4: STANDARDSCALER (Standardization)
# =============================================================================

print("\n" + "=" * 60)
print("  SECTION 4 — STANDARDSCALER (Standardization)")
print("=" * 60)

# DEFINITION: StandardScaler → z = (x - mean) / std_dev
# Result: mean ≈ 0, std ≈ 1 for every column
# Best for: data with outliers, linear models, SVM

# First: drop Date and Time (non-numeric, can't scale)
# df was already LabelEncoded in Section 2
df = df.drop(columns=['Date', 'Time'])

# Define X (features) and y (target)
# TARGET = 'money' → continuous float → REGRESSION task
x = df.drop(columns=['money'])
y = df['money']

print(f"X shape: {x.shape}  |  y shape: {y.shape}")
print("Task: REGRESSION — predicting coffee price (continuous)")

# Train-Test Split (80/20)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print("\n--- x_train BEFORE Scaling ---")
print(x_train.describe().round(2))

# Apply StandardScaler
sc = StandardScaler()
x_train_sc  = sc.fit_transform(x_train)   # fit on train + transform
x_test_sc   = sc.transform(x_test)         # transform only (no fit!)

new_x_train = pd.DataFrame(x_train_sc, columns=x_train.columns)

print("\n--- x_train AFTER StandardScaler ---")
print(np.round(new_x_train.describe(), 2))
# Check: mean ≈ 0.00, std ≈ 1.00 for all columns ✅

print("\n✅ Section 4 (StandardScaler) Done!")


# =============================================================================
# ── SECTION 5: MINMAXSCALER (Normalization)
# =============================================================================

print("\n" + "=" * 60)
print("  SECTION 5 — MINMAXSCALER (Normalization)")
print("=" * 60)

# DEFINITION: MinMaxScaler → x_scaled = (x - min) / (max - min)
# Result: all values between 0.0 and 1.0
# Best for: bounded data, KNN, Neural Networks

# BUG FIX: Original code used sc.fit_transform() inside MinMaxScaler block
# sc = StandardScaler (already defined above) — wrong scaler called!
# FIX: use mn.fit_transform()

x1 = df.drop(columns=['money'])
y1 = df['money']

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, test_size=0.2, random_state=42
)

print("\n--- x1_train BEFORE Scaling ---")
print(x1_train.describe().round(2))

mn = MinMaxScaler()
x1_train_mn  = mn.fit_transform(x1_train)   # ✅ mn not sc
x1_test_mn   = mn.transform(x1_test)

new_x1_train = pd.DataFrame(x1_train_mn, columns=x1_train.columns)

print("\n--- x1_train AFTER MinMaxScaler ---")
print(np.round(new_x1_train.describe(), 2))
# Check: min = 0.00, max = 1.00 for all columns ✅

print("\n✅ Section 5 (MinMaxScaler) Done!")


# =============================================================================
# ── SECTION 6: FILTER ROWS USING iloc[]
# =============================================================================

print("\n" + "=" * 60)
print("  SECTION 6 — FILTER ROWS USING iloc[]")
print("=" * 60)

# DEFINITION: iloc[] = integer location based indexing
# → Select rows/columns by their INTEGER POSITION (not label)
# → df.iloc[start:stop] → rows from index start up to (but NOT including) stop
#
# df.iloc[30:41] → rows at positions 30, 31, 32, ... 40
#                  (41 is excluded — Python slicing rule)

print("\n--- Rows at Index 30 to 40 (iloc[30:41]) ---")
print(df.iloc[30:41])

# Additional iloc examples:
print("\n--- First 3 rows: df.iloc[0:3] ---")
print(df.iloc[0:3])

print("\n--- Last 5 rows: df.iloc[-5:] ---")
print(df.iloc[-5:])

print("\n✅ Section 6 (iloc[] filtering) Done!")


# =============================================================================
# ── SECTION 7: RENAME A COLUMN
# =============================================================================

print("\n" + "=" * 60)
print("  SECTION 7 — RENAME COLUMN")
print("=" * 60)

print("\n--- Before rename ---")
print(df.head())

# DEFINITION: df.rename(columns={'old_name': 'new_name'})
# → Renames columns in the DataFrame
# → inplace=True → modifies df directly (no need to reassign)
# → inplace=False (default) → returns new DataFrame, must reassign

df.rename(columns={'money': 'paise'}, inplace=True)

print("\n--- After rename (money → paise) ---")
print(df.head())

# Verify column was renamed
print(f"\nColumns: {df.columns.tolist()}")

print("\n✅ Section 7 (rename) Done!")

print("\n" + "=" * 60)
print("  ALL SECTIONS COMPLETE!")
print("  Encodings: LabelEncoder | OrdinalEncoder")
print("  Scaling  : StandardScaler | MinMaxScaler")
print("  Pandas   : iloc[] | rename()")
print("  Next     → Model Training (LinearRegression / RandomForest)")
print("=" * 60)
