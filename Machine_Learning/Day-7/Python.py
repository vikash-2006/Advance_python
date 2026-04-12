# =============================================================================
# DAY - 7 : ENCODING-3 (get_dummies) + ENCODING-4 (OneHotEncoder)
# Author  : Vikash Kumawat
# Topic   : pd.get_dummies | drop_first | OneHotEncoder | 3 Datasets
# Datasets: insurance.csv  |  covid_toy.csv  |  tips.csv
# =============================================================================

# =============================================================================
# CONCEPT : ONE-HOT ENCODING — WHY DO WE NEED IT?
# =============================================================================
#
# Problem with LabelEncoder / OrdinalEncoder on NOMINAL columns:
#   city: Bangalore=0, Delhi=1, Kolkata=2, Mumbai=3
#   → Model thinks Mumbai(3) > Kolkata(2) > Delhi(1) > Bangalore(0)
#   → But cities have NO real order! This misleads the model.
#
# SOLUTION → One-Hot Encoding:
#   Instead of one column with 0,1,2,3 → create SEPARATE BINARY columns
#   city_Bangalore: [1,0,0,0]
#   city_Delhi    : [0,1,0,0]
#   city_Kolkata  : [0,0,1,0]
#   city_Mumbai   : [0,0,0,1]
#   Each row gets exactly one 1 — no false ordering!
#
# TWO WAYS TO ONE-HOT ENCODE:
#   1. pd.get_dummies()   → pandas method, quick & easy
#   2. OneHotEncoder()    → sklearn method, better for ML pipelines
#
# MULTICOLLINEARITY PROBLEM & drop_first:
#   If city has 4 values → 4 binary columns created
#   But: city_Mumbai = 1 - (city_Bangalore + city_Delhi + city_Kolkata)
#   → The 4th column is REDUNDANT — perfectly predictable from the other 3
#   → This creates MULTICOLLINEARITY (columns that carry duplicate info)
#   → Causes OVERFITTING — model memorizes instead of learning
#   FIX: drop_first=True → drops the first binary column (N-1 columns total)
#
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
# ENCODING METHOD 3 — pd.get_dummies()
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
# =============================================================================


# =============================================================================
# DATASET 1 — INSURANCE  →  get_dummies (with and without drop_first)
# =============================================================================

print("=" * 65)
print("  DATASET 1 — INSURANCE  |  pd.get_dummies()")
print("=" * 65)

df = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-3/insurance - insurance.csv")
print(df.head(2))

# insurance text columns: sex (2 vals), smoker (2 vals), region (4 vals)


# ── WITHOUT drop_first ────────────────────────────────────────

# DEFINITION: pd.get_dummies(df, columns=[...])
# → Finds all unique values in each listed column
# → Creates one new binary (0/1) column per unique value
# → Original text column is REMOVED, replaced by binary columns
# → New column name format: originalcolumn_value
#
# sex has 2 values    → creates: sex_female, sex_male
# smoker has 2 values → creates: smoker_no, smoker_yes
# region has 4 values → creates: region_northeast, region_northwest,
#                                 region_southeast, region_southwest

a = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

# .astype(int) → converts True/False boolean to 1/0 integers
# get_dummies returns bool by default — .astype(int) makes it cleaner
a = a.astype(int)

print(f"\n--- WITHOUT drop_first ---")
print(f"Original columns: {df.shape[1]}  |  After get_dummies: {a.shape[1]}")
print(f"Columns: {list(a.columns)}")
print(a.head(2))

# Total new columns created:
# sex    : 2 values  → 2 columns (sex_female, sex_male)
# smoker : 2 values  → 2 columns (smoker_no, smoker_yes)
# region : 4 values  → 4 columns (region_northeast, _northwest, _southeast, _southwest)
# Original: age, bmi, children, charges = 4 columns
# Total: 4 + 2 + 2 + 4 = 12 columns


# ── WITH drop_first ───────────────────────────────────────────

# DEFINITION: drop_first=True
# → Drops the FIRST binary column for each encoded column
# → For sex (2 values): keeps sex_male only (drops sex_female)
#   → if sex_male=0 → person is female (we can infer!)
#   → if sex_male=1 → person is male
# → For region (4 values): keeps 3 columns (drops region_northeast)
#   → if all 3 are 0 → person is from northeast (we can infer!)
#
# WHY drop_first=True?
# → Removes MULTICOLLINEARITY (redundant columns)
# → Prevents OVERFITTING — model doesn't get confused by duplicate info
# → Reduces number of features — faster training

b = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
b = b.astype(int)

print(f"\n--- WITH drop_first=True ---")
print(f"Original columns: {df.shape[1]}  |  After get_dummies: {b.shape[1]}")
print(f"Columns: {list(b.columns)}")
print(b.head(2))

# With drop_first:
# sex    : 2 → 1 column (sex_male)
# smoker : 2 → 1 column (smoker_yes)
# region : 4 → 3 columns (region_northwest, region_southeast, region_southwest)
# Total: 4 + 1 + 1 + 3 = 9 columns (vs 12 without)

print("\n✅ Dataset 1 (Insurance) Done!")


# =============================================================================
# DATASET 2 — COVID TOY  →  get_dummies on all categorical columns
# =============================================================================

print("\n" + "=" * 65)
print("  DATASET 2 — COVID TOY  |  pd.get_dummies()")
print("=" * 65)

df = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-2/covid_toy - covid_toy.csv")
print(df.head())

df = df.dropna()

# Categorical columns: gender (2), cough (2), city (4), has_covid (2)
# Numerical columns : age, fever (kept as-is)


# ── WITHOUT drop_first ────────────────────────────────────────
a = pd.get_dummies(df, columns=['gender', 'cough', 'city', 'has_covid'])
a = a.astype(int)

print(f"\n--- WITHOUT drop_first ---")
print(f"Original: {df.shape[1]} cols  →  After: {a.shape[1]} cols")
print(f"Columns created: {list(a.columns)}")
print(a.head(2))

# gender   : 2 values → gender_Female, gender_Male
# cough    : 2 values → cough_Mild, cough_Strong
# city     : 4 values → city_Bangalore, city_Delhi, city_Kolkata, city_Mumbai
# has_covid: 2 values → has_covid_No, has_covid_Yes
# + age, fever (numerical, unchanged)
# Total: 2 + 2 + 2 + 4 + 2 = 12 columns


# ── WITH drop_first ───────────────────────────────────────────
b = pd.get_dummies(df, columns=['gender', 'cough', 'city', 'has_covid'], drop_first=True)
b = b.astype(int)

print(f"\n--- WITH drop_first=True ---")
print(f"Original: {df.shape[1]} cols  →  After: {b.shape[1]} cols")
print(f"Columns: {list(b.columns)}")
print(b.head(2))

# With drop_first:
# gender   : 2 → 1 (gender_Male, drops gender_Female)
# cough    : 2 → 1 (cough_Strong, drops cough_Mild)
# city     : 4 → 3 (drops city_Bangalore)
# has_covid: 2 → 1 (has_covid_Yes, drops has_covid_No)
# Total: 2 + 1 + 1 + 3 + 1 = 8 columns

print("\n✅ Dataset 2 (COVID) Done!")


# =============================================================================
# DATASET 3 — TIPS  →  get_dummies + OneHotEncoder
# =============================================================================

print("\n" + "=" * 65)
print("  DATASET 3 — TIPS  |  get_dummies + OneHotEncoder")
print("=" * 65)

df = pd.read_csv("tips - tips.csv")
print(df.head())

# Tips categorical: sex (2), smoker (2), day (4), time (2)
# Tips numerical : total_bill, tip, size (unchanged)


# ── PART A: pd.get_dummies without drop_first ─────────────────
a = pd.get_dummies(df, columns=['sex', 'smoker', 'day', 'time'])
a = a.astype(int)

print(f"\n--- Tips WITHOUT drop_first ---")
print(f"Original: {df.shape[1]} cols  →  After: {a.shape[1]} cols")
print(a.head(2))

# sex  : 2 → sex_Female, sex_Male
# smoker:2 → smoker_No, smoker_Yes
# day  : 4 → day_Fri, day_Sat, day_Sun, day_Thur
# time : 2 → time_Dinner, time_Lunch


# ── PART B: pd.get_dummies WITH drop_first ────────────────────
b = pd.get_dummies(df, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)
b = b.astype(int)

print(f"\n--- Tips WITH drop_first=True ---")
print(f"Original: {df.shape[1]} cols  →  After: {b.shape[1]} cols")
print(b.head(2))
print(f"Columns: {list(b.columns)}")


# =============================================================================
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
# ENCODING METHOD 4 — sklearn OneHotEncoder
# Applied on: Tips dataset (sex, smoker, day, time)
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
# =============================================================================

print("\n" + "=" * 65)
print("  ENCODING-4 — OneHotEncoder (sklearn)")
print("=" * 65)

# DEFINITION: OneHotEncoder from sklearn.preprocessing
# → Same concept as get_dummies: creates binary columns per unique value
# → ADVANTAGE over get_dummies: integrates with sklearn Pipelines
# → Can be used inside Pipeline + ColumnTransformer for clean ML workflows
# → Works on 2D input (DataFrame or array), not whole df at once
#
# Parameters:
#   drop='first'        → same as drop_first=True in get_dummies
#                          drops first category to avoid multicollinearity
#   sparse_output=False → returns a regular NumPy array (not sparse matrix)
#                          sparse=True saves memory for huge datasets
#   dtype=np.int32      → output as integers (not float64 by default)

df = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-7/tips - tips.csv")   # reload fresh

ohe = OneHotEncoder(
    drop='first',          # drop first category per column → avoids multicollinearity
    sparse_output=False,   # return dense array (easier to work with)
    dtype=np.int32         # output as int, not float
)

# .fit_transform() on selected categorical columns only
# Pass as DataFrame — OneHotEncoder needs 2D input
ohe_array = ohe.fit_transform(df[['sex', 'smoker', 'day', 'time']])

# ohe_array is a NumPy array — no column names yet
# Get the actual column names generated by OneHotEncoder
ohe_col_names = ohe.get_feature_names_out(['sex', 'smoker', 'day', 'time'])

print(f"\nOneHotEncoder column names: {ohe_col_names}")

# Convert to DataFrame with proper column names
ohe_df = pd.DataFrame(ohe_array, columns=ohe_col_names)
print(f"\nOneHotEncoder result shape: {ohe_df.shape}")
print(ohe_df.head(3))

# ── Combine with numerical columns ───────────────────────────
# Reset index to align properly before concatenating
df_reset = df.reset_index(drop=True)
ohe_df   = ohe_df.reset_index(drop=True)

# Keep numerical columns: total_bill, tip, size
num_cols = df_reset[['total_bill', 'tip', 'size']]

# Combine: numerical + one-hot encoded categorical
df_final = pd.concat([num_cols, ohe_df], axis=1)

print(f"\nFinal combined DataFrame: {df_final.shape}")
print(df_final.head(3))

# ── Verify encoding ──────────────────────────────────────────
# Check what categories were learned per column
print("\n--- Categories learned by OHE ---")
for col, cats in zip(['sex','smoker','day','time'], ohe.categories_):
    print(f"  {col}: {list(cats)}")

print("\n✅ OneHotEncoder (Encoding-4) Complete!")
print("\n" + "=" * 65)
print("  ALL ENCODING METHODS COVERED:")
print("  Day1-3: .map()  |  Day4-5: LabelEncoder")
print("  Day6: OrdinalEncoder  |  Day7: get_dummies + OneHotEncoder")
print("=" * 65)
