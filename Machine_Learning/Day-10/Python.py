"""
Column Transformer Demo
Author: You
Goal: Learn manual preprocessing vs ColumnTransformer in sklearn
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


# ============================================================
# 2. DATASET 1: COVID TOY DATASET
# ============================================================
print("\n" + "=" * 60)
print("DATASET 1: COVID TOY")
print("=" * 60)

# Load dataset
df_covid = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-2/covid_toy - covid_toy.csv")

# Basic checks
print("\nFirst 5 rows:")
print(df_covid.head())

print("\nMissing values:")
print(df_covid.isnull().sum())

# Split into features and target
X_train, X_test, y_train, y_test = train_test_split(
    df_covid.drop(columns=["has_covid"]),
    df_covid["has_covid"],
    test_size=0.2,
    random_state=42
)

print("\nX_train sample:")
print(X_train.head())


# ============================================================
# 3. MANUAL PREPROCESSING (STEP BY STEP)
# ============================================================
print("\n" + "-" * 60)
print("MANUAL PREPROCESSING")
print("-" * 60)

# 3.1 Imputation for fever column
si = SimpleImputer(strategy="mean") # Fill missing values with mean of the column
X_train_fever = si.fit_transform(X_train[["fever"]])
X_test_fever = si.transform(X_test[["fever"]])  # Only transform test data

# 3.2 Ordinal Encoding for cough
oe = OrdinalEncoder(categories=[["Mild", "Strong"]])
X_train_cough = oe.fit_transform(X_train[["cough"]])
X_test_cough = oe.transform(X_test[["cough"]])

# 3.3 One-Hot Encoding for gender and city
ohe = OneHotEncoder(drop="first", sparse_output=False)
X_train_gender_city = ohe.fit_transform(X_train[["gender", "city"]])
X_test_gender_city = ohe.transform(X_test[["gender", "city"]])

# 3.4 Extract age (remaining numerical column)
X_train_age = X_train.drop(columns=["gender", "fever", "cough", "city"]).values
X_test_age = X_test.drop(columns=["gender", "fever", "cough", "city"]).values

# 3.5 Concatenate all transformed features
X_train_manual = np.concatenate(
    (X_train_age, X_train_fever, X_train_gender_city, X_train_cough),
    axis=1
)

X_test_manual = np.concatenate(
    (X_test_age, X_test_fever, X_test_gender_city, X_test_cough),
    axis=1
)

print(f"Manual X_train shape: {X_train_manual.shape}")
print(f"Manual X_test shape: {X_test_manual.shape}")


# ============================================================
# 4. SAME TASK USING COLUMNTRANSFORMER (RECOMMENDED)
# ============================================================
print("\n" + "-" * 60)
print("COLUMN TRANSFORMER PREPROCESSING")
print("-" * 60)

transformer_covid = ColumnTransformer(
    transformers=[
        ("imputer_fever", SimpleImputer(strategy="mean"), ["fever"]),
        ("ordinal_cough", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
        ("onehot_gender_city", OneHotEncoder(drop="first", sparse_output=False), ["gender", "city"]),
    ],
    remainder="passthrough"  # keep other columns as they are (e.g., age)
)

X_train_ct = transformer_covid.fit_transform(X_train)
X_test_ct = transformer_covid.transform(X_test)

print(f"ColumnTransformer X_train shape: {X_train_ct.shape}")
print(f"ColumnTransformer X_test shape: {X_test_ct.shape}")


# ============================================================
# 5. DATASET 2: INSURANCE DATASET
# ============================================================
print("\n" + "=" * 60)
print("DATASET 2: INSURANCE")
print("=" * 60)

# Load dataset
df_insurance = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-3/insurance - insurance.csv")

# Basic checks
print("\nFirst 5 rows:")
print(df_insurance.head())

print("\nMissing values:")
print(df_insurance.isnull().sum())

# Split into features and target
X_train_ins, X_test_ins, y_train_ins, y_test_ins = train_test_split(
    df_insurance.drop(columns=["charges"]),
    df_insurance["charges"],
    test_size=0.2,
    random_state=42
)

# Apply ColumnTransformer
transformer_insurance = ColumnTransformer(
    transformers=[
        ("ordinal_sex", OrdinalEncoder(categories=[["male", "female"]]), ["sex"]),
        ("onehot_smoker_region", OneHotEncoder(drop="first", sparse_output=False), ["smoker", "region"]),
    ],
    remainder="passthrough"
)

X_train_ins_ct = transformer_insurance.fit_transform(X_train_ins)
X_test_ins_ct = transformer_insurance.transform(X_test_ins)

print(f"Insurance X_train transformed shape: {X_train_ins_ct.shape}")
print(f"Insurance X_test transformed shape: {X_test_ins_ct.shape}")

print("\n✅ Preprocessing completed successfully.")