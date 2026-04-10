# =============================================================================
# DAY - 5 : ENCODING + SCALING — 3 DATASET PRACTICE (CLEANED VERSION)
# Author  : Vikash Kumawat
# Topic   : LabelEncoder | StandardScaler | MinMaxScaler | Bug Fixes
# Datasets: covid_toy.csv  |  tips.csv  |  insurance.csv
# =============================================================================


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv("/Users/apple/data science/Advance_Python/Pandas/Day_5_Pandas/tips - tips.csv")
except FileNotFoundError:
    print("Error: 'tips.csv' not found. Please make sure the file is in the correct directory.")
    exit()

df = df.dropna()

# --- 2. Encode Categorical Features ---
lb = LabelEncoder()
df['sex'] = lb.fit_transform(df['sex'])
df['smoker'] = lb.fit_transform(df['smoker'])
df['day'] = lb.fit_transform(df['day'])
df['time'] = lb.fit_transform(df['time'])

# --- 3. Separate Features (X) and Target (y) ---
# In this case, we'll try to predict 'total_bill' based on other features
X = df.drop(columns=['total_bill'])
y = df['total_bill']

# --- 4. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Apply Feature Scaling (Normalization) ---
# MinMaxScaler scales data to a range between 0 and 1
mn = MinMaxScaler()
X_train_normalized = mn.fit_transform(X_train)
X_test_normalized = mn.transform(X_test) # Use the same scaler from training set

# Convert back to a DataFrame for easy viewing
X_train_normalized_df = pd.DataFrame(X_train_normalized, columns=X_train.columns)

print("--- Tips Dataset Analysis ---")
print("\nOriginal Data Head:")
print(df.head())
print("\nDescription of Scaled Training Data (Normalization):")
print(X_train_normalized_df.describe().round(2))