import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# ==========================================
# APPROACH 1: Encoding the Entire DataFrame
# ==========================================
print("--- APPROACH 1: Entire DataFrame ---")

# 1. Load the dataset
df_covid = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-2/covid_toy - covid_toy.csv")

# 2. Clean the data
df_covid = df_covid.dropna() # Remove missing values
# Drop columns we don't need for this specific encoding example
df_covid = df_covid.drop(columns=['city', 'age', 'fever']) 

# Remaining columns are assumed to be: ['gender', 'cough', 'has_covid']
print("Original Data:\n", df_covid.head(2))

# 3. Define the specific order for each column
# 'gender' -> Male, Female
# 'cough' -> Mild, Strong
# 'has_covid' -> No, Yes
categories_order = [['Male', 'Female'], ['Mild', 'Strong'], ['No', 'Yes']]

# 4. Initialize and apply Ordinal Encoder
oe = OrdinalEncoder(categories=categories_order)
encoded_array = oe.fit_transform(df_covid)

# 5. Convert the resulting numpy array back to a Pandas DataFrame
df_encoded = pd.DataFrame(encoded_array, columns=df_covid.columns)
print("\nEncoded Data:\n", df_encoded.head(2))


# ==========================================
# APPROACH 2: Standard ML Split (X and y)
# ==========================================
print("\n--- APPROACH 2: Standard X and y Split ---")

# 1. Reload and clean data
df_covid2 = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-2/covid_toy - covid_toy.csv").dropna()
df_covid2 = df_covid2.drop(columns=['age', 'fever', 'city'])

# 2. Separate Features (X) and Target (y)
X = df_covid2.drop(columns=['has_covid']) # Features: gender, cough
y = df_covid2['has_covid']                # Target: has_covid

# 3. Encode Features (X) using OrdinalEncoder (Allows custom order)
oe_features = OrdinalEncoder(categories=[['Male', 'Female'], ['Mild', 'Strong']])
X_encoded = oe_features.fit_transform(X)

# 4. Encode Target (y) using LabelEncoder (Automatic 0, 1 mapping)
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print("\nX Encoded (Features):\n", X_encoded[:2])
print("y Encoded (Target):\n", y_encoded[:2])


# ==========================================
# APPROACH 3: Practice with 'Tips' Dataset
# ==========================================
print("\n--- APPROACH 3: Tips Dataset ---")

# Task: Target is 'total_bill'. Keep categorical columns, remove numerical, encode categoricals.

# 1. Load data
df_tips = pd.read_csv("/Users/apple/data science/Advance_Python/Pandas/Day_5_Pandas/tips - tips.csv").dropna()

# 2. Separate Target
y_tips = df_tips['total_bill']

# 3. Drop numerical columns (total_bill, tip, size) to leave only categorical columns
# (Note: Fixed from your original code to ensure we actually KEEP the categorical columns)
X_tips = df_tips.drop(columns=['total_bill', 'tip', 'size']) 
# Remaining categorical columns: 'sex', 'smoker', 'day', 'time'

print("Tips Categorical Features:\n", X_tips.head(2))

# 4. Apply OrdinalEncoder to all categorical columns
# If we don't specify 'categories', it will auto-assign integers alphabetically
oe_tips = OrdinalEncoder()
X_tips_encoded = oe_tips.fit_transform(X_tips)

# Convert back to DataFrame for readability
df_tips_encoded = pd.DataFrame(X_tips_encoded, columns=X_tips.columns)
print("\nTips Encoded Features:\n", df_tips_encoded.head(2))