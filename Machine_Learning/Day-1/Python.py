# =============================================================================
# MACHINE LEARNING - TRAIN TEST SPLIT
# Topic  : Splitting Data into Training and Testing Sets
# Author : Vikash
# Tool   : VS Code | Python 3.x
# Libs   : pandas, scikit-learn
# =============================================================================

# -----------------------------------------------------------------------------
# CONCEPT OVERVIEW
# -----------------------------------------------------------------------------
# In Machine Learning, we work with data that has two parts:
#   x  --> Input Features  (the columns used to PREDICT the answer)
#   y  --> Target / Label  (the column we WANT to predict)
#
# We split both x and y into Training and Testing portions:
#   x_train, y_train --> Fed to the model so it can LEARN patterns
#   x_test           --> Given to the model to make PREDICTIONS
#   y_test           --> The real answers we use to CHECK model accuracy
#
# Simple Analogy:
#   1+1=2,  1+2=3,  1+4=5,  1+6=7   --> Training examples (model learns from these)
#   1+7=?,  1+8=?                    --> Testing examples  (model predicts these)
#   Actual answers: 8, 9             --> y_test (used to verify predictions)
# -----------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split


# =============================================================================
# DATASET 1 : COVID TOY DATASET
# Goal      : Predict whether a person has COVID or not (has_covid column)
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 1 : Load the Dataset
# -----------------------------------------------------------------------------
# pd.read_csv() reads the CSV file and stores it as a DataFrame
df = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-1/covid_toy - covid_toy.csv")

# Preview first 5 rows to understand the structure
print("=" * 60)
print("COVID Dataset - First 5 Rows:")
print("=" * 60)
print(df.head(5))

# Check dimensions: (rows, columns)
print("\nDataset Shape (rows, columns):", df.shape)


# -----------------------------------------------------------------------------
# STEP 2 : Handle Missing Values
# -----------------------------------------------------------------------------
# isnull().sum() shows how many null/NaN values exist per column
print("\nMissing Values Before Filling:")
print(df.isnull().sum())

# The 'fever' column has missing values
# We fill them with the MEAN (average) of the fever column
# This is called Mean Imputation - a common way to handle missing numbers
df['fever'] = df['fever'].fillna(df['fever'].mean())

# Confirm no more nulls
print("\nMissing Values After Filling:")
print(df.isnull().sum())


# -----------------------------------------------------------------------------
# STEP 3 : Define Input (x) and Target (y)
# -----------------------------------------------------------------------------
# x = All columns EXCEPT the target column (these are the features/inputs)
# y = Only the target column (this is what we want to predict)

x = df.drop(columns=['has_covid'])   # Input features
y = df['has_covid']                  # Target label (Yes / No)

print("\nInput Features (x) - First 3 rows:")
print(x.head(3))

print("\nTarget Label (y) - First 5 values:")
print(y.head(5))


# -----------------------------------------------------------------------------
# STEP 4 : Train-Test Split
# -----------------------------------------------------------------------------
# train_test_split() randomly splits the data:
#   test_size=0.2   --> 20% of data goes to testing, 80% to training
#   random_state=42 --> Fixes the random seed so results are reproducible
#                       (every run gives the same split)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------------------------------------------------
# STEP 5 : Verify Split Shapes
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COVID DATASET - DATA SPLIT SUMMARY")
print("=" * 60)
print("Total  Data Shape  :", df.shape)
print("-" * 40)
print("Input  x    Shape  :", x.shape)
print("x_train     Shape  :", x_train.shape)
print("x_test      Shape  :", x_test.shape)
print("-" * 40)
print("Target y    Shape  :", y.shape)
print("y_train     Shape  :", y_train.shape)
print("y_test      Shape  :", y_test.shape)
print("=" * 60)


# =============================================================================
# DATASET 2 : PLACEMENT DATASET
# Goal      : Predict whether a student got placed or not (placed column)
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 1 : Load the Dataset
# -----------------------------------------------------------------------------
df = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-1/placement - placement.csv")

print("\n" + "=" * 60)
print("Placement Dataset - First 5 Rows:")
print("=" * 60)
print(df.head(5))

print("\nDataset Shape (rows, columns):", df.shape)


# -----------------------------------------------------------------------------
# STEP 2 : Check for Missing Values
# -----------------------------------------------------------------------------
print("\nMissing Values:")
print(df.isnull().sum())
# No missing values in this dataset, so no filling needed


# -----------------------------------------------------------------------------
# STEP 3 : Define Input (x) and Target (y)
# -----------------------------------------------------------------------------
x = df.drop(columns=['placed'])   # Input features (cgpa, iq, etc.)
y = df['placed']                  # Target: 1 = Placed, 0 = Not Placed

print("\nInput Features (x) - First 3 rows:")
print(x.head(3))

print("\nTarget Label (y) - First 5 values:")
print(y.head(5))


# -----------------------------------------------------------------------------
# STEP 4 : Train-Test Split
# -----------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------------------------------------------------
# STEP 5 : Verify Split Shapes
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PLACEMENT DATASET - DATA SPLIT SUMMARY")
print("=" * 60)
print("Total  Data Shape  :", df.shape)
print("-" * 40)
print("Input  x    Shape  :", x.shape)
print("x_train     Shape  :", x_train.shape)
print("x_test      Shape  :", x_test.shape)
print("-" * 40)
print("Target y    Shape  :", y.shape)
print("y_train     Shape  :", y_train.shape)
print("y_test      Shape  :", y_test.shape)
print("=" * 60)


# =============================================================================
# QUICK REVISION NOTES
# =============================================================================
# | Term          | Meaning                                           |
# |---------------|---------------------------------------------------|
# | x             | All input feature columns                         |
# | y             | Target/label column (what we predict)             |
# | x_train       | Features used to TRAIN the model                  |
# | y_train       | True labels used during TRAINING                  |
# | x_test        | Features given to model for PREDICTION            |
# | y_test        | True labels used to EVALUATE model accuracy       |
# | test_size=0.2 | 20% test data, 80% train data                     |
# | random_state  | Seed for reproducibility (same split every run)   |
# | fillna(mean)  | Replace NaN with column average (Mean Imputation) |
# =============================================================================