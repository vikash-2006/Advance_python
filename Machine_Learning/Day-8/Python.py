import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
# Note: Update the path if your file is located elsewhere
file_path = "/Users/apple/data science/Advance_Python/Machine_Learning/Day-8/Churn_Modelling - Churn_Modelling (1).csv"
df1 = pd.read_csv(file_path)

# Display the first 5 rows and check for missing values
print("Initial Dataset:")
print(df1.head())
print("\nMissing Values:")
print(df1.isnull().sum())


# 1. Label Encoding

# Definition: Label Encoding converts categorical (text) variables into numerical formats by 
# assigning a unique integer (0, 1, 2, ...) to each unique category. Note: Label encoding 
# can sometimes trick machine learning algorithms into thinking there is a mathematical 
# order (e.g., 2 > 1) when there isn't. It is mostly recommended for target variables.

from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
lb = LabelEncoder()

# Apply Label Encoding to categorical columns
df1['Surname'] = lb.fit_transform(df1['Surname'])
df1['Geography'] = lb.fit_transform(df1['Geography'])
df1['Gender'] = lb.fit_transform(df1['Gender'])

print("After Label Encoding:")
print(df1.head())


# 2. Ordinal Encoding

# Definition: Similar to Label Encoding, but Ordinal Encoding is used when there is a meaningful,
# natural order to the categories (e.g., "Low", "Medium", "High"). You can explicitly define the 
# order of the categories.

from sklearn.preprocessing import OrdinalEncoder

# Load a fresh copy of the dataset
df2 = pd.read_csv(file_path)

# Isolate the categorical columns we want to encode
df2 = df2.drop(columns=['Surname','RowNumber','CustomerId',
                        'CreditScore','Age','Tenure','Balance','NumOfProducts',
                        'NumOfProducts','IsActiveMember','EstimatedSalary','Exited','HasCrCard'])

# Define the order of the categories
categories_order = [['France', 'Germany', 'Spain'], ['Male', 'Female']]

# Initialize OrdinalEncoder with the specified order
oe = OrdinalEncoder(categories=categories_order)

# Fit and transform the data
encoded_array = oe.fit_transform(df2)

# Convert the resulting NumPy array back into a pandas DataFrame
df2_encoded = pd.DataFrame(encoded_array, columns=df2.columns)

print("After Ordinal Encoding:")
print(df2_encoded.head())



# 3. Encoding with GetDummies (Pandas)
# Definition: pd.get_dummies is a quick way to perform One-Hot Encoding. 
# It converts categorical variables into dummy/indicator variables (binary 0s and 1s). 
# Why drop_first=True? It drops one category per feature to avoid the "Dummy Variable Trap" 
# (multicollinearity), which can confuse certain models like Linear Regression.

# Load a fresh copy of the dataset
df3 = pd.read_csv(file_path)

# Standard get_dummies
a = pd.get_dummies(df3, columns=['Surname', 'Geography', 'Gender'])
a = a.astype(int)

# get_dummies dropping the first column (Recommended for ML models)
b = pd.get_dummies(df3, columns=['Surname', 'Geography', 'Gender'], drop_first=True)
b = b.astype(int)

print("After get_dummies (Drop First = True):")
print(b.head())


# 4. One-Hot Encoding (Scikit-Learn)
# Definition: This is Scikit-Learn's equivalent to get_dummies. 
# It creates binary columns for each category. 
# It is generally preferred over get_dummies in machine learning pipelines 
# because it can "remember" the categories when you apply the same transformation to new, 
# unseen test data.

from sklearn.preprocessing import OneHotEncoder

# Load a fresh copy of the dataset
df4 = pd.read_csv(file_path)

# Initialize OneHotEncoder
# drop='first' avoids the dummy variable trap
# sparse_output=False ensures it returns a dense NumPy array rather than a sparse matrix
ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)

# Fit and transform specific columns
ohe_array = ohe.fit_transform(df4[['Geography', 'Gender']])

print("One-Hot Encoded Array (first 5 rows):")
print(ohe_array[:5])


# 5. Standardization (Z-Score Scaling)
# Definition: Standardization transforms your numeric features so they have 
# a mean (average) of 0 and a standard deviation of 1. It centers the data around 0.
# This is highly recommended for algorithms like Support Vector Machines (SVM), K-Means, 
# and Neural Networks.

from sklearn.preprocessing import StandardScaler

# Define features (X) and target (y) using the label-encoded dataframe (df1)
X = df1.drop(columns=['Exited'])
y = df1['Exited']


# Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Statistics BEFORE Standardization:")
print(np.round(X_train.describe(), 2))

# Initialize StandardScaler
sc = StandardScaler()

# Fit the scaler on the training data AND transform it
X_train_sc = sc.fit_transform(X_train)
# Transform the test data (never fit on test data to prevent data leakage)
X_test_sc = sc.transform(X_test)

# Convert back to DataFrame for better readability
new_X_train = pd.DataFrame(X_train_sc, columns=X_train.columns)

print("Data Statistics AFTER Standardization (Mean ~ 0, Std ~ 1):")
print(np.round(new_X_train.describe(), 2))


# 6. Normalization (Min-Max Scaling)
# Definition: Normalization scales your numeric features to 
# fit exactly within a specific range, usually between 0 and 1. 
# It is useful when your data does not follow a normal (Gaussian) distribution, 
# or for algorithms like K-Nearest Neighbors (KNN) or image processing.


from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
mn = MinMaxScaler()

print("\nData Statistics BEFORE Normalization:")
print(np.round(X_train.describe(), 2))

# Fit the scaler on the training data AND transform it
# (Note: Fixed a bug here from your original code where `sc` was used instead of `mn`)
X_train_mn = mn.fit_transform(X_train)
X_test_mn = mn.transform(X_test)

# Convert back to DataFrame for better readability
new_X_train_mn = pd.DataFrame(X_train_mn, columns=X_train.columns)

print("Data Statistics AFTER Normalization (Min = 0, Max = 1):")
print(np.round(new_X_train_mn.describe(), 2))