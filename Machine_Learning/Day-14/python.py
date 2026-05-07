

# Machine Learning Pipeline using Scikit-learn
#  What is a Pipeline?

# A Pipeline in Machine Learning is used to automate and organize the complete workflow of data preprocessing and model training.

# Instead of writing separate code for:

# Handling missing values
# Encoding categorical data
# Feature scaling
# Training model

# we combine everything into one clean workflow using Pipeline.







# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-2/covid_toy - covid_toy.csv")

print(df.head())


# ==========================================
# SPLIT DATA
# ==========================================

X = df.drop(columns=['has_covid'])
y = df['has_covid']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# FEATURE TYPES
# ==========================================

categorical_features = ['gender', 'city']
numeric_features = ['age', 'fever']


# ==========================================
# NUMERIC PIPELINE
# ==========================================

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())

])


# ==========================================
# CATEGORICAL PIPELINE
# ==========================================

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])


# ==========================================
# COLUMN TRANSFORMER
# ==========================================

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)

    ]

)


# ==========================================
# FINAL PIPELINE
# ==========================================

model = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('classifier', LogisticRegression())

])


# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("COVID Model Trained Successfully")




import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# LOAD DATA
df = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-3/insurance - insurance.csv")

print(df.head())


# SPLIT DATA
X = df.drop(columns=['charges'])
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# FEATURE TYPES
categorical_features = ['sex', 'region', 'smoker']
numeric_features = ['age', 'children']


# NUMERIC PIPELINE
numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())

])


# CATEGORICAL PIPELINE
categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])


# PREPROCESSOR
preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)

    ]

)


# FINAL PIPELINE
model = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('regressor', LinearRegression())

])


# TRAIN MODEL
model.fit(X_train, y_train)

print("Insurance Model Trained Successfully")