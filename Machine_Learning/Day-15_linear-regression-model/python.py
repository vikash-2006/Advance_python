# %% [markdown]
# # Machine Learning Workflow and Concepts
# 
# **Workflow of Machine Learning:**
# Data Ingestion ➔ Data Cleaning and Preprocessing ➔ Select Best ML Model According to data ➔ Model Training ➔ Model Prediction ➔ Model Performance Evaluation
# 
# ### Types of ML Models:
# 1. **Supervised ML Models** ➔ In this, we will train our labelled data.
# 2. **Unsupervised ML Models** ➔ In this, we will train our unlabelled data.
# 
# ### Supervised ML Models based on Target Data:
# 1. **Numerical:** LinearRegression, Decision Tree Regressor, Random Forest Regressor
# 2. **Categorical:** LogisticRegression, Decision Tree Classifier, Random Forest Classifier, Naive Bayes, SVM, KNN

# %% [markdown]
# ## 1. Linear Regression Model (Insurance Dataset)
# Note: Ensure the dataset is in the same folder as this script, or update the file path.

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Data Ingestion
# Changed path from Colab ("/content/...") to a local relative path
df_insurance = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-3/insurance - insurance.csv")
print("Insurance Data Head:")
print(df_insurance.head())

# %%
# 2. Data Cleaning and Preprocessing
lb = LabelEncoder()

df_insurance['sex'] = lb.fit_transform(df_insurance['sex'])
df_insurance['smoker'] = lb.fit_transform(df_insurance['smoker'])
df_insurance['region'] = lb.fit_transform(df_insurance['region'])

print("\nProcessed Insurance Data:")
print(df_insurance.head())

# %%
# 3. Model Training & Evaluation
x = df_insurance.drop(columns=['charges'])
y = df_insurance['charges']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Select and train model
lr_insurance = LinearRegression()
lr_insurance.fit(x_train, y_train)

# Model Prediction
y_pred = lr_insurance.predict(x_test)

# Model Performance Evaluation
r2 = r2_score(y_test, y_pred)
print(f"R2 Score for Insurance Dataset: {r2:.4f}")

# %% [markdown]
# ## 2. Linear Regression Model (Tips Dataset)

# %%
# 1. Data Ingestion
df_tips = pd.read_csv("/Users/apple/data science/Advance_Python/Machine_Learning/Day-7/tips - tips.csv")
print("\nTips Data Head:")
print(df_tips.head())

# %%
# 2. Data Cleaning and Preprocessing
lb_tips = LabelEncoder()

df_tips['sex'] = lb_tips.fit_transform(df_tips['sex'])
df_tips['smoker'] = lb_tips.fit_transform(df_tips['smoker'])
df_tips['day'] = lb_tips.fit_transform(df_tips['day'])
df_tips['time'] = lb_tips.fit_transform(df_tips['time'])

print("\nProcessed Tips Data:")
print(df_tips.head())

# %%
# 3. Model Training & Evaluation
x_tips = df_tips.drop(columns=['total_bill'])
y_tips = df_tips['total_bill']

# Train-test split
x_train_tips, x_test_tips, y_train_tips, y_test_tips = train_test_split(x_tips, y_tips, test_size=0.2, random_state=42)

# Select and train model
lr_tips = LinearRegression()
lr_tips.fit(x_train_tips, y_train_tips)

# Model Prediction
y_pred_tips = lr_tips.predict(x_test_tips)

# Model Performance Evaluation
r2_tips = r2_score(y_test_tips, y_pred_tips)
print(f"R2 Score for Tips Dataset: {r2_tips:.4f}")
