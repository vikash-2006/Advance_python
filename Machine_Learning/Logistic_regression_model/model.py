# Logistic Regression Models for Classification Problems
# 1) Covid Toy Dataset
# 2) Attrition Dataset
# 3) Churn Modelling Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# 1) COVID TOY DATASET
# =========================
print("\n--- COVID TOY DATASET ---")

df = pd.read_csv("data/covid_toy.csv")
df["fever"] = df["fever"].fillna(df["fever"].mean())

lb = LabelEncoder()
df["gender"] = lb.fit_transform(df["gender"])
df["cough"] = lb.fit_transform(df["cough"])
df["city"] = lb.fit_transform(df["city"])
df["has_covid"] = lb.fit_transform(df["has_covid"])

x = df.drop(columns=["has_covid"])
y = df["has_covid"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# =========================
# 2) ATTRITION DATASET
# =========================
print("\n--- ATTRITION DATASET ---")

df = pd.read_csv("data/Attrition.csv")

df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])

lb = LabelEncoder()
df["EducationField"] = lb.fit_transform(df["EducationField"])
df["Department"] = lb.fit_transform(df["Department"])
df["BusinessTravel"] = lb.fit_transform(df["BusinessTravel"])
df["Attrition"] = lb.fit_transform(df["Attrition"])
df["Gender"] = lb.fit_transform(df["Gender"])
df["JobRole"] = lb.fit_transform(df["JobRole"])
df["MaritalStatus"] = lb.fit_transform(df["MaritalStatus"])
df["OverTime"] = lb.fit_transform(df["OverTime"])

x = df.drop(columns=["Attrition"])
y = df["Attrition"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# =========================
# 3) CHURN MODELLING DATASET
# =========================
print("\n--- CHURN MODELLING DATASET ---")

df = pd.read_csv("data/Churn_Modelling.csv")

lb = LabelEncoder()
df["CustomerId"] = lb.fit_transform(df["CustomerId"])
df["Surname"] = lb.fit_transform(df["Surname"])
df["Geography"] = lb.fit_transform(df["Geography"])
df["Gender"] = lb.fit_transform(df["Gender"])

x = df.drop(columns=["Exited"])
y = df["Exited"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))