import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# 1) ATTRITION DATASET
# =========================
print("\n--- ATTRITION DATASET ---")

df = pd.read_csv("data/Attrition.csv")
print(df.head())

lb = LabelEncoder()

df["Attrition"] = lb.fit_transform(df["Attrition"])
df["BusinessTravel"] = lb.fit_transform(df["BusinessTravel"])
df["Department"] = lb.fit_transform(df["Department"])
df["EducationField"] = lb.fit_transform(df["EducationField"])
df["Gender"] = lb.fit_transform(df["Gender"])
df["JobRole"] = lb.fit_transform(df["JobRole"])
df["MaritalStatus"] = lb.fit_transform(df["MaritalStatus"])
df["Over18"] = lb.fit_transform(df["Over18"])
df["OverTime"] = lb.fit_transform(df["OverTime"])

x = df.drop(columns=["Attrition"])
y = df["Attrition"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

a = np.round(accuracy_score(y_test, y_pred), 2)
print("Accuracy (Attrition):", a)

# -------------------------
# Find object columns
# -------------------------
df_original = pd.read_csv("data/Attrition.csv")

b = []
for i in df_original.columns:
    if df_original[i].dtype == object:
        b.append(i)

print("Object columns (loop):", b)

object_df = df_original.select_dtypes(include=["object"])
print("Object columns (select_dtypes):", object_df.columns.tolist())

object_cols = df_original.select_dtypes(include=["object"]).columns.tolist()
print("Object columns list:", object_cols)

# =========================
# 2) COVID TOY DATASET
# =========================
print("\n--- COVID TOY DATASET ---")

df = pd.read_csv("data/covid_toy.csv")
print(df.head())

lb = LabelEncoder()

df["gender"] = lb.fit_transform(df["gender"])
df["fever"] = lb.fit_transform(df["fever"])
df["cough"] = lb.fit_transform(df["cough"])
df["city"] = lb.fit_transform(df["city"])
df["has_covid"] = lb.fit_transform(df["has_covid"])

print(df.head())

x = df.drop(columns=["has_covid"])
y = df["has_covid"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

covid_acc = accuracy_score(y_test, y_pred)
print("Accuracy (Covid):", covid_acc)