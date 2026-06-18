# Decision Tree Model -> CART (Classification And Regression Tree)

# For numerical target -> use DecisionTreeRegressor
# For categorical target -> use DecisionTreeClassifier

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

# =========================
# 1) INSURANCE DATASET (REGRESSION)
# =========================
print("\n--- INSURANCE DATASET (REGRESSION) ---")

df = pd.read_csv("data/insurance.csv")
print(df.head(2))

lb = LabelEncoder()
df["sex"] = lb.fit_transform(df["sex"])
df["smoker"] = lb.fit_transform(df["smoker"])
df["region"] = lb.fit_transform(df["region"])

x = df.drop(columns=["charges"])
y = df["charges"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

dt = DecisionTreeRegressor(random_state=42)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

score = r2_score(y_test, y_pred)
print("R2 Score:", score)

# =========================
# 2) SOCIAL NETWORK ADS (CLASSIFICATION)
# =========================
print("\n--- SOCIAL NETWORK ADS (CLASSIFICATION) ---")

df = pd.read_csv("data/Social_Network_Ads.csv")
print(df.head(2))

lb = LabelEncoder()
df["Gender"] = lb.fit_transform(df["Gender"])

x = df.drop(columns=["Purchased"])
y = df["Purchased"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", score)

# =========================
# 3) COVID TOY DATASET (CLASSIFICATION)
# =========================
print("\n--- COVID TOY DATASET (CLASSIFICATION) ---")

df = pd.read_csv("data/covid_toy.csv")
print(df.head(2))

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

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", score)

# =========================
# 4) CHURN MODELLING DATASET (CLASSIFICATION)
# =========================
print("\n--- CHURN MODELLING DATASET (CLASSIFICATION) ---")

df = pd.read_csv("data/Churn_Modelling.csv")
print(df.head(2))

lb = LabelEncoder()
df["Surname"] = lb.fit_transform(df["Surname"])
df["Geography"] = lb.fit_transform(df["Geography"])
df["Gender"] = lb.fit_transform(df["Gender"])

x = df.drop(columns=["Exited"])
y = df["Exited"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", score)# Decision Tree Model -> CART (Classification And Regression Tree)

# For numerical target -> use DecisionTreeRegressor
# For categorical target -> use DecisionTreeClassifier

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

# =========================
# 1) INSURANCE DATASET (REGRESSION)
# =========================
print("\n--- INSURANCE DATASET (REGRESSION) ---")

df = pd.read_csv("data/insurance.csv")
print(df.head(2))

lb = LabelEncoder()
df["sex"] = lb.fit_transform(df["sex"])
df["smoker"] = lb.fit_transform(df["smoker"])
df["region"] = lb.fit_transform(df["region"])

x = df.drop(columns=["charges"])
y = df["charges"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

dt = DecisionTreeRegressor(random_state=42)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

score = r2_score(y_test, y_pred)
print("R2 Score:", score)

# =========================
# 2) SOCIAL NETWORK ADS (CLASSIFICATION)
# =========================
print("\n--- SOCIAL NETWORK ADS (CLASSIFICATION) ---")

df = pd.read_csv("data/Social_Network_Ads.csv")
print(df.head(2))

lb = LabelEncoder()
df["Gender"] = lb.fit_transform(df["Gender"])

x = df.drop(columns=["Purchased"])
y = df["Purchased"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", score)

# =========================
# 3) COVID TOY DATASET (CLASSIFICATION)
# =========================
print("\n--- COVID TOY DATASET (CLASSIFICATION) ---")

df = pd.read_csv("data/covid_toy.csv")
print(df.head(2))

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

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", score)

# =========================
# 4) CHURN MODELLING DATASET (CLASSIFICATION)
# =========================
print("\n--- CHURN MODELLING DATASET (CLASSIFICATION) ---")

df = pd.read_csv("data/Churn_Modelling.csv")
print(df.head(2))

lb = LabelEncoder()
df["Surname"] = lb.fit_transform(df["Surname"])
df["Geography"] = lb.fit_transform(df["Geography"])
df["Gender"] = lb.fit_transform(df["Gender"])

x = df.drop(columns=["Exited"])
y = df["Exited"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", score)