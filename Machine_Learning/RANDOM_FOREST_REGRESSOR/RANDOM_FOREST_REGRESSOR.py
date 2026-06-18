import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("data/insurance.csv")
print(df.head())

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================
lb = LabelEncoder()

df["sex"] = lb.fit_transform(df["sex"])
df["smoker"] = lb.fit_transform(df["smoker"])
df["region"] = lb.fit_transform(df["region"])

print("\nEncoded data:")
print(df.head())

# =========================
# DEFINE FEATURES AND TARGET
# =========================
x = df.drop(columns=["charges"])
y = df["charges"]

# =========================
# TRAIN-TEST SPLIT
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

# =========================
# RANDOM FOREST REGRESSOR
# =========================
rfr = RandomForestRegressor(random_state=42)
rfr.fit(x_train, y_train)

y_pred = rfr.predict(x_test)

# =========================
# EVALUATION
# =========================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MSE:", mse)
print("R2 Score:", r2)