import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LassoCV, ElasticNetCV

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("data/insurance.csv")

print("First 2 rows:")
print(df.head(2))

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================
lb = LabelEncoder()

df["sex"] = lb.fit_transform(df["sex"])
df["smoker"] = lb.fit_transform(df["smoker"])
df["region"] = lb.fit_transform(df["region"])

print("\nAfter encoding:")
print(df.head(2))

# =========================
# DEFINE FEATURES AND TARGET
# =========================
x = df.drop(columns=["charges"])
y = df["charges"]

# =========================
# TRAIN-TEST SPLIT
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# =========================
# RIDGE REGRESSION (L2)
# =========================
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)

ridge_pred = ridge_model.predict(x_test)

ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print("\n===== RIDGE REGRESSION =====")
print("MSE:", ridge_mse)
print("R2 Score:", ridge_r2)
print("Coefficients:", ridge_model.coef_)

# =========================
# LASSO REGRESSION (L1)
# =========================
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train)

lasso_pred = lasso_model.predict(x_test)

lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print("\n===== LASSO REGRESSION =====")
print("MSE:", lasso_mse)
print("R2 Score:", lasso_r2)
print("Coefficients:", lasso_model.coef_)

# =========================
# ELASTIC NET REGRESSION (L1 + L2)
# =========================
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_model.fit(x_train, y_train)

elastic_pred = elastic_model.predict(x_test)

elastic_mse = mean_squared_error(y_test, elastic_pred)
elastic_r2 = r2_score(y_test, elastic_pred)

print("\n===== ELASTIC NET REGRESSION =====")
print("MSE:", elastic_mse)
print("R2 Score:", elastic_r2)
print("Coefficients:", elastic_model.coef_)

# =========================
# CROSS VALIDATION - LASSO
# =========================
lasso_cv = LassoCV(
    alphas=[0.001, 0.01, 0.1, 1, 10],
    cv=5,
    random_state=42
)

lasso_cv.fit(x_train, y_train)

print("\n===== LASSO CV =====")
print("Best Alpha:", lasso_cv.alpha_)

# =========================
# CROSS VALIDATION - ELASTIC NET
# =========================
elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1],
    alphas=[0.001, 0.01, 0.1, 1, 10],
    cv=5,
    random_state=42
)

elastic_cv.fit(x_train, y_train)

print("\n===== ELASTIC NET CV =====")
print("Best Alpha:", elastic_cv.alpha_)
print("Best l1_ratio:", elastic_cv.l1_ratio_)