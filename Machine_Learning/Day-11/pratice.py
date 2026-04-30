# =============================================================================
#  NUMPY & PANDAS PRACTICE — VS Code Ready
#  Author  : Vikash Kumawat
#  Topics  : NumPy Basics | Pandas (Attrition) | Preprocessing (Insurance)
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — NumPy Basics
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

# ── Q1. Difference Between NumPy Array and Python List ───────────────────────
# List  → stores multiple data types | slower operations
# Array → stores same data type      | much faster (vectorized operations)

# ── Q2. Create an Array and Find Shape, Size, and Dimension ──────────────────

array = np.array([[1, 2, 3],
                  [4, 5, 6]])

print("─── Q2: Array Info ───")
print("Array:\n", array)
print("Shape     :", array.shape)   # (rows, columns)
print("Size      :", array.size)    # total number of elements
print("Dimension :", array.ndim)    # number of dimensions

# ── Q3. zeros, ones, eye, diag, linspace ─────────────────────────────────────

print("\n─── Q3: Array Creation Functions ───")

# zeros() — fill matrix with 0s
print("\nzeros():")
print(np.zeros((2, 3)))

# ones() — fill matrix with 1s
print("\nones():")
print(np.ones((2, 3)))

# eye() — identity matrix (1s on diagonal, 0s elsewhere)
print("\neye():")
print(np.eye(3))

# diag() — custom diagonal matrix
print("\ndiag():")
print(np.diag([1, 2, 3]))

# linspace() — evenly spaced values between start and end
print("\nlinspace(0, 10, 5):")
print(np.linspace(0, 10, 5))

# ── Q4. Horizontal and Vertical Stacking ─────────────────────────────────────

print("\n─── Q4: Stacking ───")

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# hstack — join arrays column-wise (left → right)
print("\nhstack (horizontal):")
print(np.hstack((a, b)))

# vstack — join arrays row-wise (top → bottom)
print("\nvstack (vertical):")
print(np.vstack((a, b)))

# ── Q5. View vs Copy ──────────────────────────────────────────────────────────

print("\n─── Q5: View vs Copy ───")

# VIEW — changes affect the original array
a = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print("\nView Example:")
print("Original before :", a)
a[3:6] = 0                        # directly modifies original
print("Original after  :", a)

# COPY — changes do NOT affect the original array
a = np.array([10, 20, 30, 40, 50, 60, 70, 80])
b = a[3:6].copy()
b[:] = 0
print("\nCopy Example:")
print("Original :", a)            # unchanged
print("Copy     :", b)            # only copy is modified

# ── Q6. Matrix Multiplication (2×2) ──────────────────────────────────────────

print("\n─── Q6: Matrix Multiplication ───")

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

c = np.dot(a, b)
print("A:\n", a)
print("B:\n", b)
print("A · B:\n", c)

# ── Q7. Reshape + Sum (total, row-wise, column-wise) ─────────────────────────

print("\n─── Q7: Reshape and Summations ───")

a = np.arange(1, 10).reshape(3, 3)
print("Array (3×3):\n", a)

print("\nTotal Sum          :", np.sum(a))
print("Row Sum   (axis=1) :", np.sum(a, axis=1))
print("Column Sum(axis=0) :", np.sum(a, axis=0))

# ── Q8. Random: randint, rand, seed ──────────────────────────────────────────

print("\n─── Q8: Random Functions ───")

# randint() — random integers in a range
print("\nrandint(1, 10, 5):", np.random.randint(1, 10, 5))

# rand() — random decimals between 0 and 1
print("rand(3)           :", np.random.rand(3))

# seed() — fix random output so it's the same every run
np.random.seed(4)
print("seed(4) + rand(3) :", np.random.rand(3))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Pandas: Employee Attrition Dataset
# ─────────────────────────────────────────────────────────────────────────────
# Dataset : Attrition.csv
# Update the path below to match your local file location.

import pandas as pd
import numpy as np

ATTRITION_PATH = "/Users/apple/data science/Advance_Python/MatPlotlib-PyPlot-SeaBorn/Day_2_Project/Attrition - Attrition.csv"
print("\n" + "=" * 60)
print(" SECTION 2 — Pandas: Employee Attrition Dataset")
print("=" * 60)

df = pd.read_csv(ATTRITION_PATH)

# ── Q1. First 10 Rows | Shape | Column Names ──────────────────────────────────
print("\n─── Q1: Dataset Overview ───")
print(df.head(10))
print("\nShape (rows, columns) :", df.shape)
print("\nColumn Names:\n", df.columns.tolist())

# ── Q2. Unique Values in Department ──────────────────────────────────────────
print("\n─── Q2: Unique Department Values ───")
print(df['Department'].value_counts())

# ── Q3. Employees Who Left (Attrition = Yes) ──────────────────────────────────
print("\n─── Q3: Employees Who Left ───")
count = (df['Attrition'] == 'Yes').sum()
print("Count:", count)

# ── Q4. Filter: Age > 40 ──────────────────────────────────────────────────────
print("\n─── Q4: Employees Older Than 40 ───")
print(df[df['Age'] > 40])

# ── Q5. Select Specific Columns ───────────────────────────────────────────────
print("\n─── Q5: Age | Department | MonthlyIncome ───")
print(df[['Age', 'Department', 'MonthlyIncome']])

# ── Q6. Employees with OverTime = Yes ─────────────────────────────────────────
print("\n─── Q6: Employees with OverTime ───")
print(df[df['OverTime'] == 'Yes'])

# ── Q7. Add Column: Income After Hike ─────────────────────────────────────────
# Formula: MonthlyIncome + (MonthlyIncome × PercentSalaryHike / 100)
print("\n─── Q7: Income After Hike ───")
df['income_after_hike'] = (
    df['MonthlyIncome'] + (df['MonthlyIncome'] * df['PercentSalaryHike'] / 100)
)
print(df[['MonthlyIncome', 'PercentSalaryHike', 'income_after_hike']].head())

# ── Q8. Rename Column: MonthlyIncome → salary ─────────────────────────────────
print("\n─── Q8: Rename Column ───")
df.rename(columns={'MonthlyIncome': 'salary'}, inplace=True)
print("Updated Columns:\n", df.columns.tolist())

# ── Q9. Average Salary per Department ─────────────────────────────────────────
print("\n─── Q9: Avg Salary per Department ───")
avg_income = df.groupby('Department')['salary'].mean()
print(np.round(avg_income, 2))

# ── Q10. Average Age per Job Role ─────────────────────────────────────────────
print("\n─── Q10: Avg Age per Job Role ───")
avg_age = df.groupby('JobRole')['Age'].mean()
print(np.round(avg_age, 2))

# ── Q11. Minimum Distance from Home per Department ────────────────────────────
print("\n─── Q11: Min Distance from Home per Department ───")
min_dist = df.groupby('Department')['DistanceFromHome'].min()
print(min_dist)

# ── Q12. Total Employees per Education Field ──────────────────────────────────
print("\n─── Q12: Employees per Education Field ───")
print(df['EducationField'].value_counts())

# ── Q13. Employee Count by Marital Status & Attrition ────────────────────────
print("\n─── Q13: Count by Marital Status & Attrition ───")
result = df.groupby(['MaritalStatus', 'Attrition'])['EmployeeNumber'].count()
print(result)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Insurance Dataset: Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Dataset : insurance.csv  (columns: age, sex, bmi, children, smoker, region, charges)
# Update the path below to match your local file location.

          
INSURANCE_PATH = "/Users/apple/data science/Advance_Python/Machine_Learning/Day-3/insurance - insurance.csv"    # ← change if needed
print("\n" + "=" * 60)
print(" SECTION 3 — Insurance Dataset: Preprocessing")
print("=" * 60)

# ── Load Dataset ──────────────────────────────────────────────────────────────
df_ins = pd.read_csv(INSURANCE_PATH)

print("\n─── Dataset Info ───")
print("Shape       :", df_ins.shape)
print("\nFirst 5 Rows:\n", df_ins.head())
print("\nColumn Names:\n", df_ins.columns.tolist())
print("\nData Types:\n", df_ins.dtypes)

# ── Check Missing Values ──────────────────────────────────────────────────────
print("\n─── Missing Values ───")
print(df_ins.isnull().sum())

# ── Label Encoding (Categorical → Numeric) ────────────────────────────────────
print("\n─── Label Encoding ───")
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df_ins['sex']    = lb.fit_transform(df_ins['sex'])      # female=0, male=1
df_ins['smoker'] = lb.fit_transform(df_ins['smoker'])   # no=0, yes=1
df_ins['region'] = lb.fit_transform(df_ins['region'])   # numeric labels
print("After encoding:\n", df_ins.head())

# ── Feature / Target Split ────────────────────────────────────────────────────
print("\n─── Feature / Target Split ───")
X = df_ins.drop(columns=['charges'])
y = df_ins['charges']
print("Features shape :", X.shape)
print("Target shape   :", y.shape)

# ── Train-Test Split ──────────────────────────────────────────────────────────
print("\n─── Train-Test Split (80/20) ───")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)

print("\nX_train Stats (before scaling):")
print(np.round(X_train.describe(), 2))

# ── StandardScaler ────────────────────────────────────────────────────────────
print("\n─── StandardScaler (Z-score Normalization) ───")
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)     # fit on train only
X_test_sc  = sc.transform(X_test)          # transform test with same scaler

X_train_scaled = pd.DataFrame(X_train_sc, columns=X_train.columns)

print("\nX_train Stats (after scaling):")
print(np.round(X_train_scaled.describe(), 2))

# =============================================================================
#  END OF FILE
# =============================================================================