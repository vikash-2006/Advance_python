# =============================================================================
#  PREPROCESSING PRACTICE — VS Code Ready (With MySQL)
#  Author  : Vikash Kumawat
#  Topics  : Label Encoding | StandardScaler | MinMaxScaler | MySQL
#  Datasets: Bank Churn | COVID
# =============================================================================
#
#  ⚙️  MYSQL SETUP (run once in Terminal before running this file):
#      brew install mysql
#      brew services start mysql
#      mysql -u root -e "ALTER USER 'root'@'localhost' IDENTIFIED BY 'root123';"
#
#  📦  INSTALL PACKAGES (run once in Terminal):
#      pip install mysql-connector-python sqlalchemy pandas scikit-learn
#
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

import mysql.connector
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE CONFIG  ← change password if different
# ─────────────────────────────────────────────────────────────────────────────

DB_HOST     = "127.0.0.1"
DB_USER     = "root"
DB_PASSWORD = "root123"
DB_NAME     = "mydb"

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Reusable MySQL Connection
# ─────────────────────────────────────────────────────────────────────────────

def get_connection():
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    return conn

def get_engine():
    url = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    return create_engine(url)

def setup_database():
    """Create database if it doesn't exist."""
    conn = mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};")
    conn.close()
    print("✅ MySQL Running!")
    print("✅ Connected!")
    print(f"✅ Database '{DB_NAME}' Ready!")

# Setup DB once
setup_database()


# =============================================================================
#  SECTION 1 — Bank Churn Dataset
# =============================================================================

# ── Update this path to your CSV location ────────────────────────────────────
CHURN_PATH = Path("/Users/apple/data science/Advance_Python/Machine_Learning/Day-8/Churn_Modelling - Churn_Modelling (1).csv")

print("\n" + "=" * 60)
print(" SECTION 1 — Bank Churn Dataset")
print("=" * 60)

# ── Load CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CHURN_PATH)

print("\n─── Dataset Info ───")
print("Shape       :", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nColumn Names:\n", df.columns.tolist())
print("\nData Types:\n", df.dtypes)

# ── Insert into MySQL ─────────────────────────────────────────────────────────
print("\n─── Inserting into MySQL ───")
engine = get_engine()
df.to_sql(
    name="bank_churn",       # table name in MySQL
    con=engine,
    if_exists="replace",     # 'replace' drops & recreates | 'append' adds rows
    index=False
)
print(f"✅ {len(df)} rows inserted into table 'bank_churn'!")

# ── Read Back from MySQL ──────────────────────────────────────────────────────
print("\n─── Reading from MySQL ───")
conn = get_connection()
df = pd.read_sql("SELECT * FROM bank_churn LIMIT 10;", conn)
conn.close()
print("First 10 rows from MySQL:\n", df)

# ── Exploration ───────────────────────────────────────────────────────────────
print("\n─── Exploration ───")
print("\nSurname Value Counts:\n", df['Surname'].value_counts())
print("\nMissing Values:\n", df.isnull().sum())

# ── Label Encoding (Categorical → Numeric) ────────────────────────────────────
# LabelEncoder converts text categories to numbers e.g. Male=1, Female=0
print("\n─── Label Encoding ───")
lb = LabelEncoder()

df['Surname']   = lb.fit_transform(df['Surname'])
df['Geography'] = lb.fit_transform(df['Geography'])
df['Gender']    = lb.fit_transform(df['Gender'])

print("After Encoding:\n", df.head())

# ── Feature / Target Split ────────────────────────────────────────────────────
print("\n─── Feature / Target Split ───")
X = df.drop(columns=['Exited'])   # features
y = df['Exited']                  # target
print("Features shape :", X.shape)
print("Target shape   :", y.shape)

# ── Train-Test Split ──────────────────────────────────────────────────────────
print("\n─── Train-Test Split (80/20) ───")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("\nX_train Stats (before scaling):")
print(np.round(X_train.describe(), 2))

# ── StandardScaler ────────────────────────────────────────────────────────────
# Formula: z = (x - mean) / std  →  output mean≈0, std≈1
print("\n─── StandardScaler (Z-score Normalisation) ───")
sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)   # fit only on train
X_test_sc  = sc.transform(X_test)        # transform test with same scaler

X_train_scaled = pd.DataFrame(X_train_sc, columns=X_train.columns)
print("X_train Stats (after StandardScaler):")
print(np.round(X_train_scaled.describe(), 2))

# ── MinMaxScaler ──────────────────────────────────────────────────────────────
# Formula: x_scaled = (x - min) / (max - min)  →  output range [0, 1]
print("\n─── MinMaxScaler (Normalisation) ───")
mn = MinMaxScaler()

X_train_mn = mn.fit_transform(X_train)   # fit only on train
X_test_mn  = mn.transform(X_test)        # transform test with same scaler

X_train_norm = pd.DataFrame(X_train_mn, columns=X_train.columns)
print("X_train Stats (after MinMaxScaler):")
print(np.round(X_train_norm.describe(), 2))


# =============================================================================
#  SECTION 2 — COVID Dataset
# =============================================================================

# ── Update this path to your CSV location ────────────────────────────────────
COVID_PATH = Path("/Users/apple/data science/Advance_Python/Machine_Learning/Day-2/covid_toy - covid_toy.csv")

print("\n" + "=" * 60)
print(" SECTION 2 — COVID Dataset")
print("=" * 60)

# ── Load CSV ──────────────────────────────────────────────────────────────────
df2 = pd.read_csv(COVID_PATH)

print("\n─── Dataset Info ───")
print("Shape       :", df2.shape)
print("\nFirst 5 Rows:\n", df2.head())
print("\nColumn Names:\n", df2.columns.tolist())
print("\nData Types:\n", df2.dtypes)

# ── Insert into MySQL ─────────────────────────────────────────────────────────
print("\n─── Inserting into MySQL ───")
engine = get_engine()
df2.to_sql(
    name="covid",            # table name in MySQL
    con=engine,
    if_exists="replace",
    index=False
)
print(f"✅ {len(df2)} rows inserted into table 'covid'!")

# ── Read Back from MySQL ──────────────────────────────────────────────────────
print("\n─── Reading from MySQL ───")
conn = get_connection()
df2 = pd.read_sql("SELECT * FROM covid LIMIT 10;", conn)
conn.close()
print("First 10 rows from MySQL:\n", df2)

# ── Missing Values ────────────────────────────────────────────────────────────
print("\n─── Missing Values ───")
print(df2.isnull().sum())
df2 = df2.dropna()
print("After dropna shape:", df2.shape)

# ── Label Encoding ────────────────────────────────────────────────────────────
print("\n─── Label Encoding ───")
lb2 = LabelEncoder()

df2['gender']    = lb2.fit_transform(df2['gender'])
df2['cough']     = lb2.fit_transform(df2['cough'])
df2['city']      = lb2.fit_transform(df2['city'])
df2['has_covid'] = lb2.fit_transform(df2['has_covid'])

print("After Encoding:\n", df2.head())

# ── Feature / Target Split ────────────────────────────────────────────────────
print("\n─── Feature / Target Split ───")
X2 = df2.drop(columns=['has_covid'])
y2 = df2['has_covid']
print("Features shape :", X2.shape)
print("Target shape   :", y2.shape)

# ── Train-Test Split ──────────────────────────────────────────────────────────
print("\n─── Train-Test Split (80/20) ───")
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)
print("X_train :", X2_train.shape)
print("X_test  :", X2_test.shape)
print("\nX_train Stats (before scaling):")
print(np.round(X2_train.describe(), 2))

# ── StandardScaler ────────────────────────────────────────────────────────────
print("\n─── StandardScaler (Z-score Normalisation) ───")
sc2 = StandardScaler()

X2_train_sc = sc2.fit_transform(X2_train)
X2_test_sc  = sc2.transform(X2_test)

X2_train_scaled = pd.DataFrame(X2_train_sc, columns=X2_train.columns)
print("X_train Stats (after StandardScaler):")
print(np.round(X2_train_scaled.describe(), 2))

# ── MinMaxScaler ──────────────────────────────────────────────────────────────
print("\n─── MinMaxScaler (Normalisation) ───")
mn2 = MinMaxScaler()

X2_train_mn = mn2.fit_transform(X2_train)
X2_test_mn  = mn2.transform(X2_test)

X2_train_norm = pd.DataFrame(X2_train_mn, columns=X2_train.columns)
print("X_train Stats (after MinMaxScaler):")
print(np.round(X2_train_norm.describe(), 2))

# =============================================================================
#  END OF FILE
# =============================================================================