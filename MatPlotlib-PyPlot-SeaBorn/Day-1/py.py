# =============================================================================
#  EXPLORATORY DATA ANALYSIS (EDA) — TITANIC & TIPS DATASETS
#  File   : eda_analysis.py
#  Topics : Univariate Analysis, Bivariate Analysis, Multivariate Analysis,
#           Count Plot, Pie Chart, Histogram, Box Plot, Scatter Plot
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Install if not already installed:
# pip install matplotlib
# pip install seaborn


# =============================================================================
#  WHAT IS EDA (Exploratory Data Analysis)?
#  EDA is the process of analyzing datasets to summarize their main
#  characteristics, often using visual methods. It helps us understand
#  the data before applying any machine learning model.
#
#  PARTS OF EDA:
#  1. Univariate Analysis   → Analysis on a SINGLE column
#  2. Bivariate Analysis    → Analysis on TWO columns
#  3. Multivariate Analysis → Analysis on MORE THAN 2 columns
# =============================================================================


# -----------------------------------------------------------------------------
# LOAD DATASET
# pd.read_csv() reads a CSV file from disk and loads it into a DataFrame.
# A DataFrame is a 2D labeled table — like an Excel sheet.
# Update the path to where your titanic CSV file is located.
# -----------------------------------------------------------------------------
df = pd.read_csv("/Users/apple/data science/Advance_Python/MatPlotlib-PyPlot-SeaBorn/Day-1/titanic - titanic.csv")

print("=" * 60)
print("FULL DATAFRAME — TITANIC:")
print("=" * 60)
print(df)

print("\n")
print("=" * 60)
print("FIRST 5 ROWS — df.head():")
print("=" * 60)
print(df.head())
# df.head() shows the first 5 rows by default.
# Useful to quickly preview the dataset after loading.

print("\n")
print("=" * 60)
print("ALL COLUMN NAMES — df.columns:")
print("=" * 60)
print(df.columns)
# df.columns returns the list of all column names in the DataFrame.


# =============================================================================
#  SECTION 1 : UNIVARIATE ANALYSIS
#  Definition : Analysis performed on a SINGLE variable/column at a time.
#  Goal       : Understand the distribution and frequency of a single feature.
#  Used for   : Categorical data → Bar Chart, Count Plot, Pie Chart
#               Numerical data  → Histogram, Box Plot
# =============================================================================


# -----------------------------------------------------------------------------
# 1A. value_counts()
#     Returns the count/frequency of each unique value in a column.
#     Very useful for understanding categorical columns like 'Survived'.
#     0 = Did not survive | 1 = Survived
# -----------------------------------------------------------------------------
print("\n")
print("=" * 60)
print("VALUE COUNTS — df['Survived'].value_counts():")
print("=" * 60)
print(df['Survived'].value_counts())


# -----------------------------------------------------------------------------
# 1B. COUNT PLOT (sns.countplot)
#     A Count Plot displays the count of observations in each category.
#     It is the bar chart equivalent for categorical data.
#     Here we visualize how many passengers Survived (1) vs Did Not (0).
# -----------------------------------------------------------------------------
print("\n[PLOT] Count Plot — Survived Column")
sns.countplot(x=df['Survived'])
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


# -----------------------------------------------------------------------------
# 1C. BAR CHART using .plot(kind='bar')
#     Another way to visualize value counts using pandas built-in plotting.
#     Gives the same information as countplot but uses matplotlib style.
# -----------------------------------------------------------------------------
print("\n[PLOT] Bar Chart — Survived Column")
df['Survived'].value_counts().plot(kind='bar', color=['salmon', 'steelblue'])
plt.title("Survival Bar Chart")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()


# -----------------------------------------------------------------------------
# 1D. PIE CHART using .plot(kind='pie')
#     A Pie Chart is used when we want to show PERCENTAGE distribution.
#     autopct='%.2f' → displays percentage with 2 decimal places.
#     Best used for categorical data to see proportional share.
# -----------------------------------------------------------------------------
print("\n[PLOT] Pie Chart — Survived Percentage")
df['Survived'].value_counts().plot(
    kind='pie',
    autopct='%.2f%%',
    labels=['Not Survived', 'Survived'],
    colors=['salmon', 'steelblue']
)
plt.title("Survival Percentage")
plt.ylabel("")   # hide default y-label
plt.show()


# -----------------------------------------------------------------------------
# 1E. HISTOGRAM — plt.hist()
#     A Histogram is used for NUMERICAL (continuous) data.
#     It shows the DISTRIBUTION of data — how values are spread.
#     X-axis = Value ranges (bins), Y-axis = Frequency
#     Example: Age distribution of Titanic passengers.
# -----------------------------------------------------------------------------
print("\n[PLOT] Histogram — Age Distribution")
plt.hist(x=df['Age'], color='steelblue', edgecolor='black')
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# -----------------------------------------------------------------------------
# 1F. BOX PLOT — sns.boxplot()
#     A Box Plot is used to detect OUTLIERS and understand data spread.
#     It shows 5 key statistics:
#
#       ┌─────────────────────────────────-┐
#       │  1. Lower Fence  = Q1 - 1.5*IQR  │  (values below = outliers)
#       │  2. Q1 (25%)     = 25th percentile
#       │  3. Median (50%) = Middle value  │
#       │  4. Q3 (75%)     = 75th percentile
#       │  5. Upper Fence  = Q3 + 1.5*IQR  │  (values above = outliers)
#       └─────────────────────────────────-┘
#
#     IQR (Inter Quartile Range) = Q3 - Q2 (75% - 25%)
#
#     WHY BOXPLOT MATTERS?
#     Mean is sensitive to outliers. Example:
#       x = [1,2,3,4,5]      → mean = 3    (correct)
#       x = [1,2,3,4,5,100]  → mean = 19.3 (misleading!)
#     Boxplot helps us spot that 100 is an OUTLIER.
# -----------------------------------------------------------------------------
print("\n[PLOT] Box Plot — Age Column (Outlier Detection)")
sns.boxplot(x=df['Age'], color='lightblue')
plt.title("Age Box Plot — Outlier Detection")
plt.xlabel("Age")
plt.show()


# =============================================================================
#  SECTION 2 : BIVARIATE ANALYSIS — TIPS DATASET
#  Definition : Analysis performed on TWO variables/columns at a time.
#  Goal       : Find the RELATIONSHIP or CORRELATION between two features.
#  Used for   : Scatter Plot, Bar Plot, Box Plot (grouped), etc.
# =============================================================================


# -----------------------------------------------------------------------------
# LOAD TIPS DATASET
# sns.load_dataset('tips') is a built-in seaborn dataset about restaurant tips.
# Columns: total_bill, tip, sex, smoker, day, time, size
# -----------------------------------------------------------------------------
tips = sns.load_dataset('tips')

print("\n")
print("=" * 60)
print("TIPS DATASET — First 5 rows:")
print("=" * 60)
print(tips.head())


# -----------------------------------------------------------------------------
# 2A. SCATTER PLOT — sns.scatterplot()
#     A Scatter Plot shows the relationship between TWO numerical variables.
#     Each point represents one observation.
#     Here: Does a higher total_bill lead to a higher tip?
# -----------------------------------------------------------------------------
print("\n[PLOT] Scatter Plot — total_bill vs tip")
sns.scatterplot(x=tips['total_bill'], y=tips['tip'])
plt.title("Total Bill vs Tip")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.show()


# -----------------------------------------------------------------------------
# 2B. SCATTER PLOT WITH HUE (Color by Category)
#     Adding 'hue' parameter colors the points by a categorical variable.
#     This turns a bivariate plot into a simple multivariate visualization.
#     Here: Same scatter plot but colored by 'sex' (Male / Female).
#     This helps us compare patterns across groups.
# -----------------------------------------------------------------------------
print("\n[PLOT] Scatter Plot — total_bill vs tip (colored by sex)")
sns.scatterplot(
    x="total_bill",
    y="tip",
    data=tips,
    hue=tips['sex'],
    palette=['steelblue', 'salmon']
)
plt.title("Total Bill vs Tip (by Gender)")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.legend(title="Gender")
plt.show()


# =============================================================================
#  QUICK SUMMARY — WHEN TO USE WHICH CHART
#  ┌──────────────────┬─────────────────────────────────────────────────┐
#  │  Chart Type      │  When to Use                                    │
#  ├──────────────────┼─────────────────────────────────────────────────┤
#  │  Count Plot      │  Categorical column — frequency of categories   │
#  │  Bar Chart       │  Categorical column — compare category values   │
#  │  Pie Chart       │  Categorical column — percentage distribution   │
#  │  Histogram       │  Numerical column  — distribution/spread        │
#  │  Box Plot        │  Numerical column  — outlier detection          │
#  │  Scatter Plot    │  Two numerical columns — relationship/trend     │
#  └──────────────────┴─────────────────────────────────────────────────┘
# =============================================================================