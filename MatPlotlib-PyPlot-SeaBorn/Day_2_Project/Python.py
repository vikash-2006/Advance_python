# =============================================================================
# EMPLOYEE ATTRITION ANALYSIS
# =============================================================================
# PURPOSE:
#   This script analyzes employee attrition data from a CSV file.
#   It filters employees who have left the company ("Yes" attrition),
#   groups them by department, and visualizes the distribution using
#   an interactive donut chart built with Plotly.
#
# REQUIREMENTS:
#   pip install numpy pandas plotly
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION 1: IMPORT LIBRARIES
# -----------------------------------------------------------------------------

import numpy as np          # NumPy: Used for numerical operations and array handling
import pandas as pd         # Pandas: Used for data manipulation and analysis (DataFrames)

import plotly.graph_objects as go   # Plotly Graph Objects: Used to build custom interactive charts
import plotly.io as pio             # Plotly IO: Used to control rendering settings and templates


# -----------------------------------------------------------------------------
# SECTION 2: CONFIGURE PLOT THEME
# -----------------------------------------------------------------------------

# Set the default visual theme for all Plotly charts to "plotly_white"
# This gives charts a clean white background, ideal for presentations/reports
pio.templates.default = "plotly_white"


# -----------------------------------------------------------------------------
# SECTION 3: LOAD THE DATASET
# -----------------------------------------------------------------------------

# pd.read_csv() reads the CSV file and loads it into a DataFrame
# A DataFrame is a 2D table with labeled rows and columns (like an Excel sheet)
# NOTE: Update the file path below if your CSV is stored in a different location
df = pd.read_csv("/Users/apple/data science/Advance_Python/MatPlotlib-PyPlot-SeaBorn/Day_2_Project/Attrition - Attrition.csv")

# Display the first few rows of the DataFrame to verify it loaded correctly
print("=== Dataset Preview ===")
print(df.head())

# Show the total number of rows and columns in the dataset
print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# -----------------------------------------------------------------------------
# SECTION 4: DATA QUALITY CHECK — NULL VALUES
# -----------------------------------------------------------------------------

# isnull().sum() counts the number of missing (NaN) values in each column
# This helps identify columns that may need cleaning before analysis
print("\n=== Missing Values Per Column ===")
print(df.isnull().sum())


# -----------------------------------------------------------------------------
# SECTION 5: FILTER ATTRITED EMPLOYEES
# -----------------------------------------------------------------------------

# Filter the DataFrame to keep only rows where Attrition == "Yes"
# These are employees who have LEFT the company
# The result is stored in a new DataFrame called attr_df
attr_df = df[df['Attrition'] == 'Yes']

print(f"\n=== Employees Who Left (Attrition = 'Yes') ===")
print(f"Total attrited employees: {len(attr_df)}")
print(attr_df.head())


# -----------------------------------------------------------------------------
# SECTION 6: GROUP ATTRITION BY DEPARTMENT
# -----------------------------------------------------------------------------

# groupby('Department') → Groups the filtered data by each unique department name
# .size()              → Counts the number of employees in each group
# .reset_index()       → Converts the grouped result back into a flat DataFrame
# name='Count'         → Names the count column "Count"
attrition_by_dept = attr_df.groupby(['Department']).size().reset_index(name='Count')

print("\n=== Attrition Count by Department ===")
print(attrition_by_dept)


# -----------------------------------------------------------------------------
# SECTION 7: CREATE INTERACTIVE DONUT CHART
# -----------------------------------------------------------------------------

# go.Figure() creates a new Plotly figure container
# go.Pie() adds a Pie chart trace inside the figure

fig = go.Figure(data=[go.Pie(
    labels=attrition_by_dept['Department'],   # Department names as slice labels
    values=attrition_by_dept['Count'],         # Attrition count determines slice size
    hole=0.4,                                  # hole=0.4 creates the donut effect (40% hollow center)
    marker=dict(colors=['red', 'green']),      # Assign custom colors to slices
    textposition='inside'                      # Display percentage/label text inside the slices
)])


# -----------------------------------------------------------------------------
# SECTION 8: CUSTOMIZE CHART LAYOUT
# -----------------------------------------------------------------------------

# update_layout() lets you modify the chart's title, fonts, margins, legend, etc.
fig.update_layout(
    title_text='Attrition By Department',      # Chart title displayed at the top
    title_x=0.5,                               # Center-align the title (0 = left, 1 = right)
    legend_title_text='Department'             # Add a label above the legend
)


# -----------------------------------------------------------------------------
# SECTION 9: DISPLAY THE CHART
# -----------------------------------------------------------------------------

# fig.show() renders the interactive Plotly chart in your default browser
# In VS Code, install the "Jupyter" extension or use a .ipynb notebook
# Alternatively, use fig.write_html("chart.html") to save and open in browser
fig.show()

# Optional: Save chart as an HTML file for sharing or offline viewing
# fig.write_html("attrition_by_department.html")