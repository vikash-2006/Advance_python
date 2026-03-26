# =============================================================================
# SUPPLY CHAIN DATA ANALYSIS
# Author   : Vikash Kumawat
# Date     : 26-03-2026
# Tools    : Python | Pandas | NumPy | Plotly | VS Code
# Dataset  : supply_chain.csv
# =============================================================================
#
# WHAT IS SUPPLY CHAIN ANALYSIS?
# Supply chain analysis is the process of examining the flow of goods,
# data, and finances from raw material to final customer delivery.
# It helps businesses identify inefficiencies, reduce costs, and
# improve overall performance across production, logistics, and sales.
#
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION 1 — IMPORT LIBRARIES
# -----------------------------------------------------------------------------
#
# WHAT ARE LIBRARIES?
# Libraries are pre-written collections of code that give us ready-made
# tools and functions so we don't have to write everything from scratch.
#
# numpy (np)        → Numerical Python. Used for mathematical operations
#                     on arrays and matrices. Example: np.mean(), np.sum()
#
# pandas (pd)       → Used for data manipulation and analysis.
#                     It loads data into a DataFrame (like an Excel table)
#                     so we can filter, group, and analyze it easily.
#
# plotly.express    → A high-level plotting library for creating
#                     interactive charts like scatter plots and pie charts
#                     with very few lines of code.
#
# plotly.io (pio)   → Controls Plotly input/output settings.
#                     Here we use it to set the default visual theme.
#
# plotly.graph_objects (go) → A lower-level Plotly library that gives
#                             more control over chart customization.
#                             Used here to build the bar chart manually.
#
# pio.templates.default = "plotly_white"
#                   → Sets the background theme of all charts to white,
#                     which gives a clean, professional look.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "plotly_white"


# -----------------------------------------------------------------------------
# SECTION 2 — LOAD DATASET
# -----------------------------------------------------------------------------
#
# WHAT IS A DATASET?
# A dataset is a structured collection of data, usually stored as a
# CSV (Comma-Separated Values) file. Each row is one record (product/SKU)
# and each column is a feature (Price, Revenue, Defect Rate, etc.)
#
# pd.read_csv()     → Reads the CSV file from your computer and converts
#                     it into a Pandas DataFrame stored in variable 'df'.
#
# df.head()         → Displays the first 5 rows of the dataset so we can
#                     quickly verify the data loaded correctly.
#
# The dataset has 24 columns including:
#   - Product type, SKU, Price, Availability
#   - Number of products sold, Revenue generated
#   - Stock levels, Lead times, Order quantities
#   - Location, Shipping carriers, Routes
#   - Manufacturing costs, Defect rates, Inspection results
# -----------------------------------------------------------------------------

df = pd.read_csv("/Users/apple/Downloads/supply_chain - supply_chain.csv")
print(df.head())


# -----------------------------------------------------------------------------
# SECTION 3 — PRICE vs REVENUE ANALYSIS
# -----------------------------------------------------------------------------
#
# OBJECTIVE:
# Understand the relationship between the price of a product and the
# revenue it generates, broken down by product type.
#
# WHAT IS A SCATTER PLOT?
# A scatter plot displays individual data points on an X-Y axis.
# Each dot represents one product (SKU). By looking at the spread
# and direction of the dots, we can identify patterns or correlations.
#
# px.scatter()      → Creates an interactive scatter plot.
#
# PARAMETERS EXPLAINED:
#   df              → The DataFrame (our dataset) to use.
#   x = 'Price'     → X-axis shows the price of the product.
#   y = 'Revenue generated' → Y-axis shows the revenue earned.
#   color = 'Product type'  → Each product type gets a different color,
#                              making it easy to compare categories.
#   hover_data      → Shows extra info (products sold) when you hover
#                     over a dot in the interactive chart.
#   trendline='ols' → Draws an OLS (Ordinary Least Squares) regression
#                     line — a straight line that best fits the data.
#                     It shows the overall trend: upward = positive
#                     correlation, downward = negative correlation.
#                     NOTE: Requires 'statsmodels' library.
#                     Install with: pip install statsmodels
#
# INSIGHT:
# Skincare products show a positive trend — as price increases,
# revenue also increases. Haircare shows a slight downward trend,
# suggesting price may not be a strong driver of revenue for that type.
# -----------------------------------------------------------------------------

fig = px.scatter(
    df,
    x='Price',
    y='Revenue generated',
    color='Product type',
    hover_data=['Number of products sold'],
    trendline='ols',
    title='Price vs Revenue Generated by Product Type'
)
fig.show()


# -----------------------------------------------------------------------------
# SECTION 4 — SALES DISTRIBUTION BY PRODUCT TYPE
# -----------------------------------------------------------------------------
#
# OBJECTIVE:
# Find out which product type contributes the most units sold,
# and visualize the proportion using a pie chart.
#
# STEP 1 — GROUP THE DATA:
# df.groupby("Product type")
#                     → Groups all rows by their Product type
#                       (skincare, haircare, cosmetics).
# ["Number of products sold"].sum()
#                     → For each group, adds up (sums) all products sold.
# .reset_index()      → Converts the result back into a clean DataFrame.
#
# WHAT IS A PIE CHART?
# A pie chart shows proportions of a whole. Each slice represents
# one category, and the size of the slice = its percentage share.
#
# px.pie()            → Creates an interactive donut/pie chart.
#
# PARAMETERS EXPLAINED:
#   values            → The column whose numbers determine slice sizes.
#   names             → The column used to label each slice.
#   hole = 0.5        → Creates a hole in the center (donut chart style).
#                       Value between 0 (full pie) and 1 (no pie).
#   color_discrete_sequence → Sets the color palette. 'Pastel' gives
#                              soft, visually appealing colors.
#   textposition='inside'   → Places percentage labels inside each slice.
#   textinfo='percent+label'→ Shows both the % and the category name.
#
# INSIGHT:
# Skincare leads with 45% of total sales, followed by Haircare (29.5%)
# and Cosmetics (25.5%). Skincare is the company's core product line.
# -----------------------------------------------------------------------------

sales_data = df.groupby("Product type")["Number of products sold"].sum().reset_index()
print(sales_data)

pie_chart = px.pie(
    sales_data,
    values="Number of products sold",
    names="Product type",
    title="Sales by Product Type",
    hover_data=["Number of products sold"],
    hole=0.5,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
pie_chart.update_traces(textposition='inside', textinfo='percent+label')
pie_chart.show()


# -----------------------------------------------------------------------------
# SECTION 5 — TOTAL REVENUE BY SHIPPING CARRIER
# -----------------------------------------------------------------------------
#
# OBJECTIVE:
# Compare the total revenue generated through each shipping carrier
# (Carrier A, Carrier B, Carrier C) to identify the most profitable one.
#
# WHAT IS A BAR CHART?
# A bar chart uses rectangular bars to compare values across categories.
# Taller bar = higher value. Ideal for comparing groups side by side.
#
# go.Figure()         → Creates a blank Plotly figure (canvas).
# fig.add_trace()     → Adds a chart layer (trace) to the figure.
# go.Bar()            → Defines a bar chart trace.
#
# PARAMETERS EXPLAINED:
#   x = 'Shipping carriers' → X-axis: the 3 carriers (A, B, C)
#   y = 'Revenue generated' → Y-axis: total revenue for each carrier
#
# fig.update_layout() → Customizes the chart title and axis labels
#                       for better readability.
#
# INSIGHT:
# Carrier B generates significantly more revenue (~250k) compared to
# Carrier A (~140k) and Carrier C (~180k). The company should prioritize
# Carrier B for high-value shipments.
# -----------------------------------------------------------------------------

total_revenue = df.groupby('Shipping carriers')['Revenue generated'].sum().reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(
    x=total_revenue['Shipping carriers'],
    y=total_revenue['Revenue generated']
))
fig.update_layout(
    title='Total Revenue by Shipping Carrier',
    xaxis_title='Shipping Carrier',
    yaxis_title='Revenue Generated'
)
fig.show()


# -----------------------------------------------------------------------------
# SECTION 6 — DEFECT RATES BY PRODUCT TYPE
# -----------------------------------------------------------------------------
#
# OBJECTIVE:
# Identify which product type has the highest proportion of defects
# so the business can prioritize quality control efforts.
#
# WHAT IS A DEFECT RATE?
# A defect rate is the percentage or count of products that fail
# quality inspection. A high defect rate means more waste, returns,
# and customer dissatisfaction — directly hurting revenue.
#
# df.groupby('Product type')['Defect rates'].sum()
#                     → Groups by product type and sums up all
#                       defect rate values for each group.
#
# The same px.pie() approach from Section 4 is used here, but
# now the values represent defect rates instead of units sold.
#
# INSIGHT:
# Skincare has the highest defect rate at 41%, followed by
# Haircare at 37.1% and Cosmetics at 21.9%.
# This is a serious quality control concern — skincare leads in
# both sales AND defects, which could damage brand reputation
# and increase return/replacement costs if not addressed.
# -----------------------------------------------------------------------------

defect_rates = df.groupby('Product type')['Defect rates'].sum().reset_index()

pie_chart = px.pie(
    defect_rates,
    values='Defect rates',
    names='Product type',
    title='Defective by Product Type',
    hole=0.5,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
pie_chart.update_traces(textposition='inside', textinfo='percent+label')
pie_chart.show()


# =============================================================================
# SUMMARY OF KEY INSIGHTS
# =============================================================================
#
# 1. PRICE vs REVENUE   → Skincare shows a strong positive price-revenue
#                          relationship. Price it higher = earn more.
#
# 2. SALES SHARE        → Skincare dominates at 45% of total units sold.
#                          Haircare and Cosmetics together make up 55%.
#
# 3. SHIPPING CARRIERS  → Carrier B is the top revenue-generating carrier.
#                          Consider increasing shipment volume via Carrier B.
#
# 4. DEFECT RATES       → Skincare has the most defects (41%).
#                          Quality control improvements needed urgently
#                          in the skincare production line.
#
# =============================================================================
# END OF ANALYSIS
# =============================================================================