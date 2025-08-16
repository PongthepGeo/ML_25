import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from lib.util import plot_regression

# 1) Load CSV
df = pd.read_csv("dataset/well_log.csv")

# 2–4) Filter by Well Name and Facies
df_filtered = df[(df["Well Name"] == "NEWBY") & (df["Facies"] == 2)]

# 5) Sort by Depth
df_sorted = df_filtered.sort_values(by="Depth")

# Extract x (Depth) and y (GR)
X = df_sorted["Depth"].values.reshape(-1, 1)
y = df_sorted["GR"].values

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plot_regression(X, y, y_pred)