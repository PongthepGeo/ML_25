# One-file: Linear regression per facies + one global regression
# --------------------------------------------------------------
# - Reads well_log.csv, downsamples every 10th row (as in your K-Means script)
# - Fits a separate LinearRegression (ILD_log10 ~ GR) for EACH facies
# - Fits ONE global LinearRegression across ALL facies
# - Plots high-contrast, color-coded scatter with black edges
# - Draws one regression line per facies (colored) + one global line (black, dashed)
# - Prints a neat summary table of slope/intercept and metrics (R2, RMSE, MAE)
#
# Assumes columns: 'GR', 'ILD_log10', 'Facies'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.util import high_contrast_colors, build_code_to_name, compute_metrics, fit_lr, plot_scatter_with_regression

# -----------------------
# Config & data loading
# -----------------------
csv_path = 'dataset/well_log.csv'   # change if needed
data = pd.read_csv(csv_path)

# basic cleanup & downsample like before
cols_needed = ['GR', 'ILD_log10', 'Facies']
data = data[cols_needed].dropna().iloc[::10].copy()

x = data['GR'].to_numpy()
y = data['ILD_log10'].to_numpy()

# Provided facies names & (optional) colors
lithofacies = ['SS','CSiS','FSiS','SiSh','MS','WS','D','PS','BS']
lithocolors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72',
               '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

# -----------------------
# Facies name mapping
# -----------------------

code_to_name, _ = build_code_to_name(data['Facies'], lithofacies)
facies_names = data['Facies'].map(lambda v: code_to_name.get(v, v))

# Colors per facies name
present_names = sorted(pd.unique(facies_names))
if lithocolors and len(lithocolors) >= len(lithofacies):
    name_to_color = {lithofacies[i]: lithocolors[i] for i in range(len(lithofacies))}
else:
    palette = high_contrast_colors(len(present_names))
    name_to_color = {nm: palette[i] for i, nm in enumerate(present_names)}

# -----------------------
# Fit per-facies models
# -----------------------
results = []  # collect dicts with metrics per facies
per_facies_models = {}  # {facies_name: (slope, intercept, predict_fn)}

for fac in present_names:
    mask = (facies_names == fac).to_numpy()
    xi = x[mask]
    yi = y[mask]
    if len(xi) == 0:
        continue
    slope, intercept, predict = fit_lr(xi, yi)
    yhat = predict(xi)
    r2, rmse, mae = compute_metrics(yi, yhat)
    per_facies_models[fac] = (slope, intercept, predict)
    results.append({
        'Facies': fac,
        'N': len(xi),
        'Slope': slope,
        'Intercept': intercept,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    })

# -----------------------
# Fit global model (all facies)
# -----------------------
g_slope, g_intercept, g_predict = fit_lr(x, y)
g_pred = g_predict(x)
g_r2, g_rmse, g_mae = compute_metrics(y, g_pred)
results.append({
    'Facies': 'ALL',
    'N': len(x),
    'Slope': g_slope,
    'Intercept': g_intercept,
    'R2': g_r2,
    'RMSE': g_rmse,
    'MAE': g_mae
})

# -----------------------
# Plot
# -----------------------
plot_scatter_with_regression(
    x, y, facies_names, name_to_color,
    per_facies_models,
    global_model=(g_slope, g_intercept, g_predict),
    title='Well Log: ILD_log10 vs GR — Linear Fits per Facies + Global'
)

# -----------------------
# Summary table (print nicely)
# -----------------------
summary_df = pd.DataFrame(results, columns=['Facies','N','Slope','Intercept','R2','RMSE','MAE'])
# order rows: facies first (alphabetical) then ALL at bottom
summary_df = pd.concat([
    summary_df[summary_df['Facies'] != 'ALL'].sort_values('Facies'),
    summary_df[summary_df['Facies'] == 'ALL']
], ignore_index=True)

pd.set_option('display.float_format', lambda v: f'{v:0.4f}')
print("\nLinear Regression Summary (ILD_log10 ~ GR):")
print(summary_df.to_string(index=False))

# Optional: save summary
out_dir = 'output'; os.makedirs(out_dir, exist_ok=True)
summary_df.to_csv(os.path.join(out_dir, 'linear_regression_summary.csv'), index=False)
