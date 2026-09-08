import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from lib.control_plot import PLOT_PARAMS  # shared global plot style

matplotlib.rcParams.update(PLOT_PARAMS)

# Output folder (one folder per script, named after the script)
OUTDIR = Path("08_residuals")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Generate sample data points {(x_i, y_i)}_{i=1}^n
np.random.seed(42)
n = 6  # number of data points

# Create some sample x values
x_values = np.array([1, 2, 3, 4, 5, 6])

# Create y values with some linear relationship plus noise
# True relationship: y = 2 + 1.5*x + noise
true_beta0, true_beta1 = 2.0, 1.5
y_values = true_beta0 + true_beta1 * x_values + np.random.normal(0, 0.5, n)

print("=" * 60)
print("LINEAR REGRESSION SETUP & NOTATION")
print("=" * 60)

# Display the data
print(f"\nData: {{(x_i, y_i)}}_{{{1}}}^{{{n}}}")
print("-" * 30)
for i in range(n):
    print(f"(x_{i+1}, y_{i+1}) = ({x_values[i]:.1f}, {y_values[i]:.2f})")

# Parameters beta = [beta_0, beta_1]^T (we'll estimate these)
# For demonstration, let's use some estimated values
beta0_hat = 1.8  # intercept estimate
beta1_hat = 1.6  # slope estimate
beta = np.array([[beta0_hat], [beta1_hat]])

print(f"\nParameters: beta = [beta_0, beta_1]^T")
print("-" * 30)
print(f"beta_0 (intercept) = {beta0_hat}")
print(f"beta_1 (slope) = {beta1_hat}")
print(f"beta = {beta.flatten()}")

# Design matrix X and vectors
print(f"\nDesign Matrix X in R^{{{n}x2}} and Vectors")
print("-" * 40)

# Design matrix: X = [1 | x] where 1 is column of ones
ones_column = np.ones(n)
X = np.column_stack([ones_column, x_values])
print("X = [1 | x] =")
print(X)

# Response vector y
y = y_values.reshape(-1, 1)  # column vector
print(f"\ny = (y_1, ..., y_{n})^T =")
print(y.flatten())

# Predicted values: y_hat = X beta
y_hat = X @ beta
print(f"\ny_hat = X beta =")
print(y_hat.flatten())

# Show the matrix multiplication step by step
print(f"\nMatrix Multiplication Breakdown:")
print(f"y_hat_i = beta_0 + beta_1 * x_i")
for i in range(n):
    pred_i = beta0_hat + beta1_hat * x_values[i]
    print(f"y_hat_{i+1} = {beta0_hat} + {beta1_hat}*{x_values[i]} = {pred_i:.2f}")

# Residuals (errors): r = y_hat - y (note: this is predicted minus actual)
print(f"\nResiduals (errors): r_i = y_hat_i - y_i")
print("-" * 40)
residuals = y_hat - y
print("Individual residuals:")
for i in range(n):
    print(f"r_{i+1} = y_hat_{i+1} - y_{i+1} = {y_hat[i,0]:.2f} - {y_values[i]:.2f} = {residuals[i,0]:.2f}")

print(f"\nr = X beta - y =")
print(residuals.flatten())

# Alternative formula shown in the notation: r_i = beta_0 + beta_1 * x_i - y_i
print(f"\nAlternative calculation: r_i = beta_0 + beta_1 * x_i - y_i")
for i in range(n):
    alt_residual = beta0_hat + beta1_hat * x_values[i] - y_values[i]
    print(f"r_{i+1} = {beta0_hat} + {beta1_hat}*{x_values[i]} - {y_values[i]:.2f} = {alt_residual:.2f}")

# Summary statistics
print(f"\nSummary Statistics")
print("-" * 25)
print(f"Sum of residuals: {np.sum(residuals):.3f}")
print(f"Sum of squared residuals: {np.sum(residuals**2):.3f}")
print(f"Mean squared error: {np.mean(residuals**2):.3f}")

# Visualization (global figure.figsize default)
plt.figure()

# Plot data points
plt.scatter(x_values, y_values, color='blue', s=100, alpha=0.7, label=r'Data points $(x_i, y_i)$')

# Plot regression line
x_line = np.linspace(0, 7, 100)
y_line = beta0_hat + beta1_hat * x_line
plt.plot(x_line, y_line, 'r-', linewidth=2, label=rf'$\hat{{y}} = {beta0_hat} + {beta1_hat}x$')

# Plot predicted points
plt.scatter(x_values, y_hat.flatten(), color='red', s=80, alpha=0.7, marker='x', label=r'Predicted $\hat{y}_i$')

# Draw residual lines
for i in range(n):
    plt.plot([x_values[i], x_values[i]], [y_values[i], y_hat[i,0]], 'g--', alpha=0.7)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression: Data, Predictions, and Residuals')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig_residuals = OUTDIR / 'residuals.png'
plt.savefig(fig_residuals, format='png', bbox_inches='tight')
print('Saved figure:', fig_residuals)
plt.show()

print(f"\nThe green dashed lines show the residuals (errors)")
print(f"   Each residual r_i represents the vertical distance from y_i to y_hat_i")
