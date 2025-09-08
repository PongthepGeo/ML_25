import numpy as np
import matplotlib.pyplot as plt

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
print(f"\n📊 Data: {{(x_i, y_i)}}_{{{1}}}^{{{n}}}")
print("-" * 30)
for i in range(n):
    print(f"(x_{i+1}, y_{i+1}) = ({x_values[i]:.1f}, {y_values[i]:.2f})")

# Parameters β = [β₀, β₁]ᵀ (we'll estimate these)
# For demonstration, let's use some estimated values
beta0_hat = 1.8  # intercept estimate
beta1_hat = 1.6  # slope estimate
beta = np.array([[beta0_hat], [beta1_hat]])

print(f"\n🔢 Parameters: β = [β₀, β₁]ᵀ")
print("-" * 30)
print(f"β₀ (intercept) = {beta0_hat}")
print(f"β₁ (slope) = {beta1_hat}")
print(f"β = {beta.flatten()}")

# Design matrix X and vectors
print(f"\n📐 Design Matrix X ∈ ℝ^{{{n}×2}} and Vectors")
print("-" * 40)

# Design matrix: X = [𝟙 | x] where 𝟙 is column of ones
ones_column = np.ones(n)
X = np.column_stack([ones_column, x_values])
print("X = [𝟙 | x] =")
print(X)

# Response vector y
y = y_values.reshape(-1, 1)  # column vector
print(f"\ny = (y₁, ..., y_{n})ᵀ =")
print(y.flatten())

# Predicted values: ŷ = Xβ
y_hat = X @ beta
print(f"\nŷ = Xβ =")
print(y_hat.flatten())

# Show the matrix multiplication step by step
print(f"\n🔍 Matrix Multiplication Breakdown:")
print(f"ŷᵢ = β₀ + β₁xᵢ")
for i in range(n):
    pred_i = beta0_hat + beta1_hat * x_values[i]
    print(f"ŷ_{i+1} = {beta0_hat} + {beta1_hat}×{x_values[i]} = {pred_i:.2f}")

# Residuals (errors): r = ŷ - y (note: this is predicted minus actual)
print(f"\n📏 Residuals (errors): rᵢ = ŷᵢ - yᵢ")
print("-" * 40)
residuals = y_hat - y
print("Individual residuals:")
for i in range(n):
    print(f"r_{i+1} = ŷ_{i+1} - y_{i+1} = {y_hat[i,0]:.2f} - {y_values[i]:.2f} = {residuals[i,0]:.2f}")

print(f"\nr = Xβ - y =")
print(residuals.flatten())

# Alternative formula shown in the notation: rᵢ = β₀ + β₁xᵢ - yᵢ
print(f"\n🔄 Alternative calculation: rᵢ = β₀ + β₁xᵢ - yᵢ")
for i in range(n):
    alt_residual = beta0_hat + beta1_hat * x_values[i] - y_values[i]
    print(f"r_{i+1} = {beta0_hat} + {beta1_hat}×{x_values[i]} - {y_values[i]:.2f} = {alt_residual:.2f}")

# Summary statistics
print(f"\n📊 Summary Statistics")
print("-" * 25)
print(f"Sum of residuals: {np.sum(residuals):.3f}")
print(f"Sum of squared residuals: {np.sum(residuals**2):.3f}")
print(f"Mean squared error: {np.mean(residuals**2):.3f}")

# Visualization
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(x_values, y_values, color='blue', s=100, alpha=0.7, label='Data points (xᵢ, yᵢ)')

# Plot regression line
x_line = np.linspace(0, 7, 100)
y_line = beta0_hat + beta1_hat * x_line
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'ŷ = {beta0_hat} + {beta1_hat}x')

# Plot predicted points
plt.scatter(x_values, y_hat.flatten(), color='red', s=80, alpha=0.7, marker='x', label='Predicted ŷᵢ')

# Draw residual lines
for i in range(n):
    plt.plot([x_values[i], x_values[i]], [y_values[i], y_hat[i,0]], 'g--', alpha=0.7)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression: Data, Predictions, and Residuals')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n✅ The green dashed lines show the residuals (errors)")
print(f"   Each residual rᵢ represents the vertical distance from yᵢ to ŷᵢ")