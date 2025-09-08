import numpy as np

# Generate sample data points {(x_i, y_i)}_{i=1}^n
np.random.seed(42)
n = 6  # number of data points

# Create some sample x values and y values
x_values = np.array([1, 2, 3, 4, 5, 6])
true_beta0, true_beta1 = 2.0, 1.5
y_values = true_beta0 + true_beta1 * x_values + np.random.normal(0, 0.5, n)

print("=" * 50)
print("THE QUADRATIC LOSS FUNCTION")
print("=" * 50)

# Display the data
print(f"\n📊 Data: {{(x_i, y_i)}}_{{{1}}}^{{{n}}}")
for i in range(n):
    print(f"(x_{i+1}, y_{i+1}) = ({x_values[i]:.1f}, {y_values[i]:.2f})")

# Parameters β (using some example estimates)
beta0_hat = 1.8  # intercept estimate
beta1_hat = 1.6  # slope estimate
beta = np.array([[beta0_hat], [beta1_hat]])

print(f"\n🔢 Parameters: β = [β₀, β₁]ᵀ = [{beta0_hat}, {beta1_hat}]ᵀ")

# Design matrix and residuals
X = np.column_stack([np.ones(n), x_values])
y = y_values.reshape(-1, 1)
residuals = X @ beta - y  # r = Xβ - y

print(f"\n📏 Residuals: r_i = β₀ + β₁x_i - y_i")
for i in range(n):
    r_i = beta0_hat + beta1_hat * x_values[i] - y_values[i]
    print(f"r_{i+1} = {beta0_hat} + {beta1_hat}×{x_values[i]} - {y_values[i]:.2f} = {r_i:.3f}")

print(f"\nr = {residuals.flatten()}")

print(f"\n" + "="*50)
print("QUADRATIC LOSS CALCULATION")
print("="*50)

# Method 1: Sum of squared residuals
print(f"\n🎯 Method 1: L(β) = (1/2n) Σᵢ rᵢ²")
individual_squares = residuals.flatten()**2
sum_squared = np.sum(individual_squares)
loss_method1 = (1/(2*n)) * sum_squared

print("Individual squared residuals:")
for i in range(n):
    print(f"r_{i+1}² = ({residuals[i,0]:.3f})² = {individual_squares[i]:.4f}")

print(f"\nΣᵢ rᵢ² = {sum_squared:.4f}")
print(f"L(β) = (1/2n) × {sum_squared:.4f} = (1/{2*n}) × {sum_squared:.4f} = {loss_method1:.4f}")

# Method 2: Using L2 norm
print(f"\n🎯 Method 2: L(β) = (1/2n) ||r||₂²")
l2_norm = np.linalg.norm(residuals)
l2_norm_squared = l2_norm**2
loss_method2 = (1/(2*n)) * l2_norm_squared

print(f"||r||₂ = √(Σᵢ rᵢ²) = √{sum_squared:.4f} = {l2_norm:.4f}")
print(f"||r||₂² = {l2_norm_squared:.4f}")
print(f"L(β) = (1/2n) × {l2_norm_squared:.4f} = {loss_method2:.4f}")

print(f"\n✓ Both methods give: L(β) = {loss_method1:.4f}")

print(f"\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"📐 Residuals vector: r = Xβ - y")
print(f"🎯 Quadratic Loss: L(β) = (1/2n)||r||₂² = (1/2n)Σᵢ rᵢ²")
print(f"📊 Current loss value: L(β) = {loss_method1:.4f}")
print(f"💡 This measures the average squared prediction error")