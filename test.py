import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1 & 2. Create Dummy Data (x = index, y = height)
# ==========================================
np.random.seed(42) # For reproducibility
num_samples = 20

# x is simply the index of each data point (0 to 19)
x = np.arange(num_samples)

# y is height. We create a base linear trend and add some random noise
# True hidden equation: height = 2.5 * index + 150
true_slope = 2.5
true_intercept = 150
noise = np.random.normal(0, 5, size=num_samples) # Random variation/noise

y = true_slope * x + true_intercept + noise

# ==========================================
# 3. Build Regression Equation (y = mx + c)
# ==========================================
# Calculate means of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate the slope (m) and intercept (c) using Ordinary Least Squares (OLS) formulas
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean)**2)

m = numerator / denominator
c = y_mean - (m * x_mean)

print(f"Regression Equation: y = {m:.2f}x + {c:.2f}")

# Calculate predicted y values using our new equation
y_pred = m * x + c

# ==========================================
# 4. Create Measuring Error (Cost Function)
# ==========================================
# We use Mean Squared Error (MSE) as the cost function
N = len(x)
mse = np.sum((y - y_pred)**2) / N

# Root Mean Squared Error (RMSE) is also helpful for understanding error in original units
rmse = np.sqrt(mse)

print(f"Mean Squared Error (Cost): {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f} (units of height)")

# ==========================================
# 5. Plot
# ==========================================
plt.figure(figsize=(10, 6))

# Plot the actual dummy data points
plt.scatter(x, y, color='blue', s=60, label='Actual Data (x=index, y=height)')

# Plot the regression line
plt.plot(x, y_pred, color='red', linewidth=2, label=f'Regression Line: y = {m:.2f}x + {c:.2f}')

# Plot error lines (residuals) to visualize the cost function
for i in range(N):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]], color='gray', linestyle='--', linewidth=1)

# Add labels and title
plt.xlabel('Index (x)', fontsize=12)
plt.ylabel('Height (y)', fontsize=12)
plt.title(f'Linear Regression\nMSE (Cost Function) = {mse:.2f}', fontsize=14)

# Add a legend and grid
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)

# Display the plot
plt.show()