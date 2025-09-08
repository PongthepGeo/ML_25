import numpy as np
import matplotlib.pyplot as plt

# Generate sample data points {(x_i, y_i)}_{i=1}^n
np.random.seed(42)
n = 6  # number of data points

# Create some sample x values and y values
x_values = np.array([1, 2, 3, 4, 5, 6])
true_beta0, true_beta1 = 2.0, 1.5
y_values = true_beta0 + true_beta1 * x_values + np.random.normal(0, 0.5, n)

print("=" * 60)
print("LINEAR REGRESSION: GRADIENT AND HESSIAN OF THE LOSS")
print("=" * 60)

# Display the data
print(f"\n📊 Data: {{(x_i, y_i)}}_{{{1}}}^{{{n}}}")
for i in range(n):
    print(f"(x_{i+1}, y_{i+1}) = ({x_values[i]:.1f}, {y_values[i]:.2f})")

# Parameters β (starting with some initial guess)
beta0_hat = 1.0  # intercept estimate
beta1_hat = 1.0  # slope estimate
beta = np.array([[beta0_hat], [beta1_hat]])

print(f"\n🔢 Initial Parameters: β = [β₀, β₁]ᵀ = [{beta0_hat}, {beta1_hat}]ᵀ")

# Design matrix
X = np.column_stack([np.ones(n), x_values])
y = y_values.reshape(-1, 1)

print(f"\n📐 Design Matrix X and response vector y:")
print("X =")
print(X)
print(f"y = {y.flatten()}")

# Calculate residuals
residuals = X @ beta - y  # r = Xβ - y
print(f"\n📏 Residuals: r = Xβ - y")
print(f"r = {residuals.flatten()}")

# Calculate current loss
current_loss = (1/(2*n)) * np.sum(residuals**2)
print(f"\n🎯 Current Loss: L(β) = (1/2n)||r||₂² = {current_loss:.4f}")

print(f"\n" + "="*60)
print("GRADIENT OF THE LOSS (First Derivative)")
print("="*60)

# Calculate gradient: ∇L(β) = (1/n)Xᵀ(Xβ - y) = (1/n)Xᵀr
gradient = (1/n) * X.T @ residuals
print(f"\n🔍 Gradient Formula: ∇L(β) = (1/n)Xᵀ(Xβ - y) = (1/n)Xᵀr")

print(f"\nStep-by-step calculation:")
print(f"Xᵀ = ")
print(X.T)
print(f"Xᵀr = ")
print((X.T @ residuals).flatten())
print(f"∇L(β) = (1/{n}) × {(X.T @ residuals).flatten()} = {gradient.flatten()}")

print(f"\n💡 Interpretation:")
print(f"   • ∇L/∂β₀ = {gradient[0,0]:.4f}: slope w.r.t. intercept")
print(f"   • ∇L/∂β₁ = {gradient[1,0]:.4f}: slope w.r.t. slope parameter")
print(f"   • Gradient points toward steepest ascent")
print(f"   • Negative gradient points toward steepest descent")

print(f"\n" + "="*60)
print("HESSIAN OF THE LOSS (Second Derivative)")
print("="*60)

# Calculate Hessian: ∇²L(β) = (1/n)XᵀX
hessian = (1/n) * X.T @ X
print(f"\n🔍 Hessian Formula: ∇²L(β) = (1/n)XᵀX")

print(f"\nStep-by-step calculation:")
print(f"XᵀX = ")
print(X.T @ X)
print(f"∇²L(β) = (1/{n}) × XᵀX = ")
print(hessian)

# Analyze Hessian properties
eigenvalues = np.linalg.eigvals(hessian)
condition_number = np.linalg.cond(hessian)
print(f"\n📊 Hessian Properties:")
print(f"   • Eigenvalues: {eigenvalues}")
print(f"   • Condition number: {condition_number:.2f}")
print(f"   • Positive definite: {np.all(eigenvalues > 0)}")
print(f"   • Independent of β: curvature same everywhere")

print(f"\n" + "="*60)
print("GRADIENT DESCENT UPDATE")
print("="*60)

# Gradient descent update: β_{k+1} = β_k - η∇L(β_k)
learning_rate = 0.2
beta_new = beta - learning_rate * gradient

print(f"\n🚀 Update Rule: β_{{k+1}} = β_k - η∇L(β_k)")
print(f"Learning rate η = {learning_rate}")
print(f"\nUpdate step:")
print(f"β_{{k+1}} = {beta.flatten()} - {learning_rate} × {gradient.flatten()}")
print(f"β_{{k+1}} = {beta.flatten()} - {(learning_rate * gradient).flatten()}")
print(f"β_{{k+1}} = {beta_new.flatten()}")

# Calculate new loss
residuals_new = X @ beta_new - y
new_loss = (1/(2*n)) * np.sum(residuals_new**2)
print(f"\n📉 Loss improvement: {current_loss:.4f} → {new_loss:.4f}")
print(f"   Loss reduction: {current_loss - new_loss:.4f}")

print(f"\n" + "="*60)
print("NEWTON'S METHOD UPDATE")
print("="*60)

# Newton's method: β_{k+1} = β_k - (∇²L)^{-1}∇L
hessian_inv = np.linalg.inv(hessian)
beta_newton = beta - hessian_inv @ gradient

print(f"\n🎯 Newton's Method: β_{{k+1}} = β_k - (∇²L)^{{-1}}∇L")
print(f"\nStep-by-step:")
print(f"(∇²L)^{{-1}} = ")
print(hessian_inv)
print(f"(∇²L)^{{-1}}∇L = ")
print((hessian_inv @ gradient).flatten())
print(f"β_{{k+1}} = {beta.flatten()} - {(hessian_inv @ gradient).flatten()}")
print(f"β_{{k+1}} = {beta_newton.flatten()}")

# This should be the analytical solution
residuals_newton = X @ beta_newton - y
newton_loss = (1/(2*n)) * np.sum(residuals_newton**2)
print(f"\n📉 Newton's method loss: {newton_loss:.6f}")

# Verify this is the analytical solution
beta_analytical = np.linalg.inv(X.T @ X) @ (X.T @ y)
print(f"✓ Analytical solution: β̂ = (XᵀX)^{{-1}}Xᵀy = {beta_analytical.flatten()}")

print(f"\n" + "="*60)
print("GEOMETRIC INTERPRETATION")
print("="*60)

print(f"\n🔍 1D Quadratic Analogy:")
print(f"   For L(θ) = aθ² + bθ + c:")
print(f"   • Gradient: dL/dθ = 2aθ + b (slope)")
print(f"   • Hessian: d²L/dθ² = 2a (curvature)")

print(f"\n🔍 Multivariate Case (Our Problem):")
print(f"   • L(β) = (1/2n)||Xβ - y||₂²")
print(f"   • ∇L(β) = (1/n)Xᵀ(Xβ - y) (slope vector)")
print(f"   • ∇²L(β) = (1/n)XᵀX (curvature matrix)")

print(f"\n💡 Key Insights:")
print(f"   • Gradient: direction of steepest ascent")
print(f"   • -Gradient: direction of steepest descent") 
print(f"   • Hessian: reshapes descent directions")
print(f"   • Condition number affects convergence speed")

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Data and current fit
x_line = np.linspace(0, 7, 100)
y_line_current = beta[0,0] + beta[1,0] * x_line
y_line_optimal = beta_analytical[0,0] + beta_analytical[1,0] * x_line

ax1.scatter(x_values, y_values, color='blue', s=100, alpha=0.7, label='Data')
ax1.plot(x_line, y_line_current, 'r--', linewidth=2, label=f'Current: L={current_loss:.4f}')
ax1.plot(x_line, y_line_optimal, 'g-', linewidth=2, label=f'Optimal: L={newton_loss:.6f}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Current vs Optimal Fit')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Gradient vector field
beta0_range = np.linspace(-1, 4, 10)
beta1_range = np.linspace(0, 3, 10)
B0, B1 = np.meshgrid(beta0_range, beta1_range)
Grad0 = np.zeros_like(B0)
Grad1 = np.zeros_like(B1)

for i in range(len(beta0_range)):
    for j in range(len(beta1_range)):
        beta_test = np.array([[B0[j,i]], [B1[j,i]]])
        grad_test = (1/n) * X.T @ (X @ beta_test - y)
        Grad0[j,i] = grad_test[0,0]
        Grad1[j,i] = grad_test[1,0]

ax2.quiver(B0, B1, -Grad0, -Grad1, alpha=0.6)  # Negative for descent direction
ax2.plot(beta[0,0], beta[1,0], 'ro', markersize=10, label='Current β')
ax2.plot(beta_analytical[0,0], beta_analytical[1,0], 'g*', markersize=15, label='Optimal β')
ax2.set_xlabel('β₀')
ax2.set_ylabel('β₁')
ax2.set_title('Gradient Vector Field (Descent Directions)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Loss contours
Loss_surface = np.zeros_like(B0)
for i in range(len(beta0_range)):
    for j in range(len(beta1_range)):
        beta_test = np.array([[B0[j,i]], [B1[j,i]]])
        residuals_test = X @ beta_test - y
        Loss_surface[j,i] = (1/(2*n)) * np.sum(residuals_test**2)

contour = ax3.contour(B0, B1, Loss_surface, levels=15)
ax3.clabel(contour, inline=True, fontsize=8)
ax3.plot(beta[0,0], beta[1,0], 'ro', markersize=10, label='Current β')
ax3.plot(beta_analytical[0,0], beta_analytical[1,0], 'g*', markersize=15, label='Optimal β')
ax3.set_xlabel('β₀')
ax3.set_ylabel('β₁')
ax3.set_title('Loss Function L(β₀,β₁)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Hessian eigenvalue visualization
theta = np.linspace(0, 2*np.pi, 100)
# Create ellipse based on Hessian eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eig(hessian)
angle = np.arctan2(eigenvecs[1,0], eigenvecs[0,0])

# Ellipse parameters
a = 1/np.sqrt(eigenvals[0])  # Semi-major axis
b = 1/np.sqrt(eigenvals[1])  # Semi-minor axis

# Parametric ellipse
ellipse_x = a * np.cos(theta)
ellipse_y = b * np.sin(theta)

# Rotate ellipse
cos_angle = np.cos(angle)
sin_angle = np.sin(angle)
x_rotated = ellipse_x * cos_angle - ellipse_y * sin_angle + beta_analytical[0,0]
y_rotated = ellipse_x * sin_angle + ellipse_y * cos_angle + beta_analytical[1,0]

ax4.plot(x_rotated, y_rotated, 'b-', linewidth=2, label='Hessian Ellipse')
ax4.plot(beta_analytical[0,0], beta_analytical[1,0], 'g*', markersize=15, label='Optimal β')
ax4.arrow(beta_analytical[0,0], beta_analytical[1,0], 
          eigenvecs[0,0]*a, eigenvecs[1,0]*a, 
          head_width=0.05, head_length=0.05, fc='red', ec='red')
ax4.arrow(beta_analytical[0,0], beta_analytical[1,0], 
          eigenvecs[0,1]*b, eigenvecs[1,1]*b, 
          head_width=0.05, head_length=0.05, fc='orange', ec='orange')
ax4.set_xlabel('β₀')
ax4.set_ylabel('β₁')
ax4.set_title('Hessian Curvature (Eigenvalue Ellipse)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axis('equal')

plt.tight_layout()
plt.show()

print(f"\n🎯 SUMMARY:")
print(f"   • Gradient ∇L(β) = (1/n)Xᵀr gives descent direction")
print(f"   • Hessian ∇²L(β) = (1/n)XᵀX gives curvature information")
print(f"   • Gradient descent: β_{{k+1}} = β_k - η∇L(β_k)")
print(f"   • Newton's method: β_{{k+1}} = β_k - (∇²L)^{{-1}}∇L(β_k)")
print(f"   • Convex quadratic bowl → unique global minimum")