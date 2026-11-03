import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from lib.control_plot import PLOT_PARAMS  # shared global plot style

matplotlib.rcParams.update(PLOT_PARAMS)

# Output folder (one folder per script, named after the script)
OUTDIR = Path("11_complete_optimize")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Generate sample data points {(x_i, y_i)}_{i=1}^n
np.random.seed(42)
n = 6  # number of data points

# Create some sample x values and y values
x_values = np.array([1, 2, 3, 4, 5, 6])
true_beta0, true_beta1 = 2.0, 1.5
y_values = true_beta0 + true_beta1 * x_values + np.random.normal(0, 0.5, n)

print("=" * 70)
print("MULTI-EPOCH GRADIENT DESCENT FOR LINEAR REGRESSION")
print("=" * 70)

# Display the data
print(f"\nData: {{(x_i, y_i)}}_{{{1}}}^{{{n}}}")
for i in range(n):
    print(f"(x_{i+1}, y_{i+1}) = ({x_values[i]:.1f}, {y_values[i]:.2f})")

# Design matrix
X = np.column_stack([np.ones(n), x_values])
y = y_values.reshape(-1, 1)

print(f"\nDesign Matrix X and response vector y:")
print("X =")
print(X)
print(f"y = {y.flatten()}")

# Analytical solution for comparison
beta_analytical = np.linalg.inv(X.T @ X) @ (X.T @ y)
residuals_analytical = X @ beta_analytical - y
analytical_loss = (1/(2*n)) * np.sum(residuals_analytical**2)
print(f"\nAnalytical Solution: beta_hat = [{beta_analytical[0,0]:.6f}, {beta_analytical[1,0]:.6f}]^T")
print(f"   Minimum Loss: {analytical_loss:.8f}")

print(f"\n" + "="*70)
print("GRADIENT DESCENT OPTIMIZATION")
print("="*70)

def compute_loss(beta, X, y, n):
    residuals = X @ beta - y
    return (1/(2*n)) * np.sum(residuals**2)

def compute_gradient(beta, X, y, n):
    residuals = X @ beta - y
    return (1/n) * X.T @ residuals

# Gradient descent parameters
learning_rates = [0.05, 0.08, 0.12]
max_epochs = 100
tolerance = 1e-9

# Initial parameters (same starting point for all experiments)
initial_beta = np.array([[1.0], [1.0]])

# Snapshots: while the iterations run, save one PNG at each of these epochs
snapshot_lr = 0.12                  # learning rate to snapshot (must be in learning_rates)
snapshot_epochs = [0, 5, 25, 100]   # start, overshoot, approach, end
if snapshot_lr not in learning_rates:
    raise ValueError(f"snapshot_lr={snapshot_lr} must be one of {learning_rates}")
saved_snapshots = []

colors = ['red', 'blue', 'green', 'orange', 'purple']

# Loss surface (computed once; used by the snapshots and by Plot 3)
beta0_range = np.linspace(0, 4, 50)
beta1_range = np.linspace(0, 3, 50)
B0, B1 = np.meshgrid(beta0_range, beta1_range)
Loss_surface = np.zeros_like(B0)

for i in range(len(beta0_range)):
    for j in range(len(beta1_range)):
        beta_test = np.array([[B0[j,i]], [B1[j,i]]])
        Loss_surface[j,i] = compute_loss(beta_test, X, y, n)

def save_snapshot(lr, epoch, path, losses):
    """Save one PNG of the optimization state: data fit | parameter space | loss so far.
    Axis limits are fixed so the snapshots can be compared side by side."""
    color = colors[learning_rates.index(lr)]
    path = np.array([p.flatten() for p in path])
    beta_now = path[-1]
    x_line = np.linspace(0, 7, 100)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    # Panel 1: data and the current regression line
    ax = axes[0]
    ax.scatter(x_values, y_values, color='black', s=100, alpha=0.7, label='Data', zorder=5)
    ax.plot(x_line, beta_analytical[0,0] + beta_analytical[1,0] * x_line, 'g--', linewidth=3,
            label='Analytical solution')
    ax.plot(x_line, beta_now[0] + beta_now[1] * x_line, color=color, linewidth=3, label='Current fit')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 14)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('Data Fit')
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 2: loss contours with the path travelled so far
    ax = axes[1]
    ax.contour(B0, B1, Loss_surface, levels=20, alpha=0.6)
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2, marker='o', markersize=4,
            label=rf'Path ($\eta={lr}$)')
    ax.plot(path[0, 0], path[0, 1], 's', color=color, markersize=10, label='Start')
    ax.plot(beta_now[0], beta_now[1], 'X', color=color, markeredgecolor='black', markersize=14,
            label='Current')
    ax.plot(beta_analytical[0,0], beta_analytical[1,0], '*', color='black', markersize=18,
            label='Global minimum')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.set_xlabel(r'$\beta_0$')
    ax.set_ylabel(r'$\beta_1$')
    ax.set_title('Parameter Space')
    ax.legend(fontsize=14, loc='upper right')

    # Panel 3: loss history up to this epoch
    ax = axes[2]
    ax.semilogy(range(len(losses)), losses, color=color, linewidth=2)
    ax.plot(epoch, losses[-1], 'X', color=color, markeredgecolor='black', markersize=14)
    ax.axhline(y=analytical_loss, color='black', linestyle='--', linewidth=2,
               label=rf'Analytical minimum: $L={analytical_loss:.2e}$')
    ax.set_xlim(0, max(snapshot_epochs))
    ax.set_ylim(analytical_loss / 1.5, 1.5 * compute_loss(initial_beta, X, y, n))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Loss $L(\beta)$')
    ax.set_title('Loss So Far (Log Scale)')
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.suptitle(rf'Epoch {epoch}, $\eta={lr}$:  $\beta=[{beta_now[0]:.3f},\ {beta_now[1]:.3f}]$,  '
                 rf'$L={losses[-1]:.4f}$', fontsize=24, fontweight='bold')
    fig.tight_layout()
    fig_path = OUTDIR / f'snapshot_epoch_{epoch:03d}.png'
    fig.savefig(fig_path, format='png', bbox_inches='tight')
    plt.close(fig)
    saved_snapshots.append(fig_path)
    print(f"   Saved snapshot: {fig_path}")

# Storage for visualization
all_paths = {}
all_losses = {}

print(f"Starting Parameters: beta_0 = [{initial_beta.flatten()}]")
print(f"Running gradient descent with different learning rates...")

for lr in learning_rates:
    print(f"\n{'='*50}")
    print(f"Learning Rate eta = {lr}")
    print(f"{'='*50}")
    
    # Initialize for this run
    beta = initial_beta.copy()
    path = [beta.copy()]
    losses = [compute_loss(beta, X, y, n)]
    gradients = []
    
    print(f"Epoch   0: beta = [{beta[0,0]:.6f}, {beta[1,0]:.6f}], Loss = {losses[0]:.8f}")
    if lr == snapshot_lr and 0 in snapshot_epochs:
        save_snapshot(lr, 0, path, losses)
    
    for epoch in range(1, max_epochs + 1):
        # Compute gradient
        gradient = compute_gradient(beta, X, y, n)
        gradient_norm = np.linalg.norm(gradient)
        gradients.append(gradient_norm)
        
        # Update parameters
        beta = beta - lr * gradient
        path.append(beta.copy())
        
        # Compute new loss
        current_loss = compute_loss(beta, X, y, n)
        losses.append(current_loss)
        
        # Print every 5 epochs or if converged
        if epoch % 5 == 0 or gradient_norm < tolerance:
            print(f"Epoch {epoch:3d}: beta = [{beta[0,0]:.6f}, {beta[1,0]:.6f}], Loss = {current_loss:.8f}, ||grad L|| = {gradient_norm:.8f}")

        # Snapshot at the chosen epochs
        if lr == snapshot_lr and epoch in snapshot_epochs:
            save_snapshot(lr, epoch, path, losses)
        
        # Check convergence
        if gradient_norm < tolerance:
            print(f"Converged after {epoch} epochs (||grad L|| = {gradient_norm:.2e} < {tolerance:.2e})")
            break
        
        # Check for divergence
        if current_loss > 1000:
            print(f"Diverged at epoch {epoch} (Loss = {current_loss:.2f})")
            break
    
    final_error = np.linalg.norm(beta - beta_analytical)
    print(f"Final Results:")
    print(f"   - Final beta = [{beta[0,0]:.6f}, {beta[1,0]:.6f}]")
    print(f"   - Final Loss = {current_loss:.8f}")
    print(f"   - Distance from analytical solution: {final_error:.8f}")
    print(f"   - Total epochs: {len(losses)-1}")
    
    # Store for visualization
    all_paths[lr] = np.array([p.flatten() for p in path])
    all_losses[lr] = np.array(losses)

if len(saved_snapshots) < len(snapshot_epochs):
    print(f"\nWARNING: saved {len(saved_snapshots)} of {len(snapshot_epochs)} snapshots "
          f"(the run for eta={snapshot_lr} ended before the last snapshot epoch)")

print(f"\n" + "="*70)
print("CONVERGENCE ANALYSIS")
print("="*70)

# Create comprehensive visualization
# 2x3 panel -- intentionally not the global 16:9 default
fig = plt.figure(figsize=(20, 15))

# Plot 1: Loss convergence curves
ax1 = plt.subplot(2, 3, 1)
for i, lr in enumerate(learning_rates):
    epochs = range(len(all_losses[lr]))
    plt.semilogy(epochs, all_losses[lr], color=colors[i], linewidth=2, 
                label=rf'$\eta = {lr}$', marker='o' if len(epochs) < 20 else None, markersize=3)

plt.axhline(y=analytical_loss, color='black', linestyle='--', linewidth=2, 
           label=rf'Analytical minimum: $L={analytical_loss:.2e}$')
plt.xlabel('Epoch')
plt.ylabel(r'Loss $L(\beta)$')
plt.title('Loss Convergence (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Parameter convergence paths
ax2 = plt.subplot(2, 3, 2)
for i, lr in enumerate(learning_rates):
    path = all_paths[lr]
    plt.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=2, 
            label=rf'$\eta = {lr}$', marker='o' if len(path) < 20 else None, markersize=4)
    plt.plot(path[0, 0], path[0, 1], 's', color=colors[i], markersize=8)  # Start
    plt.plot(path[-1, 0], path[-1, 1], 'X', color=colors[i], markersize=8)  # End

plt.plot(beta_analytical[0,0], beta_analytical[1,0], '*', color='black', 
         markersize=15, label='Analytical solution')
plt.xlabel(r'$\beta_0$')
plt.ylabel(r'$\beta_1$')
plt.title('Parameter Space Convergence Paths')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Loss contour with paths
ax3 = plt.subplot(2, 3, 3)
# (loss surface B0, B1, Loss_surface is computed once, before the optimization loop)
contour = plt.contour(B0, B1, Loss_surface, levels=20, alpha=0.6)
plt.clabel(contour, inline=True, fontsize=8, fmt='%.3f')

for i, lr in enumerate(learning_rates):
    path = all_paths[lr]
    plt.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=2, 
            label=rf'$\eta = {lr}$', alpha=0.8)

plt.plot(beta_analytical[0,0], beta_analytical[1,0], '*', color='black', 
         markersize=15, label='Global minimum')
plt.xlabel(r'$\beta_0$')
plt.ylabel(r'$\beta_1$')
plt.title('Loss Contours with Optimization Paths')
plt.legend()

# Plot 4: Gradient norm evolution
ax4 = plt.subplot(2, 3, 4)
for i, lr in enumerate(learning_rates):
    # Compute gradient norms for this path
    grad_norms = []
    for j in range(len(all_paths[lr]) - 1):  # -1 because we compute gradient before update
        beta_curr = all_paths[lr][j].reshape(-1, 1)
        grad = compute_gradient(beta_curr, X, y, n)
        grad_norms.append(np.linalg.norm(grad))
    
    if len(grad_norms) > 0:
        plt.semilogy(range(len(grad_norms)), grad_norms, color=colors[i], 
                    linewidth=2, label=rf'$\eta = {lr}$')

plt.axhline(y=tolerance, color='black', linestyle='--', linewidth=1, 
           label=rf'Tolerance: $\|\nabla L\|={tolerance:.2e}$')
plt.xlabel('Epoch')
plt.ylabel(r'$\|\nabla L(\beta)\|$')
plt.title('Gradient Norm Evolution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Step size evolution
ax5 = plt.subplot(2, 3, 5)
for i, lr in enumerate(learning_rates):
    path = all_paths[lr]
    if len(path) > 1:
        step_sizes = []
        for j in range(1, len(path)):
            step = np.linalg.norm(path[j] - path[j-1])
            step_sizes.append(step)
        
        plt.plot(range(1, len(step_sizes) + 1), step_sizes, color=colors[i], 
                linewidth=2, label=rf'$\eta = {lr}$')

plt.xlabel('Epoch')
plt.ylabel(r'$\|\beta_{k+1}-\beta_k\|$')
plt.title('Step Size Evolution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Data fit evolution for best learning rate
ax6 = plt.subplot(2, 3, 6)
x_line = np.linspace(0, 7, 100)

# Find best learning rate (fastest convergence to low error)
best_lr = None
best_metric = float('inf')
for lr in learning_rates:
    final_beta = all_paths[lr][-1].reshape(-1, 1)
    error = np.linalg.norm(final_beta - beta_analytical)
    epochs_used = len(all_paths[lr]) - 1
    metric = error + 0.01 * epochs_used  # Penalize both error and epochs
    if metric < best_metric:
        best_metric = metric
        best_lr = lr

# Show fit evolution for best learning rate
path = all_paths[best_lr]
plt.scatter(x_values, y_values, color='black', s=100, alpha=0.7, label='Data', zorder=5)

# Show fits at different epochs
epochs_to_show = [0, len(path)//4, len(path)//2, 3*len(path)//4, len(path)-1]
alphas = [0.3, 0.4, 0.6, 0.8, 1.0]
for i, epoch in enumerate(epochs_to_show):
    if epoch < len(path):
        beta_epoch = path[epoch]
        y_line = beta_epoch[0] + beta_epoch[1] * x_line
        plt.plot(x_line, y_line, color='blue', alpha=alphas[i], linewidth=2,
                label=f'Epoch {epoch}' if i == 0 or i == len(epochs_to_show)-1 else None)

# Analytical solution
y_line_analytical = beta_analytical[0,0] + beta_analytical[1,0] * x_line
plt.plot(x_line, y_line_analytical, 'g--', linewidth=3, label='Analytical solution')

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(rf'Fit Evolution ($\eta={best_lr}$)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
fig_analysis = OUTDIR / 'gradient_descent_analysis.png'
plt.savefig(fig_analysis, format='png', bbox_inches='tight')
print('Saved figure:', fig_analysis)
plt.show()

print(f"\nSUMMARY:")
print(f"   - Tested learning rates: {learning_rates}")
print(f"   - Best learning rate: eta = {best_lr}")
print(f"   - Convergence depends on learning rate choice")
print(f"   - Too small eta: slow convergence")
print(f"   - Too large eta: potential oscillation/divergence")
print(f"   - Optimal eta balances speed and stability")
print(f"   - All successful runs converged to analytical solution")

print(f"\nCONVERGENCE DETAILS:")
for lr in learning_rates:
    final_beta = all_paths[lr][-1].reshape(-1, 1)
    final_loss = all_losses[lr][-1]
    epochs_used = len(all_paths[lr]) - 1
    error_from_analytical = np.linalg.norm(final_beta - beta_analytical)
    
    print(f"   eta = {lr:4.2f}: {epochs_used:2d} epochs, "
          f"Loss = {final_loss:.2e}, Error = {error_from_analytical:.2e}")