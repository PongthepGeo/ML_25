import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from lib.control_plot import PLOT_PARAMS


# ============================================================
# GLOBAL PLOT STYLE
# ============================================================
matplotlib.rcParams.update(PLOT_PARAMS)


# ============================================================
# OUTPUT
# ============================================================
OUTDIR = Path("10_optimize")
OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GENERATE SAMPLE DATA
#
# {(x_i, y_i)}_{i=1}^n
# ============================================================
np.random.seed(42)

n = 6

x_values = np.array(
    [1, 2, 3, 4, 5, 6]
)

true_beta0 = 2.0
true_beta1 = 1.5

y_values = (
    true_beta0
    + true_beta1 * x_values
    + np.random.normal(
        0,
        0.5,
        n
    )
)


print("=" * 60)
print(
    "LINEAR REGRESSION: "
    "GRADIENT AND HESSIAN OF THE LOSS"
)
print("=" * 60)


# ============================================================
# DISPLAY DATA
# ============================================================
print(
    f"\nData: {{(x_i, y_i)}}_1^{n}"
)

for i in range(n):

    print(
        f"(x_{i + 1}, y_{i + 1}) = "
        f"({x_values[i]:.1f}, "
        f"{y_values[i]:.2f})"
    )


# ============================================================
# INITIAL PARAMETERS
# ============================================================
beta0_hat = 1.0
beta1_hat = 1.0

beta = np.array(
    [
        [beta0_hat],
        [beta1_hat]
    ]
)


print(
    "\nInitial Parameters:"
)

print(
    f"beta = [beta_0, beta_1]^T = "
    f"[{beta0_hat}, {beta1_hat}]^T"
)


# ============================================================
# DESIGN MATRIX
# ============================================================
X = np.column_stack(
    [
        np.ones(n),
        x_values
    ]
)

y = y_values.reshape(
    -1,
    1
)


print(
    "\nDesign Matrix X and response vector y:"
)

print("X =")
print(X)

print(
    f"y = {y.flatten()}"
)


# ============================================================
# RESIDUALS
#
# r = X beta - y
# ============================================================
residuals = (
    X @ beta
    - y
)


print(
    "\nResiduals:"
)

print(
    "r = X beta - y"
)

print(
    f"r = {residuals.flatten()}"
)


# ============================================================
# CURRENT LOSS
#
# L(beta) = 1/(2n) ||X beta - y||_2^2
# ============================================================
current_loss = (
    1 / (2 * n)
) * np.sum(
    residuals ** 2
)


print(
    "\nCurrent Loss:"
)

print(
    f"L(beta) = "
    f"(1/2n)||r||_2^2 = "
    f"{current_loss:.4f}"
)


# ============================================================
# GRADIENT
# ============================================================
print()
print("=" * 60)
print(
    "GRADIENT OF THE LOSS "
    "(FIRST DERIVATIVE)"
)
print("=" * 60)


# ------------------------------------------------------------
# Gradient:
#
# grad L(beta)
# =
# (1/n) X^T (X beta - y)
# =
# (1/n) X^T r
# ------------------------------------------------------------
gradient = (
    1 / n
) * X.T @ residuals


print(
    "\nGradient Formula:"
)

print(
    "grad L(beta) = "
    "(1/n) X^T (X beta - y)"
)

print(
    "             = "
    "(1/n) X^T r"
)


print(
    "\nStep-by-step calculation:"
)

print(
    "X^T ="
)

print(
    X.T
)


print(
    "\nX^T r ="
)

print(
    (X.T @ residuals).flatten()
)


print(
    f"\ngrad L(beta) = "
    f"(1/{n}) x "
    f"{(X.T @ residuals).flatten()}"
)

print(
    "=",
    gradient.flatten()
)


# ============================================================
# GRADIENT INTERPRETATION
# ============================================================
print(
    "\nInterpretation:"
)

print(
    f"  dL/dbeta_0 = "
    f"{gradient[0, 0]:.4f}: "
    f"slope with respect to intercept"
)

print(
    f"  dL/dbeta_1 = "
    f"{gradient[1, 0]:.4f}: "
    f"slope with respect to slope parameter"
)

print(
    "  Gradient points toward "
    "steepest ascent"
)

print(
    "  Negative gradient points toward "
    "steepest descent"
)


# ============================================================
# HESSIAN
# ============================================================
print()
print("=" * 60)
print(
    "HESSIAN OF THE LOSS "
    "(SECOND DERIVATIVE)"
)
print("=" * 60)


# ------------------------------------------------------------
# Hessian:
#
# H = grad^2 L(beta)
#   = (1/n) X^T X
# ------------------------------------------------------------
hessian = (
    1 / n
) * X.T @ X


print(
    "\nHessian Formula:"
)

print(
    "grad^2 L(beta) = "
    "(1/n) X^T X"
)


print(
    "\nStep-by-step calculation:"
)

print(
    "X^T X ="
)

print(
    X.T @ X
)


print(
    f"\ngrad^2 L(beta) = "
    f"(1/{n}) X^T X ="
)

print(
    hessian
)


# ============================================================
# HESSIAN PROPERTIES
# ============================================================
eigenvalues = np.linalg.eigvals(
    hessian
)

condition_number = np.linalg.cond(
    hessian
)


print(
    "\nHessian Properties:"
)

print(
    f"  Eigenvalues: "
    f"{eigenvalues}"
)

print(
    f"  Condition number: "
    f"{condition_number:.2f}"
)

print(
    f"  Positive definite: "
    f"{np.all(eigenvalues > 0)}"
)

print(
    "  Independent of beta: "
    "curvature is the same everywhere"
)


# ============================================================
# GRADIENT DESCENT UPDATE
# ============================================================
print()
print("=" * 60)
print(
    "GRADIENT DESCENT UPDATE"
)
print("=" * 60)


# ------------------------------------------------------------
# beta_{k+1}
# =
# beta_k - eta grad L(beta_k)
# ------------------------------------------------------------
learning_rate = 0.2


beta_new = (
    beta
    - learning_rate * gradient
)


print(
    "\nUpdate Rule:"
)

print(
    "beta_{k+1} = "
    "beta_k - eta grad L(beta_k)"
)

print(
    f"Learning rate eta = "
    f"{learning_rate}"
)


print(
    "\nUpdate step:"
)

print(
    f"beta_(k+1) = "
    f"{beta.flatten()} "
    f"- {learning_rate} x "
    f"{gradient.flatten()}"
)

print(
    f"beta_(k+1) = "
    f"{beta.flatten()} "
    f"- "
    f"{(learning_rate * gradient).flatten()}"
)

print(
    f"beta_(k+1) = "
    f"{beta_new.flatten()}"
)


# ============================================================
# NEW LOSS
# ============================================================
residuals_new = (
    X @ beta_new
    - y
)


new_loss = (
    1 / (2 * n)
) * np.sum(
    residuals_new ** 2
)


print(
    f"\nLoss improvement: "
    f"{current_loss:.4f} "
    f"-> "
    f"{new_loss:.4f}"
)

print(
    f"Loss reduction: "
    f"{current_loss - new_loss:.4f}"
)


# ============================================================
# NEWTON'S METHOD
# ============================================================
print()
print("=" * 60)
print(
    "NEWTON'S METHOD UPDATE"
)
print("=" * 60)


# ------------------------------------------------------------
# beta_{k+1}
# =
# beta_k
# -
# [grad^2 L]^{-1}
# grad L
# ------------------------------------------------------------
hessian_inv = np.linalg.inv(
    hessian
)


beta_newton = (
    beta
    - hessian_inv @ gradient
)


print(
    "\nNewton's Method:"
)

print(
    "beta_{k+1} = "
    "beta_k - "
    "(grad^2 L)^(-1) grad L"
)


print(
    "\nStep-by-step:"
)

print(
    "(grad^2 L)^(-1) ="
)

print(
    hessian_inv
)


print(
    "\n(grad^2 L)^(-1) grad L ="
)

print(
    (
        hessian_inv
        @ gradient
    ).flatten()
)


print(
    f"\nbeta_(k+1) = "
    f"{beta.flatten()} - "
    f"{(hessian_inv @ gradient).flatten()}"
)

print(
    f"beta_(k+1) = "
    f"{beta_newton.flatten()}"
)


# ============================================================
# NEWTON LOSS
# ============================================================
residuals_newton = (
    X @ beta_newton
    - y
)


newton_loss = (
    1 / (2 * n)
) * np.sum(
    residuals_newton ** 2
)


print(
    f"\nNewton's method loss: "
    f"{newton_loss:.6f}"
)


# ============================================================
# ANALYTICAL SOLUTION
# ============================================================
beta_analytical = (
    np.linalg.inv(
        X.T @ X
    )
    @
    (
        X.T @ y
    )
)


print(
    "\nAnalytical solution:"
)

print(
    "beta_hat = "
    "(X^T X)^(-1) X^T y"
)

print(
    beta_analytical.flatten()
)


# ============================================================
# GEOMETRIC INTERPRETATION
# ============================================================
print()
print("=" * 60)
print(
    "GEOMETRIC INTERPRETATION"
)
print("=" * 60)


print(
    "\n1D Quadratic Analogy:"
)

print(
    "For L(theta) = "
    "a theta^2 + b theta + c:"
)

print(
    "  Gradient: "
    "dL/dtheta = "
    "2a theta + b"
)

print(
    "  Hessian: "
    "d^2L/dtheta^2 = 2a"
)


print(
    "\nMultivariate Case:"
)

print(
    "  L(beta) = "
    "(1/2n)||X beta - y||_2^2"
)

print(
    "  grad L(beta) = "
    "(1/n) X^T (X beta - y)"
)

print(
    "  grad^2 L(beta) = "
    "(1/n) X^T X"
)


print(
    "\nKey Insights:"
)

print(
    "  Gradient: "
    "direction of steepest ascent"
)

print(
    "  -Gradient: "
    "direction of steepest descent"
)

print(
    "  Hessian: "
    "reshapes descent directions"
)

print(
    "  Condition number affects "
    "convergence speed"
)


# ============================================================
# VISUALIZATION
# ============================================================
fig, (
    (ax1, ax2),
    (ax3, ax4)
) = plt.subplots(
    2,
    2,
    figsize=(15, 12)
)


# ============================================================
# PLOT 1
# DATA AND CURRENT / OPTIMAL FIT
# ============================================================
x_line = np.linspace(
    0,
    7,
    100
)


y_line_current = (
    beta[0, 0]
    + beta[1, 0] * x_line
)


y_line_optimal = (
    beta_analytical[0, 0]
    + beta_analytical[1, 0] * x_line
)


ax1.scatter(
    x_values,
    y_values,
    color="blue",
    s=100,
    alpha=0.7,
    label="Data"
)


ax1.plot(
    x_line,
    y_line_current,
    "r--",
    linewidth=2,
    label=(
        rf"Current: "
        rf"$L={current_loss:.4f}$"
    )
)


ax1.plot(
    x_line,
    y_line_optimal,
    "g-",
    linewidth=2,
    label=(
        rf"Optimal: "
        rf"$L={newton_loss:.6f}$"
    )
)


ax1.set_xlabel(
    r"$x$"
)

ax1.set_ylabel(
    r"$y$"
)

ax1.set_title(
    "Current vs Optimal Fit"
)

ax1.legend()

ax1.grid(
    True,
    alpha=0.3
)


# ============================================================
# PLOT 2
# GRADIENT VECTOR FIELD
# ============================================================
beta0_range = np.linspace(
    -1,
    4,
    10
)

beta1_range = np.linspace(
    0,
    3,
    10
)


B0, B1 = np.meshgrid(
    beta0_range,
    beta1_range
)


Grad0 = np.zeros_like(
    B0
)

Grad1 = np.zeros_like(
    B1
)


for i in range(
    len(beta0_range)
):

    for j in range(
        len(beta1_range)
    ):

        beta_test = np.array(
            [
                [B0[j, i]],
                [B1[j, i]]
            ]
        )

        grad_test = (
            1 / n
        ) * X.T @ (
            X @ beta_test
            - y
        )

        Grad0[j, i] = (
            grad_test[0, 0]
        )

        Grad1[j, i] = (
            grad_test[1, 0]
        )


# Negative gradient = descent direction
ax2.quiver(
    B0,
    B1,
    -Grad0,
    -Grad1,
    alpha=0.6
)


ax2.plot(
    beta[0, 0],
    beta[1, 0],
    "ro",
    markersize=10,
    label=r"Current $\beta$"
)


ax2.plot(
    beta_analytical[0, 0],
    beta_analytical[1, 0],
    "g*",
    markersize=15,
    label=r"Optimal $\beta$"
)


ax2.set_xlabel(
    r"$\beta_0$"
)

ax2.set_ylabel(
    r"$\beta_1$"
)

ax2.set_title(
    "Gradient Vector Field "
    "(Descent Directions)"
)

ax2.legend()

ax2.grid(
    True,
    alpha=0.3
)


# ============================================================
# PLOT 3
# LOSS CONTOURS
# ============================================================
Loss_surface = np.zeros_like(
    B0
)


for i in range(
    len(beta0_range)
):

    for j in range(
        len(beta1_range)
    ):

        beta_test = np.array(
            [
                [B0[j, i]],
                [B1[j, i]]
            ]
        )

        residuals_test = (
            X @ beta_test
            - y
        )

        Loss_surface[j, i] = (
            1 / (2 * n)
        ) * np.sum(
            residuals_test ** 2
        )


contour = ax3.contour(
    B0,
    B1,
    Loss_surface,
    levels=15
)


ax3.clabel(
    contour,
    inline=True,
    fontsize=8
)


ax3.plot(
    beta[0, 0],
    beta[1, 0],
    "ro",
    markersize=10,
    label=r"Current $\beta$"
)


ax3.plot(
    beta_analytical[0, 0],
    beta_analytical[1, 0],
    "g*",
    markersize=15,
    label=r"Optimal $\beta$"
)


ax3.set_xlabel(
    r"$\beta_0$"
)

ax3.set_ylabel(
    r"$\beta_1$"
)

ax3.set_title(
    r"Loss Function "
    r"$L(\beta_0,\beta_1)$"
)

ax3.legend()

ax3.grid(
    True,
    alpha=0.3
)


# ============================================================
# PLOT 4
# HESSIAN EIGENVALUE / CURVATURE ELLIPSE
# ============================================================
theta = np.linspace(
    0,
    2 * np.pi,
    100
)


eigenvals, eigenvecs = np.linalg.eig(
    hessian
)


angle = np.arctan2(
    eigenvecs[1, 0],
    eigenvecs[0, 0]
)


# ------------------------------------------------------------
# Ellipse axes
# ------------------------------------------------------------
a = (
    1
    / np.sqrt(
        eigenvals[0]
    )
)


b = (
    1
    / np.sqrt(
        eigenvals[1]
    )
)


# ------------------------------------------------------------
# Parametric ellipse
# ------------------------------------------------------------
ellipse_x = (
    a
    * np.cos(theta)
)

ellipse_y = (
    b
    * np.sin(theta)
)


# ------------------------------------------------------------
# Rotate ellipse
# ------------------------------------------------------------
cos_angle = np.cos(
    angle
)

sin_angle = np.sin(
    angle
)


x_rotated = (
    ellipse_x * cos_angle
    - ellipse_y * sin_angle
    + beta_analytical[0, 0]
)


y_rotated = (
    ellipse_x * sin_angle
    + ellipse_y * cos_angle
    + beta_analytical[1, 0]
)


ax4.plot(
    x_rotated,
    y_rotated,
    "b-",
    linewidth=2,
    label="Hessian Ellipse"
)


ax4.plot(
    beta_analytical[0, 0],
    beta_analytical[1, 0],
    "g*",
    markersize=15,
    label=r"Optimal $\beta$"
)


# ------------------------------------------------------------
# First eigenvector
# ------------------------------------------------------------
ax4.arrow(
    beta_analytical[0, 0],
    beta_analytical[1, 0],

    eigenvecs[0, 0] * a,
    eigenvecs[1, 0] * a,

    head_width=0.05,
    head_length=0.05,
    fc="red",
    ec="red"
)


# ------------------------------------------------------------
# Second eigenvector
# ------------------------------------------------------------
ax4.arrow(
    beta_analytical[0, 0],
    beta_analytical[1, 0],

    eigenvecs[0, 1] * b,
    eigenvecs[1, 1] * b,

    head_width=0.05,
    head_length=0.05,
    fc="orange",
    ec="orange"
)


ax4.set_xlabel(
    r"$\beta_0$"
)

ax4.set_ylabel(
    r"$\beta_1$"
)

ax4.set_title(
    "Hessian Curvature "
    "(Eigenvalue Ellipse)"
)

ax4.legend()

ax4.grid(
    True,
    alpha=0.3
)

ax4.axis(
    "equal"
)


# ============================================================
# SAVE FIGURE
# ============================================================
plt.tight_layout()


fig_overview = (
    OUTDIR
    / "gradient_hessian_overview.png"
)


plt.savefig(
    fig_overview,
    format="png",
    bbox_inches="tight"
)


print(
    f"\nSaved figure: "
    f"{fig_overview}"
)


plt.show()
plt.close()


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    "Gradient:"
)

print(
    "  grad L(beta) = "
    "(1/n) X^T r"
)

print(
    "  Gives the local slope "
    "and descent direction."
)


print(
    "\nHessian:"
)

print(
    "  grad^2 L(beta) = "
    "(1/n) X^T X"
)

print(
    "  Gives curvature information."
)


print(
    "\nGradient descent:"
)

print(
    "  beta_{k+1} = "
    "beta_k - eta grad L(beta_k)"
)


print(
    "\nNewton's method:"
)

print(
    "  beta_{k+1} = "
    "beta_k - "
    "(grad^2 L)^(-1) grad L(beta_k)"
)


print(
    "\nConvex quadratic bowl "
    "-> unique global minimum"
)

print("=" * 60)