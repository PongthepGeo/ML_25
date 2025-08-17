import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os

params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 100,
	'axes.labelsize':12,  
	'axes.titlesize':12,
	'axes.titleweight': 'bold',
	'legend.fontsize': 10,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'font.family': 'serif',
	'font.serif': 'Times New Roman'
}
matplotlib.rcParams.update(params)

def plot_regression(X, y, y_pred):
    plt.figure(figsize=(6, 10))
    sns.scatterplot(x=y, y=X.flatten(), label="GR Data", color="red", 
                    s=40, alpha=0.7, edgecolor='black', marker='o')
    sns.lineplot(x=y_pred, y=X.flatten(), label="Linear Fit", color="blue")

    plt.gca().invert_yaxis()  # Depth increases downward
    plt.xlabel("Gamma Ray (GR)")
    plt.ylabel("Depth")
    plt.title("Linear Regression of GR vs Depth\n(Well: NEWBY, Facies: 2)")
    plt.legend()
    plt.tight_layout()
    os.makedirs('figure_plot', exist_ok=True)
    plt.savefig('figure_plot/regression.png', format='png', dpi=300, bbox_inches='tight',
                transparent=True)
    print("Figure saved as 'figure_plot/regression.png'")
    plt.show()

def plot_classification(df_f2, df_f4):
    plt.figure(figsize=(6, 10))
    # Enhanced scatter plots with better styling
    sns.scatterplot(x="GR", y="Depth", data=df_f2, label="Facies 2", color="blue", 
                   s=40, alpha=0.7, edgecolor='black', marker='o')
    sns.scatterplot(x="GR", y="Depth", data=df_f4, label="Facies 4", color="red", 
                   s=40, alpha=0.7, edgecolor='black', marker='o')
    plt.gca().invert_yaxis()
    plt.xlabel("Gamma Ray (GR)")
    plt.ylabel("Depth")
    plt.title("GR vs Depth: Facies 2 vs Facies 4 (Well NEWBY)")
    plt.legend()
    plt.tight_layout()
    os.makedirs('figure_plot', exist_ok=True)
    plt.savefig('figure_plot/classification.svg', format='svg', dpi=300, bbox_inches='tight',
                transparent=True)
    print("Figure saved as 'figure_plot/classification.svg'")
    plt.show()
