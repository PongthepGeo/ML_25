import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib as mpl
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

def high_contrast_colors(n: int):
    """Return n distinct, high-contrast, colorblind-friendly colors."""
    bases = ['tab10', 'Set1', 'Dark2', 'tab20']
    out, seen = [], set()
    for name in bases:
        cmap = mpl.colormaps.get_cmap(name)
        rgba_list = cmap.colors if hasattr(cmap, 'colors') else [cmap(i / 255) for i in range(256)]
        for rgba in rgba_list:
            col = mpl.colors.to_hex(rgba)
            if col not in seen:
                out.append(col); seen.add(col)
            if len(out) >= n:
                return out[:n]
    # Fallback: evenly spaced hues
    remain = n - len(out)
    if remain > 0:
        hsv = mpl.colormaps.get_cmap('hsv')
        for i in range(remain):
            col = mpl.colors.to_hex(hsv(i / max(1, remain)))
            if col not in seen:
                out.append(col); seen.add(col)
    return out[:n]

def compute_kmeans(x, y, n_clusters, random_state=42, reorder_by='GR'):
    """
    K-Means on 2-D features [x (GR), y (ILD_log10)].
    Returns:
        labels (np.ndarray): cluster id per point (possibly remapped for stability)
        centers (np.ndarray [k,2]): centroids in original units (GR, ILD_log10)
    """
    X = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=32, random_state=random_state)
    labels = km.fit_predict(Xz)
    centers = scaler.inverse_transform(km.cluster_centers_)

    if reorder_by is not None:
        key = 0 if reorder_by.upper() == 'GR' else 1
        order = np.argsort(centers[:, key])
        remap = {old: new for new, old in enumerate(order)}
        labels = np.vectorize(remap.get)(labels)
        centers = centers[order]
    return labels, centers

def build_code_to_name(data_facies, lithofacies_list):
    """
    Map facies codes in CSV to provided names; if already strings, identity map.
    Map clusters -> facies names using majority vote vs. CSV 'Facies
    """
    series = pd.Series(data_facies)
    if series.dropna().map(lambda v: isinstance(v, str)).all():
        # Already string labels (e.g., 'SS'); identity map
        unique_names = sorted(series.dropna().unique().tolist())
        return {name: name for name in unique_names}, unique_names
    # Numeric codes -> provided names (best-effort)
    codes = sorted(series.dropna().unique().tolist())
    if len(codes) == len(lithofacies_list):
        return {code: lithofacies_list[i] for i, code in enumerate(codes)}, codes
    return {code: str(code) for code in codes}, codes

def map_clusters_to_facies(cluster_labels, true_facies, lithofacies_list, litho_colors):
    """
    Returns:
      cluster_to_name: dict {cluster_id -> facies_name}
      name_to_color:  dict {facies_name -> hex color}
    One-to-one greedy assignment from clusters to facies by majority counts.
    """
    cl = pd.Series(cluster_labels, name='cluster').reset_index(drop=True)
    tf = pd.Series(true_facies, name='facies').reset_index(drop=True)

    code_to_name, _ = build_code_to_name(tf, lithofacies_list)
    tf_names = tf.map(lambda v: code_to_name.get(v, v))  # normalize to names

    # contingency table
    ctab = pd.crosstab(cl, tf_names)

    # Ensure all clusters and all facies names appear (fill missing with 0)
    clusters_all = sorted(np.unique(cluster_labels).tolist())
    names_all = sorted(pd.unique(tf_names.dropna()).tolist())
    ctab = ctab.reindex(index=clusters_all, columns=names_all, fill_value=0)

    remaining_clusters = list(ctab.index)
    remaining_names = list(ctab.columns)
    cluster_to_name = {}

    # Greedy max assignment
    while remaining_clusters and remaining_names:
        best = (-1, None, None)  # (count, cluster, name)
        for cid in remaining_clusters:
            row = ctab.loc[cid, list(remaining_names)]  # <-- list (not set)
            if row.size == 0:
                continue
            top_idx = int(np.argmax(row.values))
            top_name = remaining_names[top_idx]
            cnt = int(row.iloc[top_idx])
            if cnt > best[0]:
                best = (cnt, cid, top_name)
        _, cid, name = best
        if cid is None or name is None:
            break
        cluster_to_name[cid] = name
        remaining_clusters.remove(cid)
        remaining_names.remove(name)

    # Assign any leftovers to remaining names or generic labels
    for cid in remaining_clusters:
        name = remaining_names.pop(0) if remaining_names else f'Cluster {cid+1}'
        cluster_to_name[cid] = name

    # Colors per facies name
    if litho_colors and len(litho_colors) >= len(lithofacies_list):
        name_to_color = {lithofacies_list[i]: litho_colors[i] for i in range(len(lithofacies_list))}
    else:
        distinct_names = sorted(set(cluster_to_name.values()))
        palette = high_contrast_colors(len(distinct_names))
        name_to_color = {nm: palette[i] for i, nm in enumerate(distinct_names)}

    return cluster_to_name, name_to_color

def scatter_plot(x, y, labels, centers, cluster_to_name, name_to_color,
                 title=None, point_size=28, edge_width=0.6, centroid_size=200):
    fig, ax = plt.subplots(figsize=(8, 6.8), dpi=120)
    used_labels = set()

    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        idx = (labels == lbl)
        facies_name = cluster_to_name.get(int(lbl), f'Cluster {int(lbl)+1}')
        face = name_to_color.get(facies_name, '#808080')
        legend_label = facies_name if facies_name not in used_labels else '_nolegend_'
        used_labels.add(facies_name)

        ax.scatter(
            x[idx], y[idx],
            s=point_size,
            c=face,
            alpha=0.90,
            edgecolors='black',
            linewidths=edge_width,
            marker='o',
            label=legend_label,
            zorder=2
        )

    # Plot centroids with bold black outline
    ax.scatter(
        centers[:, 0], centers[:, 1],
        marker='X',
        s=centroid_size,
        facecolors='white',
        edgecolors='black',
        linewidths=1.6,
        label='Centroids',
        zorder=4
    )

    ax.set_xlabel('GR')
    ax.set_ylabel('ILD_log10')
    ax.set_title(title or 'Well Log Clusters (Facies)')
    ax.grid(alpha=0.25, zorder=0)

    ax.legend(ncol=2, fontsize=9, frameon=True, loc='best')
    plt.tight_layout()
    os.makedirs('figure_plot', exist_ok=True)
    plt.savefig('figure_plot/cluster.png', format='png', dpi=600, bbox_inches='tight', transparent=True, pad_inches=0)
    print("Figure saved to 'figure_plot/cluster.png'")
    plt.show()

def compute_metrics(y_true, y_pred):
    """Return R2, RMSE, MAE; handle short arrays gracefully."""
    n = len(y_true)
    if n < 2:
        return np.nan, float(np.sqrt(mean_squared_error(y_true, y_pred))), float(mean_absolute_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return float(r2), rmse, mae

def fit_lr(x_vec, y_vec):
    """Fit sklearn LinearRegression; return (slope, intercept, predict_fn)."""
    model = LinearRegression()
    X = x_vec.reshape(-1, 1)
    model.fit(X, y_vec)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    def predict_fn(x_new):
        return model.predict(np.asarray(x_new).reshape(-1, 1))
    return slope, intercept, predict_fn

def plot_scatter_with_regression(x, y, facies_names, name_to_color,
                                 per_facies_models, global_model,
                                 title='Linear Regression per Facies + Global'):
    fig, ax = plt.subplots(figsize=(8.6, 6.6), dpi=120)

    # Scatter by facies with black edges
    plotted = set()
    present_names = sorted(pd.unique(facies_names))
    for fac in present_names:
        mask = (facies_names == fac).to_numpy()
        if not mask.any():
            continue
        ax.scatter(
            x[mask], y[mask],
            s=28,
            c=name_to_color.get(fac, '#808080'),
            edgecolors='black',
            linewidths=0.6,
            alpha=0.9,
            marker='o',
            label=fac if fac not in plotted else '_nolegend_',
            zorder=2
        )
        plotted.add(fac)

    # Per-facies regression lines
    for fac, (slope, intercept, predict) in per_facies_models.items():
        mask = (facies_names == fac).to_numpy()
        if not mask.any():
            continue
        x_min, x_max = float(np.min(x[mask])), float(np.max(x[mask]))
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line,
            linewidth=2.0,
            color=name_to_color.get(fac, '#808080'),
            alpha=0.95,
            label=f'{fac} fit',
            zorder=3
        )

    # Global regression line across all data
    g_slope, g_intercept, _ = global_model
    xg = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    yg = g_slope * xg + g_intercept
    ax.plot(
        xg, yg,
        linewidth=2.6,
        color='black',
        linestyle='--',
        label='Global fit',
        zorder=4
    )

    ax.set_xlabel('GR')
    ax.set_ylabel('ILD_log10')
    ax.set_title(title)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(ncol=2, fontsize=9, frameon=True, loc='best')
    plt.tight_layout()
    os.makedirs('figure_plot', exist_ok=True)
    plt.savefig('figure_plot/linear.png', format='png', dpi=600, bbox_inches='tight', transparent=True, pad_inches=0)
    print("Figure saved to 'figure_plot/linear.png'")
    plt.show()
