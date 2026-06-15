"""
feature_analysis module - Correlation analysis and feature clustering tools.

Provides functions for computing correlation matrices, identifying feature
clusters via hierarchical clustering, plotting dendrograms and heatmaps,
and suggesting feature exclusions based on importance within clusters.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

LOGGER = logging.getLogger(__name__)


def compute_correlation_matrix(X, columns, method="spearman"):
    """
    Compute a correlation matrix for the given feature array.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).
    columns : list of str
        Feature names corresponding to columns of X.
    method : str
        Correlation method: 'spearman' or 'pearson'.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix with feature names as index/columns.
    """
    df = pd.DataFrame(X, columns=columns)

    # Drop constant columns (no variance)
    nunique = df.nunique(dropna=True)
    valid_cols = nunique[nunique > 1].index.tolist()
    df = df[valid_cols].dropna(axis=1, how="all")

    n_samples, n_features = df.shape

    if n_features == 0:
        raise ValueError(
            "No non-constant features remain after dropping constant columns. "
            "Cannot compute a correlation matrix."
        )
    if n_features == 1:
        LOGGER.info("Only 1 non-constant feature; returning trivial 1x1 correlation matrix.")
        return pd.DataFrame(
            [[1.0]], index=df.columns.tolist(), columns=df.columns.tolist()
        )

    LOGGER.info(
        f"Computing {method} correlation matrix for {n_features} features "
        f"across {n_samples:,} samples... (this may take a few minutes for large grids)"
    )

    if method == "spearman":
        corr = spearmanr(df).correlation
        # spearmanr returns a scalar when exactly 2 features
        if np.ndim(corr) == 0:
            corr = np.array([[1.0, corr], [corr, 1.0]])
    elif method == "pearson":
        corr = df.corr(method="pearson").values
    else:
        raise ValueError(f"Unknown correlation method: {method}. Use 'spearman' or 'pearson'.")

    LOGGER.info("Correlation matrix computed. Post-processing...")

    # Handle NaN and ensure symmetry
    corr = np.nan_to_num(corr, nan=0.0)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    return pd.DataFrame(corr, index=df.columns.tolist(), columns=df.columns.tolist())


def compute_feature_clusters(corr_matrix, threshold=0.7):
    """
    Identify clusters of correlated features using hierarchical clustering.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix (output of compute_correlation_matrix).
    threshold : float
        Distance threshold for forming clusters. Lower values create more
        clusters (features must be more similar to cluster together).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'clusters': dict mapping cluster IDs to lists of feature names
        - 'linkage': ndarray, the Ward linkage matrix
        - 'dendrogram': dict, the dendrogram result from scipy
    """
    corr = corr_matrix.values
    features = corr_matrix.columns.tolist()

    LOGGER.info(f"Computing hierarchical clustering for {len(features)} features...")

    # Convert correlation to distance
    distance_matrix = 1 - np.abs(corr)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0.0)

    # Ward hierarchical clustering
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    # Form flat clusters
    cluster_labels = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")

    feature_clusters = {}
    for idx, cluster_id in enumerate(cluster_labels):
        cluster_key = f"cluster_{cluster_id}"
        if cluster_key not in feature_clusters:
            feature_clusters[cluster_key] = []
        feature_clusters[cluster_key].append(features[idx])

    LOGGER.info(f"Identified {len(feature_clusters)} feature clusters at threshold={threshold}")

    # Generate dendrogram data (without plotting)
    dendro = hierarchy.dendrogram(dist_linkage, labels=features, no_plot=True)

    return {
        "clusters": feature_clusters,
        "linkage": dist_linkage,
        "dendrogram": dendro,
    }


def plot_dendrogram(corr_matrix, linkage, save_path):
    """
    Plot and save a hierarchical clustering dendrogram.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix (for feature names).
    linkage : np.ndarray
        Ward linkage matrix from compute_feature_clusters.
    save_path : str or Path
        Path to save the PNG output.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = corr_matrix.columns.tolist()
    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(features) * 0.4), 8))

    hierarchy.dendrogram(
        linkage,
        labels=features,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=max(6, min(10, 200 // len(features))),
    )
    ax.set_title("Feature Clustering Dendrogram (Ward Linkage)", fontsize=14, pad=20)
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Distance (1 - |correlation|)", fontsize=12)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Dendrogram saved to {save_path}")


def plot_correlation_heatmap(corr_matrix, dendro_result, save_path):
    """
    Plot and save a clustered correlation heatmap.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix.
    dendro_result : dict
        Dendrogram result from compute_feature_clusters (contains 'leaves' order).
    save_path : str or Path
        Path to save the PNG output.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corr = corr_matrix.values
    leaves = dendro_result["leaves"]
    leaf_labels = [dendro_result["ivl"][i] for i in range(len(dendro_result["ivl"]))]

    # Reorder correlation matrix by dendrogram leaf order
    ordered_corr = corr[leaves, :][:, leaves]

    n = len(leaf_labels)
    fig, ax = plt.subplots(1, 1, figsize=(max(12, n * 0.35), max(10, n * 0.3)))

    im = ax.imshow(ordered_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(leaf_labels, rotation=90, fontsize=max(6, min(10, 200 // n)))
    ax.set_yticklabels(leaf_labels, fontsize=max(6, min(10, 200 // n)))
    ax.set_title("Feature Correlation Matrix (Clustered)", fontsize=14, pad=20)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=270, labelpad=20, fontsize=12)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Correlation heatmap saved to {save_path}")


def suggest_exclusions(clusters, corr_matrix):
    """
    For each cluster with more than one feature, suggest dropping the most
    redundant feature (highest average absolute correlation to other members).

    Parameters
    ----------
    clusters : dict
        Feature clusters mapping cluster_id -> list of feature names.
    corr_matrix : pd.DataFrame
        Correlation matrix with feature names as index/columns.

    Returns
    -------
    dict
        Dictionary mapping cluster IDs to suggestion dicts with keys:
        ``drop`` (feature_name), ``keep`` (list of other features),
        ``drop_avg_correlation`` (float), ``keep_avg_correlations`` (list of floats).
    """
    suggestions = {}

    for cluster_id, features in clusters.items():
        if len(features) <= 1:
            continue

        avg_corrs = []
        for f in features:
            others = [o for o in features if o != f]
            if f in corr_matrix.index and all(o in corr_matrix.columns for o in others):
                avg_corr = float(corr_matrix.loc[f, others].abs().mean())
            else:
                avg_corr = 0.0
            avg_corrs.append(avg_corr)

        max_idx = int(np.argmax(avg_corrs))
        drop_feature = features[max_idx]
        keep_features = [f for i, f in enumerate(features) if i != max_idx]

        suggestions[cluster_id] = {
            "drop": drop_feature,
            "keep": keep_features,
            "drop_avg_correlation": avg_corrs[max_idx],
            "keep_avg_correlations": [
                avg_corrs[i] for i in range(len(features)) if i != max_idx
            ],
        }

    LOGGER.info(
        f"Generated exclusion suggestions for {len(suggestions)} multi-feature clusters"
    )
    return suggestions


def save_analysis_outputs(corr_matrix, cluster_result, out_dir):
    """
    Save all analysis artifacts to the output directory.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix.
    cluster_result : dict
        Output from compute_feature_clusters.
    out_dir : str or Path
        Output directory.

    Returns
    -------
    dict
        Paths to the saved files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    n_features = len(corr_matrix.columns)
    LOGGER.info(f"Saving analysis outputs ({n_features} features) to {out_dir}...")

    # Save correlation matrix CSV
    csv_path = out_dir / "correlation_matrix.csv"
    corr_matrix.to_csv(csv_path)
    paths["correlation_matrix"] = str(csv_path)
    LOGGER.info(f"  [1/4] Correlation matrix CSV saved to {csv_path}")

    # Save feature clusters JSON
    clusters_path = out_dir / "feature_clusters.json"
    with open(clusters_path, "w") as f:
        json.dump(cluster_result["clusters"], f, indent=2)
    paths["feature_clusters"] = str(clusters_path)
    LOGGER.info(f"  [2/4] Feature clusters JSON saved to {clusters_path}")

    # Save dendrogram PNG
    dendro_path = out_dir / "dendrogram.png"
    LOGGER.info("  [3/4] Rendering dendrogram...")
    plot_dendrogram(corr_matrix, cluster_result["linkage"], dendro_path)
    paths["dendrogram"] = str(dendro_path)

    # Save heatmap PNG
    heatmap_path = out_dir / "correlation_heatmap.png"
    LOGGER.info("  [4/4] Rendering correlation heatmap...")
    plot_correlation_heatmap(corr_matrix, cluster_result["dendrogram"], heatmap_path)
    paths["correlation_heatmap"] = str(heatmap_path)

    return paths
