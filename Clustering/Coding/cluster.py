"""
cluster.py — K-Means and MeanShift clustering helpers for NILM appliance states.

Provides:
  - cluster          : select the optimal number of K-Means clusters by
                       maximising the silhouette score.
  - _transform_data  : sub-sampling and OFF-state filtering.
  - _apply_clustering: core K-Means routine.
  - hart85_means_shift_cluster : MeanShift clustering on Hart-style transition pairs.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score


# ============================================================
# Public API
# ============================================================

def cluster(X, max_num_clusters=3, exact_num_clusters=None):
    """
    Cluster a 1-D power series and return sorted cluster centroids.

    Parameters
    ----------
    X : pd.Series or array-like
        Power values (W).  Values ≤ 10 W are treated as the OFF state and
        excluded before clustering.
    max_num_clusters : int
        Maximum number of clusters to consider (not counting the OFF state).
    exact_num_clusters : int or None
        If given, skip silhouette scoring and use exactly this many clusters.

    Returns
    -------
    centroids : np.ndarray of int32
        Sorted array of cluster centroids including the implicit OFF state (0 W).
    """
    data = _transform_data(X)

    if len(data) == 0:
        return np.array([0], dtype=np.int32)

    centers = _apply_clustering(data, max_num_clusters, exact_num_clusters)
    # Prepend the OFF state centroid (0 W)
    centroids = np.sort(np.append(centers, 0)).astype(np.int32)
    return centroids


def hart85_means_shift_cluster(pair_buffer_df, columns):
    """
    Apply MeanShift clustering to a DataFrame of paired Hart-style transitions.

    Parameters
    ----------
    pair_buffer_df : pd.DataFrame
        Each row is a paired ON/OFF transition.  Expected columns depend on
        what was measured:
          - 'active'   : active power change (W)  — always present
          - 'reactive' : reactive power change (VAr) — optional
          - 'apparent' : apparent power change (VA)  — optional
    columns : list of str
        Subset of ['active', 'reactive', 'apparent'] to use as features.

    Returns
    -------
    cluster_centers : pd.DataFrame
        One row per cluster centroid, same columns as *columns*.
    """
    features = []
    for col in columns:
        if col in pair_buffer_df.columns:
            features.append(pair_buffer_df[col].values.reshape(-1, 1))

    if not features:
        return pd.DataFrame(columns=columns)

    X = np.hstack(features)

    ms = MeanShift()
    ms.fit(X)
    centers = ms.cluster_centers_

    return pd.DataFrame(centers, columns=[c for c in columns
                                          if c in pair_buffer_df.columns])


# ============================================================
# Internal helpers
# ============================================================

def _transform_data(X):
    """
    Prepare data for clustering: flatten, remove OFF-state values, sub-sample.

    Parameters
    ----------
    X : pd.Series or array-like

    Returns
    -------
    data : np.ndarray, shape (n,)
    """
    if isinstance(X, pd.Series):
        x = X.dropna().values.flatten()
    else:
        x = np.asarray(X).flatten()

    x = x[x > 10]  # discard OFF state

    if len(x) > 2000:
        rng = np.random.default_rng(0)
        x = rng.choice(x, size=2000, replace=False)

    return x


def _apply_clustering(X, max_num_clusters=3, exact_num_clusters=None):
    """
    Core K-Means routine.

    Parameters
    ----------
    X : np.ndarray, shape (n,)
        Pre-processed power values (OFF state already removed).
    max_num_clusters : int
    exact_num_clusters : int or None

    Returns
    -------
    centers : np.ndarray, shape (k,)
        Cluster centroids (excluding the OFF state).
    """
    X_2d = X.reshape(-1, 1)

    if exact_num_clusters is not None:
        k = max(1, int(exact_num_clusters))
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_2d)
        return km.cluster_centers_.flatten()

    # ---- silhouette-based k selection ----
    best_k, best_score, best_centers = 1, -1.0, None

    for k in range(1, max_num_clusters):
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_2d)
        if k == 1:
            if best_centers is None:
                best_centers = km.cluster_centers_.flatten()
            continue
        score = silhouette_score(X_2d, km.labels_)
        if score > best_score:
            best_score = score
            best_k = k
            best_centers = km.cluster_centers_.flatten()

    if best_centers is None:
        km = KMeans(n_clusters=1, n_init=10, random_state=0).fit(X_2d)
        best_centers = km.cluster_centers_.flatten()

    return best_centers


# ============================================================
# Standalone demo
# ============================================================

if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "Processed_Data_CSV")
    csv_path = os.path.join(data_dir, "House_1.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["Time"], index_col="Time")
        col = next((c for c in df.columns if "Refrigerator" in c), None)
        col = col or df.columns[1]  # fallback: first appliance column
        series = df[col].iloc[:10000]
    else:
        rng = np.random.default_rng(5)
        # Simulate refrigerator: mostly OFF (0-5 W) with compressor cycles (~150 W)
        series = pd.Series(
            np.concatenate([
                rng.normal(2, 2, 7000),
                rng.normal(150, 10, 3000),
            ])
        )

    centroids = cluster(series, max_num_clusters=3)
    print(f"[cluster] Centroids (incl. OFF=0): {centroids}")
