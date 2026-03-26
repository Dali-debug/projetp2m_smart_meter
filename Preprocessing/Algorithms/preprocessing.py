"""
Preprocessing module for NILM pipeline.

Contains:
  - hampel_filter : outlier detection and correction using the Hampel identifier
  - compare_interpolation_methods : comparison of pandas interpolation strategies
    for handling missing data in power-consumption signals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error


# ============================================================
# 1.  Hampel filter — outlier detection & correction
# ============================================================

def hampel_filter(series, window_size, n_sigmas=3):
    """
    Apply the Hampel identifier to remove outliers from a 1-D time series.

    Parameters
    ----------
    series : pd.Series
        Input signal (e.g. 'Aggregate' power in Watts).
    window_size : int
        Half-width of the sliding window (full window = 2 * window_size + 1).
    n_sigmas : float, optional
        Number of MAD-scaled standard deviations used as the outlier threshold.
        Default is 3.

    Returns
    -------
    filtered : pd.Series
        Signal with outliers replaced by the local median.
    outlier_mask : pd.Series (bool)
        True at positions that were identified as outliers.

    Notes
    -----
    The MAD is multiplied by k = 1.4826 (consistency factor for a Gaussian
    distribution) so that  k * MAD  is a consistent estimator of σ.
    """
    k = 1.4826  # consistency factor for Gaussian distribution
    n = len(series)
    filtered = series.copy().astype(float)
    outlier_mask = pd.Series(False, index=series.index)

    for i in range(n):
        lo = max(0, i - window_size)
        hi = min(n - 1, i + window_size)
        window = series.iloc[lo: hi + 1]
        median = window.median()
        mad = k * (window - median).abs().median()

        if mad == 0:
            continue

        if abs(series.iloc[i] - median) > n_sigmas * mad:
            filtered.iloc[i] = median
            outlier_mask.iloc[i] = True

    return filtered, outlier_mask


def plot_hampel_result(original, filtered, outlier_mask, title="Hampel Filter — Outlier Correction"):
    """
    Visualise the original vs filtered signal with outliers highlighted.

    A zoomed inset is added automatically over the densest region of outliers
    to give a closer view of the correction.

    Parameters
    ----------
    original : pd.Series
        Raw signal before filtering.
    filtered : pd.Series
        Signal after the Hampel filter has been applied.
    outlier_mask : pd.Series (bool)
        Mask produced by :func:`hampel_filter`.
    title : str, optional
        Figure title.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(original.values, color="steelblue", linewidth=0.8, label="Original")
    ax.plot(filtered.values, color="green", linewidth=0.8, label="Filtered")
    ax.scatter(
        np.where(outlier_mask)[0],
        original.values[outlier_mask],
        color="red", s=20, zorder=5, label="Outliers"
    )
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power (W)")
    ax.legend()

    # ---- zoomed inset (centred on the first detected outlier cluster) --------
    outlier_indices = np.where(outlier_mask)[0]
    if len(outlier_indices) > 0:
        centre = int(np.median(outlier_indices))
        lo, hi = max(0, centre - 50), min(len(original) - 1, centre + 50)
        ax_inset = ax.inset_axes([0.65, 0.55, 0.32, 0.35])
        ax_inset.plot(range(lo, hi + 1), original.values[lo: hi + 1],
                      color="steelblue", linewidth=0.8)
        ax_inset.plot(range(lo, hi + 1), filtered.values[lo: hi + 1],
                      color="green", linewidth=0.8)
        cluster_mask = outlier_mask.iloc[lo: hi + 1].values
        ax_inset.scatter(
            np.where(cluster_mask)[0] + lo,
            original.values[lo: hi + 1][cluster_mask],
            color="red", s=30, zorder=5
        )
        ax_inset.set_title("Zoom", fontsize=8)
        ax_inset.tick_params(labelsize=7)
        ax.indicate_inset_zoom(ax_inset, edgecolor="black")

    plt.tight_layout()
    plt.show()


# ============================================================
# 2.  Interpolation comparison
# ============================================================

def compare_interpolation_methods(series, missing_start=18, missing_end=22,
                                   random_mask_frac=0.02, seed=42):
    """
    Compare four pandas interpolation methods on a power-consumption signal.

    Methods evaluated:
      - linear
      - polynomial order 3
      - polynomial order 5
      - spline (cubic, order 3)

    Parameters
    ----------
    series : pd.Series
        Input power signal (e.g. first 200 samples of 'Aggregate').
    missing_start : int
        Index of the first missing sample for the visual gap test.
    missing_end : int
        Index of the last missing sample (inclusive) for the visual gap test.
    random_mask_frac : float
        Fraction of samples randomly set to NaN for the MAE evaluation.
    seed : int
        Random seed for reproducibility of the random mask.

    Returns
    -------
    mae_results : dict
        Dictionary mapping method name → MAE on the randomly masked segment.
    """
    methods = [
        ("linear", dict(method="linear")),
        ("poly3",  dict(method="polynomial", order=3)),
        ("poly5",  dict(method="polynomial", order=5)),
        ("spline3", dict(method="spline", order=3)),
    ]

    # --- visual comparison on a fixed gap --------------------------------
    s_gap = series.copy().astype(float)
    s_gap.iloc[missing_start: missing_end + 1] = np.nan
    # Add zero-fill baseline
    s_zero = s_gap.copy().fillna(0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.values, color="black", linewidth=1.2, label="Original", zorder=5)
    ax.plot(s_zero.values, color="grey", linewidth=0.8,
            linestyle="--", label="Zero-fill", alpha=0.7)

    colors = ["royalblue", "tomato", "darkorange", "limegreen"]
    for (name, kwargs), color in zip(methods, colors):
        s_interp = s_gap.copy().interpolate(**kwargs)
        ax.plot(s_interp.values, linewidth=1.0, color=color,
                label=name, alpha=0.85)

    ax.axvspan(missing_start - 0.5, missing_end + 0.5, alpha=0.15,
               color="yellow", label="Missing gap")
    ax.set_title("Interpolation methods — visual comparison on fixed gap")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power (W)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- MAE evaluation on random missing samples ------------------------
    rng = np.random.default_rng(seed)
    n_missing = max(1, int(len(series) * random_mask_frac))
    missing_idx = rng.choice(len(series), size=n_missing, replace=False)

    mae_results = {}

    # Baseline: zero fill
    s_rand = series.copy().astype(float)
    s_rand.iloc[missing_idx] = np.nan
    s_zero_rand = s_rand.fillna(0)
    mae_results["zero_fill"] = mean_absolute_error(
        series.iloc[missing_idx], s_zero_rand.iloc[missing_idx]
    )

    for name, kwargs in methods:
        s_rand = series.copy().astype(float)
        s_rand.iloc[missing_idx] = np.nan
        try:
            s_interp = s_rand.interpolate(**kwargs)
            mae_results[name] = mean_absolute_error(
                series.iloc[missing_idx], s_interp.iloc[missing_idx]
            )
        except Exception:
            mae_results[name] = float("nan")

    print("\n=== Interpolation MAE results (lower is better) ===")
    for method_name, mae in sorted(mae_results.items(), key=lambda x: x[1]):
        print(f"  {method_name:<12} MAE = {mae:.4f} W")

    return mae_results


# ============================================================
# 3.  Demo / standalone entry point
# ============================================================

if __name__ == "__main__":
    import os

    # Locate data relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "Processed_Data_CSV")
    csv_path = os.path.join(data_dir, "House_1.csv")

    if not os.path.exists(csv_path):
        print(f"[INFO] Data file not found: {csv_path}")
        print("[INFO] Generating synthetic data for demonstration …")
        rng = np.random.default_rng(0)
        t = pd.date_range("2013-01-01", periods=96 * 30, freq="8s")
        values = 300 + 50 * np.sin(np.linspace(0, 6 * np.pi, len(t)))
        values += rng.normal(0, 20, len(t))
        # inject synthetic outliers
        outlier_pos = rng.choice(len(t), size=30, replace=False)
        values[outlier_pos] += rng.choice([-1, 1], size=30) * rng.uniform(500, 1000, 30)
        df = pd.DataFrame({"Time": t, "Aggregate": values})
        df.set_index("Time", inplace=True)
    else:
        df = pd.read_csv(csv_path, parse_dates=["Time"], index_col="Time")

    # ---- Hampel filter demo ----
    excerpt = df["Aggregate"].iloc[: 96 * 30]
    filtered, outlier_mask = hampel_filter(excerpt, window_size=15)
    print(f"[Hampel] Outliers detected: {outlier_mask.sum()} / {len(excerpt)}")
    plot_hampel_result(excerpt, filtered, outlier_mask)

    # ---- Interpolation demo ----
    segment = filtered.iloc[:200].reset_index(drop=True)
    compare_interpolation_methods(segment)
