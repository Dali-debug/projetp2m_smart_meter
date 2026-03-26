"""
check_preprocessing.py — Validation script for the preprocessing pipeline.

This script loads a house CSV (or generates synthetic data when the real
dataset is not available), applies the Hampel filter and the four
interpolation methods, and prints a summary table of the results so that
the user can quickly check that the preprocessing functions work correctly.

Usage
-----
    python check_preprocessing.py [path_to_csv]

If no argument is given the script will look for
  ../../Processed_Data_CSV/House_1.csv
relative to its own directory.
"""

import os
import sys
import numpy as np
import pandas as pd

# Make sure the current directory is on the path so that `preprocessing`
# can always be imported whether the script is run from the project root or
# from inside Preprocessing/Algorithms/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import hampel_filter, compare_interpolation_methods


def load_data(csv_path=None):
    """Load the aggregate power column from a REFIT-formatted CSV.

    Falls back to synthetic data when the file is not found.

    Parameters
    ----------
    csv_path : str or None
        Path to a REFIT CSV file.  If None, a default path is tried.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with at least an 'Aggregate' column and a DatetimeIndex.
    """
    if csv_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.normpath(
            os.path.join(script_dir, "..", "..", "Processed_Data_CSV", "House_1.csv")
        )

    if os.path.exists(csv_path):
        print(f"[check] Loading data from: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["Time"], index_col="Time")
        print(f"[check] Loaded {len(df):,} rows, columns: {list(df.columns)}")
    else:
        print(f"[check] File not found — using synthetic data: {csv_path}")
        rng = np.random.default_rng(42)
        n = 96 * 30  # 30 days @ 8-second sampling
        t = pd.date_range("2013-01-01", periods=n, freq="8s")
        values = 300 + 50 * np.sin(np.linspace(0, 6 * np.pi, n))
        values += rng.normal(0, 20, n)
        # Inject synthetic outliers
        outlier_pos = rng.choice(n, size=40, replace=False)
        values[outlier_pos] += rng.choice([-1, 1], size=40) * rng.uniform(400, 900, 40)
        df = pd.DataFrame({"Aggregate": values}, index=t)

    return df


def check_hampel(series):
    """Run the Hampel filter and print a brief summary.

    Parameters
    ----------
    series : pd.Series
        Raw aggregate power signal.

    Returns
    -------
    filtered : pd.Series
        Cleaned signal.
    outlier_mask : pd.Series (bool)
        Boolean mask of detected outliers.
    """
    print("\n" + "=" * 60)
    print("CHECK 1 — Hampel filter")
    print("=" * 60)

    filtered, outlier_mask = hampel_filter(series, window_size=15)

    n_outliers = int(outlier_mask.sum())
    pct = 100.0 * n_outliers / len(series)
    delta_mean = series.mean() - filtered.mean()
    delta_std  = series.std()  - filtered.std()

    print(f"  Input  samples  : {len(series):,}")
    print(f"  Outliers found  : {n_outliers:,}  ({pct:.3f} %)")
    print(f"  Mean shift      : {delta_mean:+.4f} W")
    print(f"  Std  shift      : {delta_std:+.4f} W")

    # Sanity checks
    assert len(filtered) == len(series), "Output length mismatch!"
    assert len(outlier_mask) == len(series), "Mask length mismatch!"
    assert not filtered.isna().any(), "Filtered series contains NaN!"

    print("  [OK] All sanity checks passed.")
    return filtered, outlier_mask


def check_interpolation(series):
    """Run the interpolation comparison and print the MAE table.

    Parameters
    ----------
    series : pd.Series
        Clean power signal (ideally after Hampel filtering).

    Returns
    -------
    mae_results : dict
        Mapping method → MAE value.
    """
    print("\n" + "=" * 60)
    print("CHECK 2 — Interpolation methods")
    print("=" * 60)

    segment = series.iloc[:200].reset_index(drop=True)
    mae_results = compare_interpolation_methods(segment)

    # Verify that every method produced a numeric MAE
    for method_name, mae in mae_results.items():
        assert not np.isnan(mae), f"MAE for '{method_name}' is NaN!"
    print("  [OK] All interpolation methods returned valid MAE values.")

    return mae_results


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_data(csv_path)

    excerpt = df["Aggregate"].iloc[: 96 * 30]

    filtered, outlier_mask = check_hampel(excerpt)
    mae_results = check_interpolation(filtered)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Hampel outliers detected : {int(outlier_mask.sum()):,}")
    print("  Interpolation MAE (W):")
    for name, mae in sorted(mae_results.items(), key=lambda x: x[1]):
        print(f"    {name:<12} {mae:.4f}")
    print("\n[check_preprocessing] All checks passed successfully.")


if __name__ == "__main__":
    main()
