"""
binary_segmentation.py — Change-point detection via binary segmentation
                          (ruptures library) with K-Means ON/OFF labelling.

Pipeline
--------
1. Detect change-points in the power signal using ruptures.Binseg (L2 cost).
2. Extract a feature (mean power) for each segment.
3. Cluster segments into ON and OFF states with K-Means (k=2).
4. Pair consecutive ON → OFF transitions to build a list of complete cycles.
5. Store cycle data in a DataFrame and produce three plots.

This script is designed to be run stand-alone or imported as a module.
When run stand-alone it demonstrates the pipeline on either a real REFIT CSV
or synthetic data if the real file is not found.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except ImportError:
    _HAS_RUPTURES = False
    print("[binary_segmentation] WARNING: 'ruptures' package not found. "
          "Install it with:  pip install ruptures")


def detect_breakpoints(signal, pen_factor=1.0):
    """
    Detect change-points in a 1-D signal using binary segmentation (L2 cost).

    Parameters
    ----------
    signal : array-like
        1-D power signal (W).
    pen_factor : float
        Scaling factor for the BIC-like penalty:
          penalty = pen_factor * log(n) * dim * sigma²

    Returns
    -------
    breakpoints : list of int
        Sorted list of detected break-point indices (in samples).
        Excludes the trailing n (end-of-signal marker).

    Raises
    ------
    ImportError
        If the *ruptures* package is not installed.
    """
    if not _HAS_RUPTURES:
        raise ImportError("The 'ruptures' package is required. "
                          "Install it with:  pip install ruptures")

    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if signal.ndim == 1:
        signal_2d = signal.reshape(-1, 1)
    else:
        signal_2d = signal

    dim = signal_2d.shape[1]
    sigma2 = np.var(signal)

    penalty = pen_factor * np.log(n) * dim * sigma2
    algo = rpt.Binseg(model="l2").fit(signal_2d)
    raw_bkps = algo.predict(pen=penalty)

    # ruptures includes n as the last breakpoint — remove it
    breakpoints = [bp for bp in raw_bkps if bp < n]
    return breakpoints


def extract_segment_features(signal, breakpoints):
    """
    Extract mean power for each segment defined by the breakpoints.

    Parameters
    ----------
    signal : array-like
    breakpoints : list of int

    Returns
    -------
    features : pd.DataFrame
        Columns: ['start', 'end', 'mean_power']
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    edges = [0] + sorted(breakpoints) + [n]
    records = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        mean_pwr = float(signal[lo:hi].mean())
        records.append({"start": lo, "end": hi, "mean_power": mean_pwr})
    return pd.DataFrame(records)


def label_on_off(features_df):
    """
    Use K-Means (k=2) to label each segment as ON or OFF.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of :func:`extract_segment_features`.

    Returns
    -------
    labelled : pd.DataFrame
        Input DataFrame with an additional 'state' column ('ON' or 'OFF').
    """
    X = features_df[["mean_power"]].values
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    # The cluster with the higher centroid is 'ON'
    on_label = int(np.argmax(km.cluster_centers_.flatten()))
    labelled = features_df.copy()
    labelled["state"] = np.where(km.labels_ == on_label, "ON", "OFF")
    return labelled


def pair_on_off_cycles(labelled_df, appliance_name="Dishwasher"):
    """
    Pair consecutive ON and OFF segments into complete appliance cycles.

    Parameters
    ----------
    labelled_df : pd.DataFrame
        Output of :func:`label_on_off`.
    appliance_name : str
        Name used in the 'Appliance' column of the output.

    Returns
    -------
    cycles : pd.DataFrame
        Columns: [Appliance, ON_Time, OFF_Time, Duration, Power_Level]
    """
    records = []
    i = 0
    rows = labelled_df.to_dict("records")
    while i < len(rows):
        if rows[i]["state"] == "ON":
            on_start = rows[i]["start"]
            power_level = rows[i]["mean_power"]
            # Look for the following OFF segment
            j = i + 1
            while j < len(rows) and rows[j]["state"] == "ON":
                j += 1
            if j < len(rows) and rows[j]["state"] == "OFF":
                off_end = rows[j]["end"]
                duration = off_end - on_start
                records.append({
                    "Appliance":   appliance_name,
                    "ON_Time":     on_start,
                    "OFF_Time":    off_end,
                    "Duration":    duration,
                    "Power_Level": round(power_level, 2),
                })
                i = j + 1
            else:
                i += 1
        else:
            i += 1

    return pd.DataFrame(records)


def visualise_results(signal, breakpoints, labelled_df, cycles_df):
    """
    Produce three matplotlib figures:
      1. Time series with detected breakpoints.
      2. Scatter plot of segment mean powers coloured by ON/OFF cluster.
      3. Time series with paired cycle markers.

    Parameters
    ----------
    signal : array-like
    breakpoints : list of int
    labelled_df : pd.DataFrame
    cycles_df : pd.DataFrame
    """
    signal = np.asarray(signal, dtype=float)

    # ---- Figure 1: signal + breakpoints --------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal, color="steelblue", linewidth=0.6)
    for bp in breakpoints:
        ax.axvline(bp, color="orange", linewidth=0.8, alpha=0.7)
    ax.set_title(f"Binary segmentation — {len(breakpoints)} breakpoints detected")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power (W)")
    plt.tight_layout()
    plt.show()

    # ---- Figure 2: scatter plot of ON vs OFF clusters ------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    for state, color in [("ON", "tomato"), ("OFF", "steelblue")]:
        sub = labelled_df[labelled_df["state"] == state]
        ax.scatter(sub.index, sub["mean_power"], color=color,
                   label=state, s=40, alpha=0.8)
    ax.set_title("K-Means clustering — ON vs OFF segments")
    ax.set_xlabel("Segment index")
    ax.set_ylabel("Mean power (W)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ---- Figure 3: signal + cycle markers ------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal, color="steelblue", linewidth=0.6, label="Signal")
    for _, row in cycles_df.iterrows():
        ax.axvspan(row["ON_Time"], row["OFF_Time"],
                   alpha=0.25, color="green")
        ax.axvline(row["ON_Time"],  color="green", linewidth=1.0,
                   label="ON" if _ == 0 else "")
        ax.axvline(row["OFF_Time"], color="red",   linewidth=1.0,
                   label="OFF" if _ == 0 else "")
    ax.set_title(f"Detected cycles ({len(cycles_df)} complete cycles)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power (W)")
    if len(cycles_df):
        ax.legend()
    plt.tight_layout()
    plt.show()


def run_binary_segmentation(signal, appliance_name="Dishwasher",
                              pen_factor=1.0, plot=True):
    """
    Full binary-segmentation pipeline.

    Parameters
    ----------
    signal : array-like or pd.Series
        1-D power signal for one appliance.
    appliance_name : str
    pen_factor : float
        Passed to :func:`detect_breakpoints`.
    plot : bool
        Whether to display the three result figures.

    Returns
    -------
    cycles_df : pd.DataFrame
        DataFrame with columns
        [Appliance, ON_Time, OFF_Time, Duration, Power_Level].
    labelled_df : pd.DataFrame
        Segment-level labels.
    breakpoints : list of int
    """
    signal_arr = np.asarray(signal, dtype=float)
    breakpoints = detect_breakpoints(signal_arr, pen_factor=pen_factor)
    features_df = extract_segment_features(signal_arr, breakpoints)
    labelled_df = label_on_off(features_df)
    cycles_df   = pair_on_off_cycles(labelled_df, appliance_name=appliance_name)

    if plot:
        visualise_results(signal_arr, breakpoints, labelled_df, cycles_df)

    return cycles_df, labelled_df, breakpoints


# ============================================================
# Standalone demo
# ============================================================

if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "Processed_Data_CSV")
    csv_path = os.path.join(data_dir, "House_1.csv")

    if os.path.exists(csv_path) and _HAS_RUPTURES:
        df = pd.read_csv(csv_path, parse_dates=["Time"], index_col="Time")
        col = "Dishwasher" if "Dishwasher" in df.columns else "Aggregate"
        signal = df[col].iloc[:5000].values
    else:
        rng = np.random.default_rng(4)
        n = 5000
        signal = rng.normal(10, 3, n)
        signal[200:1200]  += 700 + rng.normal(0, 20, 1000)
        signal[2000:2800] += 500 + rng.normal(0, 15, 800)
        signal[3500:4500] += 650 + rng.normal(0, 18, 1000)

    if not _HAS_RUPTURES:
        print("[binary_segmentation] Skipping demo — ruptures not installed.")
    else:
        cycles, labelled, bkps = run_binary_segmentation(
            signal, appliance_name="Dishwasher", plot=True
        )
        print(f"[binary_segmentation] {len(bkps)} breakpoints, "
              f"{len(cycles)} complete cycles detected.")
        if len(cycles):
            print(cycles.to_string(index=False))
