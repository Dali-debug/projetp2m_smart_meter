"""
local_threshold_based.py — Adaptive local-threshold event detector.

Implements:
  - detect_and_plot_dishwasher_events : detects events in a power signal
    using a sliding-window adaptive threshold and scipy peak detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt


def detect_and_plot_dishwasher_events(power_series, window_size=1000,
                                       a=1.0, b=1.0, plot=True):
    """
    Detect ON/OFF events using an adaptive local threshold.

    Algorithm
    ---------
    1. Apply a median filter (kernel size 61, i.e. ~60 samples) to smooth
       the signal into *p_med*.
    2. Compute the absolute first difference of *p_med* → *delta_p*.
    3. For each sliding window of *window_size* samples compute the local
       mean *mu_w* and standard deviation *sigma_w* of *delta_p*.
    4. Derive global thresholds:
         s_p = mean(sigma_w) + mean(mu_w)   (activity threshold)
         s_a = s_p / 2                       (smaller threshold for active windows)
    5. Identify **active windows** where sigma_w > s_a.
    6. Within each active window detect peaks in *delta_p* that exceed
         local_threshold = a * mu_w[w] + b * sigma_w[w].
    7. De-duplicate event indices and return the sorted list.

    Parameters
    ----------
    power_series : array-like or pd.Series
        1-D power signal in Watts.
    window_size : int
        Number of samples in each sliding analysis window.
    a : float
        Weight applied to the local mean for the local threshold.
    b : float
        Weight applied to the local standard deviation for the threshold.
    plot : bool
        Whether to display the annotated plot.

    Returns
    -------
    event_indices : np.ndarray of int
        Sorted array of sample indices where events were detected.
    """
    power = np.asarray(power_series, dtype=float)
    n = len(power)

    # Step 1 — median filter (kernel must be odd)
    kernel_size = 61
    p_med = medfilt(power, kernel_size=kernel_size).astype(float)

    # Step 2 — absolute differences
    delta_p = np.abs(np.diff(p_med))
    delta_p = np.append(delta_p, 0.0)  # keep same length

    # Step 3 — sliding window statistics
    n_windows = max(1, n - window_size + 1)
    mu_w    = np.zeros(n_windows)
    sigma_w = np.zeros(n_windows)

    for w in range(n_windows):
        seg = delta_p[w: w + window_size]
        mu_w[w]    = seg.mean()
        sigma_w[w] = seg.std()

    # Step 4 — global thresholds
    s_p = mu_w.mean() + sigma_w.mean()
    s_a = s_p / 2.0

    # Step 5 — active windows
    active_windows = np.where(sigma_w > s_a)[0]

    # Step 6 — peak detection within each active window
    event_set = set()
    for w in active_windows:
        local_thresh = a * mu_w[w] + b * sigma_w[w]
        seg = delta_p[w: w + window_size]
        peaks, _ = find_peaks(seg, height=local_thresh)
        # Convert window-relative indices to absolute indices
        for p_idx in peaks:
            event_set.add(w + p_idx)

    event_indices = np.array(sorted(event_set), dtype=int)

    # Step 7 — visualisation
    if plot:
        _plot_events(power, event_indices)

    return event_indices


def _plot_events(power, event_indices):
    """Plot the power signal with detected event markers."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(power, color="steelblue", linewidth=0.7, label="Power signal")
    if len(event_indices) > 0:
        ax.scatter(event_indices, power[event_indices],
                   color="red", s=25, zorder=5,
                   label=f"Events ({len(event_indices)})")
    ax.set_title("Local threshold-based event detection")
    ax.set_ylabel("Power (W)")
    ax.legend()

    # Delta signal
    delta = np.abs(np.diff(power))
    delta = np.append(delta, 0.0)
    ax2 = axes[1]
    ax2.plot(delta, color="darkorange", linewidth=0.6, label="|diff(power)|")
    if len(event_indices) > 0:
        ax2.scatter(event_indices, delta[event_indices],
                    color="red", s=25, zorder=5)
    ax2.set_ylabel("|ΔPower| (W)")
    ax2.set_xlabel("Sample index")
    ax2.legend()

    plt.tight_layout()
    plt.show()


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
        col = "Dishwasher" if "Dishwasher" in df.columns else "Aggregate"
        power_series = df[col].iloc[:5000]
    else:
        rng = np.random.default_rng(3)
        n = 5000
        base = 10 + rng.normal(0, 3, n)
        # Simulate dishwasher cycles
        base[300:800]   += 600 + rng.normal(0, 20, 500)
        base[1000:1300] += 400 + rng.normal(0, 15, 300)
        base[2500:3000] += 700 + rng.normal(0, 25, 500)
        power_series = pd.Series(base)

    events = detect_and_plot_dishwasher_events(
        power_series, window_size=1000, a=1.0, b=1.0, plot=True
    )
    print(f"[local_threshold] Detected {len(events)} events.")
    if len(events):
        print(f"  First 10 event indices: {events[:10]}")
