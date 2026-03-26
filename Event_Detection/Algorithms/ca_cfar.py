"""
ca_cfar.py — Cell-Averaging CFAR (Constant False Alarm Rate) event detector.

Implements:
  - ca_cfar : vectorised CA-CFAR detector that returns a binary detection mask.

The CA-CFAR algorithm estimates the local noise level from reference cells
that surround a cell under test (CUT) while ignoring guard cells immediately
adjacent to the CUT.  The adaptive threshold is:

    threshold[i] = alpha * mean(reference_cells[i])

and the CUT is declared an event when signal[i] > threshold[i].
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ca_cfar(signal, num_guard_cells=10, num_ref_cells=20, alpha=2.0):
    """
    Cell-Averaging CFAR event detector.

    Parameters
    ----------
    signal : array-like
        1-D power signal (W).
    num_guard_cells : int
        Number of guard cells on each side of the cell under test.
        These cells are excluded from the local noise estimate.
    num_ref_cells : int
        Number of reference cells on each side of the guard window.
        The noise estimate is the mean of these 2*num_ref_cells cells.
    alpha : float
        Scaling factor applied to the noise estimate to obtain the
        detection threshold.  Increase to reduce false alarms; decrease
        to improve sensitivity.

    Returns
    -------
    detection_mask : np.ndarray of bool, shape (n,)
        True at every sample that exceeds the local adaptive threshold.

    Notes
    -----
    Samples that are too close to the signal edges (fewer than
    num_guard_cells + num_ref_cells neighbours on one side) are set to
    False in the output mask.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    detection_mask = np.zeros(n, dtype=bool)

    half_window = num_guard_cells + num_ref_cells

    for i in range(half_window, n - half_window):
        # Left reference cells (skip guard cells)
        left_refs = signal[i - half_window: i - num_guard_cells]
        # Right reference cells (skip guard cells)
        right_refs = signal[i + num_guard_cells + 1: i + half_window + 1]

        ref_cells = np.concatenate([left_refs, right_refs])

        if len(ref_cells) == 0:
            continue

        noise_estimate = ref_cells.mean()
        threshold = alpha * noise_estimate

        if signal[i] > threshold:
            detection_mask[i] = True

    return detection_mask


def plot_cfar_detections(signal, detection_mask, title="CA-CFAR Detections"):
    """
    Visualise the signal and the detected events.

    Parameters
    ----------
    signal : array-like
        Original power signal.
    detection_mask : np.ndarray of bool
        Detection mask produced by :func:`ca_cfar`.
    title : str
        Figure title.
    """
    signal = np.asarray(signal, dtype=float)
    event_indices = np.where(detection_mask)[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(signal, color="steelblue", linewidth=0.7, label="Signal")
    ax.scatter(event_indices, signal[event_indices],
               color="red", s=20, zorder=5, label=f"Events ({len(event_indices)})")
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power (W)")
    ax.legend()
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
        signal = df[col].iloc[:5000].values
    else:
        rng = np.random.default_rng(2)
        # Simulate a dishwasher cycle embedded in background noise
        signal = rng.normal(10, 5, 5000)
        signal[500:600]   += 800   # wash
        signal[1200:1400] += 600   # rinse
        signal[2000:2050] += 400   # drain

    detection_mask = ca_cfar(signal, num_guard_cells=10, num_ref_cells=20, alpha=2.0)

    if isinstance(signal, np.ndarray):
        df_out = pd.DataFrame({"Signal": signal, "Detection": detection_mask})
    else:
        df_out = pd.DataFrame({"Signal": signal, "Detection": detection_mask})

    n_events = int(detection_mask.sum())
    print(f"[CA-CFAR] Detected {n_events} event samples "
          f"({100*n_events/len(signal):.2f}% of signal).")

    plot_cfar_detections(signal, detection_mask)
