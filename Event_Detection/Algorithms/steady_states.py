"""
steady_states.py — Hart (1985) steady-state detection for NILM.

Implements:
  - find_steady_states  : identifies steady-state intervals and the transitions
                          between them in a single power-consumption signal.
  - find_steady_states_transients : wrapper that processes multiple meter
                          groups / data chunks and concatenates results.
  - cluster             : lightweight K-Means helper (identical to cluster.py)
                          kept here for standalone operation.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# Internal K-Means helper (mirrors cluster.py Section 3.5)
# ============================================================

def cluster(x, max_num_clusters=3):
    """
    Cluster a 1-D power series into up to *max_num_clusters* states.

    Parameters
    ----------
    x : array-like
        Power values (W).
    max_num_clusters : int
        Maximum number of clusters to consider.

    Returns
    -------
    centroids : np.ndarray of int32
        Sorted cluster centroids including the 0-W (OFF) state.
    """
    x = np.asarray(x).flatten()
    x = x[x > 10]  # discard OFF state during clustering
    if len(x) == 0:
        return np.array([0], dtype=np.int32)

    # Subsample for speed
    if len(x) > 2000:
        rng = np.random.default_rng(0)
        x = rng.choice(x, size=2000, replace=False)

    X = x.reshape(-1, 1)
    best_k, best_score, best_labels = 1, -1.0, np.zeros(len(x), dtype=int)

    for k in range(1, max_num_clusters):
        if k >= len(x):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        if k == 1:
            best_k = 1
            best_labels = km.labels_
            best_centers = km.cluster_centers_.flatten()
            continue
        score = silhouette_score(X, km.labels_)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = km.labels_
            best_centers = km.cluster_centers_.flatten()

    if best_k == 1:
        km = KMeans(n_clusters=1, n_init=10, random_state=0).fit(X)
        best_centers = km.cluster_centers_.flatten()

    centroids = np.sort(np.append(best_centers, 0)).astype(np.int32)
    return centroids


# ============================================================
# Hart (1985) steady-state detection
# ============================================================

def find_steady_states(dataframe, min_n_samples=2, state_threshold=15,
                        noise_level=70):
    """
    Detect steady-state intervals and power transitions (Hart 1985).

    The algorithm walks the time series sample by sample, maintaining an
    incremental estimate of the current steady-state level.  A transition is
    flagged when the instantaneous change exceeds *state_threshold*.  Only
    transitions whose magnitude is larger than *noise_level* are kept.

    Parameters
    ----------
    dataframe : pd.DataFrame or pd.Series
        Input data.  Must contain at least one numeric column representing
        active power (W).  When two columns are present the second is treated
        as reactive power (VAr).
    min_n_samples : int
        Minimum number of consecutive samples required for a state to be
        considered stable.
    state_threshold : float
        Minimum instantaneous change (W) to declare a transition.
    noise_level : float
        Minimum *magnitude* (W) of a transition to be included in the output.

    Returns
    -------
    steady_states : pd.DataFrame
        One row per steady-state interval.
        Columns: ``['power_active', 'power_reactive']`` (reactive only when
        present in the input).
    transitions : pd.DataFrame
        One row per significant transition.
        Columns: ``['power_active', 'power_reactive', 'time']`` .
    """
    if isinstance(dataframe, pd.Series):
        dataframe = dataframe.to_frame(name="power_active")

    cols = list(dataframe.columns)
    has_reactive = len(cols) >= 2

    values = dataframe.values.astype(float)
    n = len(values)

    ss_records = []   # steady-state rows
    tr_records = []   # transition rows

    state_start = 0
    state_sum = values[0].copy()
    state_count = 1
    prev = values[0].copy()

    print(f"[find_steady_states] Processing {n} samples …", flush=True)

    for i in range(1, n):
        current = values[i]
        delta = abs(current[0] - prev[0])

        if delta > state_threshold:
            # --- record current steady state (if long enough) ---
            if state_count >= min_n_samples:
                mean_state = state_sum / state_count
                if has_reactive:
                    ss_records.append({
                        "power_active":   mean_state[0],
                        "power_reactive": mean_state[1],
                    })
                else:
                    ss_records.append({"power_active": mean_state[0]})

                # --- record transition if magnitude is significant ---
                transition_mag = abs(current[0] - (state_sum[0] / state_count))
                if transition_mag > noise_level:
                    ts = dataframe.index[i] if isinstance(dataframe.index, pd.DatetimeIndex) \
                        else i
                    if has_reactive:
                        tr_records.append({
                            "power_active":   current[0] - (state_sum[0] / state_count),
                            "power_reactive": current[1] - (state_sum[1] / state_count),
                            "time": ts,
                        })
                    else:
                        tr_records.append({
                            "power_active": current[0] - (state_sum[0] / state_count),
                            "time": ts,
                        })

            # --- start new state ---
            state_start = i
            state_sum = current.copy()
            state_count = 1
        else:
            state_sum += current
            state_count += 1

        prev = current

    # Flush the last state
    if state_count >= min_n_samples:
        mean_state = state_sum / state_count
        if has_reactive:
            ss_records.append({
                "power_active":   mean_state[0],
                "power_reactive": mean_state[1],
            })
        else:
            ss_records.append({"power_active": mean_state[0]})

    steady_states = pd.DataFrame(ss_records)
    transitions   = pd.DataFrame(tr_records)

    print(f"[find_steady_states] Found {len(steady_states)} steady states, "
          f"{len(transitions)} significant transitions.", flush=True)

    return steady_states, transitions


def find_steady_states_transients(metergroup, columns, noise_level=70,
                                   state_threshold=15):
    """
    Apply :func:`find_steady_states` to multiple meter segments and
    concatenate results.

    Parameters
    ----------
    metergroup : iterable of pd.DataFrame
        Sequence of DataFrames, each representing a contiguous time segment.
    columns : list of str
        Column names to extract from each DataFrame.
    noise_level : float
        Forwarded to :func:`find_steady_states`.
    state_threshold : float
        Forwarded to :func:`find_steady_states`.

    Returns
    -------
    all_steady_states : pd.DataFrame
        Concatenated steady-state records from all segments.
    all_transitions : pd.DataFrame
        Concatenated transition records from all segments.
    """
    ss_list, tr_list = [], []

    for idx, segment in enumerate(metergroup):
        if isinstance(segment, pd.DataFrame):
            data = segment[columns] if columns else segment
        else:
            data = segment

        ss, tr = find_steady_states(
            data,
            noise_level=noise_level,
            state_threshold=state_threshold,
        )
        ss_list.append(ss)
        tr_list.append(tr)

    all_steady_states = pd.concat(ss_list, ignore_index=True) if ss_list else pd.DataFrame()
    all_transitions   = pd.concat(tr_list, ignore_index=True) if tr_list else pd.DataFrame()
    return all_steady_states, all_transitions


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
        series = df["Aggregate"].iloc[:5000]
    else:
        rng = np.random.default_rng(0)
        n = 5000
        values = np.zeros(n)
        # Simulate ON/OFF events
        values[:1000] = 200 + rng.normal(0, 10, 1000)
        values[1000:1500] = 600 + rng.normal(0, 15, 500)
        values[1500:3000] = 200 + rng.normal(0, 10, 1500)
        values[3000:4000] = 400 + rng.normal(0, 12, 1000)
        values[4000:] = 200 + rng.normal(0, 10, 1000)
        series = pd.Series(values, name="Aggregate")

    ss, tr = find_steady_states(series)
    print(ss.head())
    print(tr.head())
