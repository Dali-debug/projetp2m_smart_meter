"""
Appliance.py — Online K-Means clustering and HMM parameter generation
               for a single electrical appliance.

The Appliance class wraps a pd.Series of power measurements and performs:
  1. Online nearest-centroid clustering to assign each sample to a state.
  2. Incremental update of per-state statistics (mean, variance, min, max).
  3. Construction of the HMM transition matrix from state-dwell sequences.
  4. Pretty-printing of the HMM parameters (π, A, μ, Σ) ready to paste
     into a SSHMM configuration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import clear_output
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False


class Appliance:
    """
    Represents a single electrical appliance with online clustering.

    Parameters
    ----------
    name : str
        Human-readable appliance name (e.g. ``"refrigerator"``).
    series : pd.Series
        Time series of appliance power consumption in Watts.
    initial_means : list of float
        Initial cluster centroids sorted in ascending order.
        The number of centroids determines the number of HMM states.
        Conventionally the first entry is 0 (OFF state).

    Attributes
    ----------
    numberOfClusters : int
    means : list of float
    covs : list of float
    mins : list of float
    maxs : list of float
    numberOfPoints : list of int
    transitionMatrix : np.ndarray, shape (K, K)
    clusteredSeries : pd.Series
        Same length as *series*; each value is the cluster index [0..K-1].
    stateTransitions : list of (int, int)
        Sequence of (state_index, duration_in_samples) pairs.
    """

    def __init__(self, name, series, initial_means):
        self.name = name
        self.series = series.copy()
        self.numberOfClusters = len(initial_means)

        self.means = list(float(m) for m in initial_means)
        self.covs  = [1.0] * self.numberOfClusters
        self.mins  = [float("inf")] * self.numberOfClusters
        self.maxs  = [float("-inf")] * self.numberOfClusters
        self.numberOfPoints = [0] * self.numberOfClusters

        self.transitionMatrix = np.eye(self.numberOfClusters)
        self.clusteredSeries  = pd.Series(dtype=int)
        self.stateTransitions = []
        self.seriesPerCluster = [pd.Series(dtype=float)
                                 for _ in range(self.numberOfClusters)]

    # ------------------------------------------------------------------
    # Core clustering helpers
    # ------------------------------------------------------------------

    def findCluster(self, val):
        """Return the index of the closest cluster centroid to *val*.

        Parameters
        ----------
        val : float

        Returns
        -------
        idx : int
        """
        distances = [abs(val - m) for m in self.means]
        return int(np.argmin(distances))

    def calculateClustersMeans(self):
        """
        Single pass over *series* to update cluster means incrementally.

        Uses an online (Welford-style) running mean so that the centroids
        converge towards the true cluster means.
        """
        counts = [0] * self.numberOfClusters
        sums   = [0.0] * self.numberOfClusters

        print(f"[{self.name}] Calculating cluster means …", flush=True)

        for val in self.series:
            idx = self.findCluster(val)
            counts[idx] += 1
            sums[idx]   += val

        for k in range(self.numberOfClusters):
            if counts[k] > 0:
                self.means[k] = sums[k] / counts[k]

    def classifyPoints(self):
        """
        Assign every sample in *series* to the nearest cluster centroid.

        Populates :attr:`seriesPerCluster`.
        """
        cluster_data = [[] for _ in range(self.numberOfClusters)]
        cluster_idx  = []

        for val in self.series:
            idx = self.findCluster(val)
            cluster_data[idx].append(val)
            cluster_idx.append(idx)

        self.clusteredSeries  = pd.Series(cluster_idx, index=self.series.index)
        self.seriesPerCluster = [pd.Series(cluster_data[k])
                                 for k in range(self.numberOfClusters)]

    def updateParameters(self):
        """
        Recompute per-cluster statistics, state transitions, and
        the HMM transition matrix from the current cluster assignments.
        """
        for k in range(self.numberOfClusters):
            pts = self.seriesPerCluster[k]
            n   = len(pts)
            self.numberOfPoints[k] = n
            if n > 0:
                self.means[k] = float(pts.mean())
                self.covs[k]  = float(pts.var()) if n > 1 else 1.0
                self.mins[k]  = float(pts.min())
                self.maxs[k]  = float(pts.max())
            else:
                self.covs[k]  = 1.0

        self.stateTransitions = self.getStateTransitions()
        self.transitionMatrix = self.calculateTransitionMatrix()

    def updateClusters(self):
        """
        Full update workflow:
        1. :meth:`calculateClustersMeans`
        2. :meth:`classifyPoints`
        3. :meth:`updateParameters`
        """
        self.calculateClustersMeans()
        self.classifyPoints()
        self.updateParameters()

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    def getStateTransitions(self):
        """
        Extract (state, duration) pairs from *clusteredSeries*, filtering
        out short transients (duration ≤ 3 consecutive samples).

        Returns
        -------
        transitions : list of (int, int)
            Each entry is (state_index, duration).
        """
        if len(self.clusteredSeries) == 0:
            return []

        transitions = []
        prev_state  = self.clusteredSeries.iloc[0]
        duration    = 1

        for state in self.clusteredSeries.iloc[1:]:
            if state == prev_state:
                duration += 1
            else:
                # Filter out transients (short intermediate passages)
                if duration > 3:
                    transitions.append((int(prev_state), duration))
                prev_state = state
                duration   = 1

        if duration > 3:
            transitions.append((int(prev_state), duration))

        return transitions

    def calculateTransitionMatrix(self):
        """
        Build and row-normalise the HMM transition matrix from
        :attr:`stateTransitions`.

        Returns
        -------
        A : np.ndarray, shape (K, K)
        """
        K = self.numberOfClusters
        A = np.zeros((K, K))

        transitions = self.stateTransitions
        for t in range(len(transitions) - 1):
            from_state = transitions[t][0]
            to_state   = transitions[t + 1][0]
            if 0 <= from_state < K and 0 <= to_state < K:
                A[from_state, to_state] += 1

        # Row-normalise; rows that sum to 0 become uniform distributions
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        A = A / row_sums

        return A

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plotClusteredSeries(self):
        """Plot *clusteredSeries* (cluster index over time)."""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.clusteredSeries.values, linewidth=0.6, color="steelblue")
        ax.set_title(f"{self.name} — cluster index over time")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Cluster index")
        ax.set_yticks(range(self.numberOfClusters))
        plt.tight_layout()
        plt.show()

    def plot(self):
        """Plot the raw power series with cluster centroids as horizontal lines."""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.series.values, linewidth=0.6, color="steelblue",
                label="Power")
        for k, mean_val in enumerate(self.means):
            ax.axhline(mean_val, linestyle="--", linewidth=1.0,
                       label=f"State {k} ({mean_val:.1f} W)")
        ax.set_title(f"{self.name} — power series and cluster centroids")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Power (W)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # HMM parameter export
    # ------------------------------------------------------------------

    def __str__(self):
        """
        Return a Python code snippet that initialises the HMM parameters
        for this appliance.

        The output follows the naming convention used by the SSHMM library:
          pi['name'] = np.array([...])
          a['name']  = np.array([[...]])
          mean['name'] = np.array([[μ0], [μ1], ...])
          cov['name']  = np.array([[[σ²0]], [[σ²1]], ...])
        """
        K = self.numberOfClusters
        n_total = max(sum(self.numberOfPoints), 1)

        # Initial state distribution: proportional to time spent in each state
        pi_vals = [self.numberOfPoints[k] / n_total for k in range(K)]
        pi_str  = "np.array([" + ", ".join(f"{v:.6f}" for v in pi_vals) + "])"

        # Transition matrix
        a_rows = []
        for row in self.transitionMatrix:
            a_rows.append("[" + ", ".join(f"{v:.6f}" for v in row) + "]")
        a_str = "np.array([" + ", ".join(a_rows) + "])"

        # Means
        mean_rows = [f"[{self.means[k]:.4f}]" for k in range(K)]
        mean_str  = "np.array([" + ", ".join(mean_rows) + "])"

        # Covariances
        cov_rows = [f"[[{max(self.covs[k], 1e-6):.4f}]]" for k in range(K)]
        cov_str  = "np.array([" + ", ".join(cov_rows) + "])"

        name = self.name
        lines = [
            f"pi['{name}']   = {pi_str}",
            f"a['{name}']    = {a_str}",
            f"mean['{name}'] = {mean_str}",
            f"cov['{name}']  = {cov_str}",
        ]
        return "\n".join(lines)


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
        col = col or df.columns[1]
        series = df[col].iloc[:10000]
    else:
        rng = np.random.default_rng(6)
        n = 10000
        values = np.zeros(n)
        # Refrigerator: OFF (0 W) and compressor ON (~150 W) cycles
        idx = 0
        while idx < n:
            off_dur = rng.integers(200, 600)
            on_dur  = rng.integers(100, 300)
            end_off = min(idx + off_dur, n)
            values[idx:end_off] = rng.normal(2, 3, end_off - idx)
            idx = end_off
            end_on = min(idx + on_dur, n)
            values[idx:end_on] = rng.normal(150, 10, end_on - idx)
            idx = end_on
        series = pd.Series(values, name="Refrigerator")

    appliance = Appliance("refrigerator", series, initial_means=[0, 80, 160])
    appliance.updateClusters()
    appliance.plot()
    appliance.plotClusteredSeries()
    print(appliance)
