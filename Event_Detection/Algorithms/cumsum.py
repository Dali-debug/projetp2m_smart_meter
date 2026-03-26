"""
cumsum.py — Change-point detection via CUSUM for NILM event detection.

Implements two detector classes:
  - CUSUM_Detector       : standard cumulative-sum detector.
  - ProbCUSUM_Detector   : probabilistic variant based on normal p-values.

Both expose the same high-level interface:
  detector.detect_change_points(data) → (pos_changes, neg_changes, change_points)
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# ============================================================
# Standard CUSUM detector
# ============================================================

class CUSUM_Detector:
    """
    Cumulative-sum (CUSUM) change-point detector.

    During the warm-up period the detector estimates the baseline mean (μ)
    and standard deviation (σ) of the signal.  After warm-up it maintains
    two cumulative sums:

      S_pos[t] = max(0, S_pos[t-1] + (x[t] - μ) / σ - delta/2)
      S_neg[t] = max(0, S_neg[t-1] - (x[t] - μ) / σ - delta/2)

    A change is declared when either sum exceeds *threshold*, after which
    both sums are reset to zero.

    Parameters
    ----------
    warmup_period : int
        Number of initial samples used to estimate μ and σ.
    delta : float
        Allowance parameter (minimum detectable shift measured in σ units).
    threshold : float
        Detection threshold for the cumulative sums.
    """

    def __init__(self, warmup_period=10, delta=10, threshold=20):
        self.warmup_period = warmup_period
        self.delta = delta
        self.threshold = threshold
        self._reset_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_all(self):
        self._warmup_buffer = []
        self._mu = None
        self._sigma = None
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._step = 0

    def _reset(self):
        """Reset cumulative sums after a detection (keep μ and σ)."""
        self._s_pos = 0.0
        self._s_neg = 0.0

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def predict_next(self, observation):
        """
        Process a single observation and return a change flag.

        Parameters
        ----------
        observation : float
            Latest power sample.

        Returns
        -------
        change_detected : bool
            True if a change-point was detected at this step.
        """
        self._step += 1

        # ---- warm-up phase ----
        if self._step <= self.warmup_period:
            self._warmup_buffer.append(observation)
            if self._step == self.warmup_period:
                self._mu = np.mean(self._warmup_buffer)
                self._sigma = max(np.std(self._warmup_buffer), 1e-6)
            return False

        # ---- detection phase ----
        z = (observation - self._mu) / self._sigma
        self._s_pos = max(0.0, self._s_pos + z - self.delta / 2.0)
        self._s_neg = max(0.0, self._s_neg - z - self.delta / 2.0)

        if self._s_pos > self.threshold or self._s_neg > self.threshold:
            self._reset()
            return True

        return False

    # ------------------------------------------------------------------
    # Batch interface
    # ------------------------------------------------------------------

    def detect_change_points(self, data):
        """
        Apply :meth:`predict_next` over an entire sequence.

        Parameters
        ----------
        data : array-like
            1-D sequence of power observations.

        Returns
        -------
        pos_changes : list of int
            Sample indices where S_pos crossed the threshold.
        neg_changes : list of int
            Sample indices where S_neg crossed the threshold.
        change_points : list of int
            Union of pos_changes and neg_changes (sorted).
        """
        self._reset_all()
        pos_changes, neg_changes = [], []

        for i, obs in enumerate(data):
            step = self._step + 1  # look-ahead before calling predict_next

            old_s_pos = self._s_pos
            old_s_neg = self._s_neg
            changed = self.predict_next(obs)

            if changed and step > self.warmup_period:
                z = (obs - self._mu) / self._sigma
                if old_s_pos > old_s_neg:
                    pos_changes.append(i)
                else:
                    neg_changes.append(i)

        change_points = sorted(set(pos_changes) | set(neg_changes))
        return pos_changes, neg_changes, change_points

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_change_points(self, data, change_points, pos_changes, neg_changes):
        """
        Plot the signal with detected change-points annotated.

        Parameters
        ----------
        data : array-like
            Original signal.
        change_points : list of int
            All detected change-point indices.
        pos_changes : list of int
            Positive change-point indices (marked in green).
        neg_changes : list of int
            Negative change-point indices (marked in red).
        """
        data = np.asarray(data)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data, color="steelblue", linewidth=0.8, label="Signal")

        for idx in pos_changes:
            ax.axvline(idx, color="green", linewidth=1.0, alpha=0.7,
                       label="Positive change" if idx == pos_changes[0] else "")
        for idx in neg_changes:
            ax.axvline(idx, color="red", linewidth=1.0, alpha=0.7,
                       label="Negative change" if idx == neg_changes[0] else "")

        ax.set_title(
            f"CUSUM — {len(change_points)} change-points detected  "
            f"(δ={self.delta}, threshold={self.threshold})"
        )
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Power (W)")
        ax.legend()
        plt.tight_layout()
        plt.show()


# ============================================================
# Probabilistic CUSUM detector
# ============================================================

class ProbCUSUM_Detector:
    """
    Probabilistic variant of the CUSUM change-point detector.

    Instead of comparing a cumulative sum to a fixed threshold this variant
    computes the two-tailed p-value of the current observation under the
    baseline Gaussian distribution estimated during warm-up.  A change is
    flagged when the p-value drops below *threshold_probability*.

    Parameters
    ----------
    warmup_period : int
        Number of initial samples used to estimate μ and σ.
    threshold_probability : float
        Significance level (p-value threshold).  Typical value: 0.05.
    """

    def __init__(self, warmup_period=10, threshold_probability=0.05):
        self.warmup_period = warmup_period
        self.threshold_probability = threshold_probability
        self._reset_all()

    def _reset_all(self):
        self._warmup_buffer = []
        self._mu = None
        self._sigma = None
        self._step = 0

    def predict_next(self, observation):
        """Process one sample and return a change flag."""
        self._step += 1

        if self._step <= self.warmup_period:
            self._warmup_buffer.append(observation)
            if self._step == self.warmup_period:
                self._mu = np.mean(self._warmup_buffer)
                self._sigma = max(np.std(self._warmup_buffer), 1e-6)
            return False

        z = abs((observation - self._mu) / self._sigma)
        p_value = 2 * (1 - norm.cdf(z))
        return p_value < self.threshold_probability

    def detect_change_points(self, data):
        """
        Detect change-points over an entire sequence.

        Parameters
        ----------
        data : array-like

        Returns
        -------
        pos_changes : list of int
        neg_changes : list of int
        change_points : list of int
        """
        self._reset_all()
        pos_changes, neg_changes, change_points = [], [], []

        for i, obs in enumerate(data):
            changed = self.predict_next(obs)
            if changed and self._mu is not None:
                change_points.append(i)
                if obs > self._mu:
                    pos_changes.append(i)
                else:
                    neg_changes.append(i)

        return pos_changes, neg_changes, change_points


# ============================================================
# Standalone demo
# ============================================================

if __name__ == "__main__":
    import os
    import pandas as pd

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "Processed_Data_CSV")
    csv_path = os.path.join(data_dir, "House_1.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["Time"], index_col="Time")
        # Try to use a specific appliance column; fall back to Aggregate
        col = "Dishwasher" if "Dishwasher" in df.columns else "Aggregate"
        data = df[col].iloc[:2000].values
    else:
        rng = np.random.default_rng(1)
        data = np.concatenate([
            rng.normal(200, 10, 500),
            rng.normal(600, 15, 300),
            rng.normal(200, 10, 700),
            rng.normal(100, 8,  500),
        ])

    detector = CUSUM_Detector(warmup_period=20, delta=15, threshold=30)
    pos_changes, neg_changes, change_points = detector.detect_change_points(data)
    print(f"[CUSUM] Detected {len(change_points)} change-points.")
    detector.plot_change_points(data, change_points, pos_changes, neg_changes)

    prob_detector = ProbCUSUM_Detector(warmup_period=20, threshold_probability=0.01)
    _, _, cp2 = prob_detector.detect_change_points(data)
    print(f"[ProbCUSUM] Detected {len(cp2)} change-points.")
