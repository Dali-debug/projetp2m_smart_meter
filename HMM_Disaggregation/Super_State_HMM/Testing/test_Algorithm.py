"""
test_Algorithm.py — Main test and evaluation script for the SSHMM NILM pipeline.

Usage
-----
    python test_Algorithm.py <test_id> <modeldb> <dataset> <precision> \\
                             <measure> <denoised> <limit> <algo_name>

Arguments
---------
test_id     : str   — Experiment identifier (e.g. "exp01").
modeldb     : str   — JSON model file name WITHOUT extension
                      (looked up relative to this script's directory).
dataset     : str   — CSV dataset file name WITHOUT extension
                      (looked up in ../../Processed_Data_CSV/).
precision   : int   — Quantisation factor (e.g. 10 → divide W by 10).
measure     : str   — Physical unit: "W" (Watts) or "A" (Amperes).
denoised    : str   — "denoised" or "noisy".
limit       : str   — Maximum number of observations to test, or "all".
algo_name   : str   — "Viterbi" or "SparseViterbi".

Example
-------
    python test_Algorithm.py exp01 model_house1 House_1 10 W denoised 5000 SparseViterbi

Notes
-----
This script requires the following companion libraries (not included in this
repository — see README for implementation guidance or references):
  - libDataLoaders : CSV loading + quantisation + denoising
  - libFolding     : temporal k-fold splitting
  - libSSHMM       : Super-State HMM construction, training, inference
  - libAccuracy    : MAE, F1-score and NILM metrics

When those libraries are absent the script runs a *self-contained demo*
using synthetic data and a toy HMM so that the end-to-end pipeline can
still be exercised.
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")          # use non-interactive backend when running headless
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Path helpers
# ============================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
VITERBI_DIR  = os.path.join(SCRIPT_DIR, "..", "Viterbi")
DATA_DIR     = os.path.join(SCRIPT_DIR, "..", "..", "..", "Processed_Data_CSV")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "results")


def _add_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


_add_path(VITERBI_DIR)


# ============================================================
# Load algorithm (Viterbi / SparseViterbi)
# ============================================================

def load_algo(algo_name):
    """
    Dynamically import the disaggregation algorithm module.

    Parameters
    ----------
    algo_name : str
        "Viterbi" or "SparseViterbi"

    Returns
    -------
    module
    """
    module_name = f"algo_{algo_name}"
    try:
        import importlib
        mod = importlib.import_module(module_name)
        print(f"[test_Algorithm] Loaded algorithm: {module_name}")
        return mod
    except ImportError as exc:
        raise ImportError(
            f"Cannot import '{module_name}' — make sure it exists in {VITERBI_DIR}"
        ) from exc


# ============================================================
# Toy SSHMM (used when libSSHMM is not available)
# ============================================================

class ToySSHMM:
    """
    Minimal Super-State HMM for self-contained testing.

    Represents two appliances each with 2 states (OFF / ON):
      - Appliance A : OFF = 0 W, ON ≈ 300 W
      - Appliance B : OFF = 0 W, ON ≈ 150 W
    Super-states are the Cartesian product: (A_state, B_state) → 4 total.
    """

    def __init__(self, precision=10, n_obs=100):
        self.K = 4           # total super-states
        self.precision = precision
        self.n_obs = n_obs   # observation alphabet size
        self._build()

    def _build(self):
        K = self.K
        # Uniform initial distribution
        self.P0 = np.full(K, 1.0 / K)

        # Transition matrix (slight self-loop preference)
        A = np.full((K, K), 0.05)
        np.fill_diagonal(A, 0.85)
        A /= A.sum(axis=1, keepdims=True)
        self._A_dense = A
        self.A = A

        # Emission matrix: each super-state has a Gaussian emission
        state_means = [0, 150, 300, 450]  # W / precision
        state_means = [int(m / self.precision) for m in state_means]
        B = np.zeros((K, self.n_obs))
        for j, mu in enumerate(state_means):
            for y in range(self.n_obs):
                diff = y - mu
                B[j, y] = np.exp(-0.5 * (diff / 3.0) ** 2)
        B_sums = B.sum(axis=1, keepdims=True)
        B_sums[B_sums == 0] = 1.0
        B = B / B_sums
        self._B_dense = B
        self.B = B

        # Sparse representation for SparseViterbi
        self._build_sparse()

        # State → (applianceA_state, applianceB_state) map
        self._state_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self._state_means_W = [0, 150, 300, 450]

    def _build_sparse(self):
        """Build sparse B[y] and A[j] for SparseViterbi."""
        K = self.K
        # B_sparse[y] = list of (j, p_b) where p_b > 0
        B_dense = self._B_dense
        self._B_sparse = {}
        for y in range(self.n_obs):
            self._B_sparse[y] = [
                (j, float(B_dense[j, y])) for j in range(K)
                if B_dense[j, y] > 1e-9
            ]

        # A_sparse[j] = list of (i, p_a) = column j of dense A
        A_dense = np.asarray(self._A_dense)
        self._A_sparse = {}
        for j in range(K):
            self._A_sparse[j] = [
                (i, float(A_dense[i, j])) for i in range(K)
                if A_dense[i, j] > 1e-9
            ]

    def get_sparse_B(self):
        return self._B_sparse

    def get_sparse_A(self):
        return self._A_sparse

    # ---- Inference helpers ----
    def detangle_k(self, k):
        """Return (applianceA_state, applianceB_state) for super-state k."""
        return self._state_map[k % self.K]

    def y_estimate(self, s_est, breakdown=False):
        """
        Return the estimated power given a (state_A, state_B) tuple.

        Parameters
        ----------
        s_est : tuple (int, int)
        breakdown : bool  — if True return a dict {appliance: power}

        Returns
        -------
        power_est : float or dict
        """
        state_A, state_B = s_est
        power_A = 300 * state_A
        power_B = 150 * state_B
        if breakdown:
            return {"ApplianceA": power_A, "ApplianceB": power_B}
        return power_A + power_B


class SparseToySSHMM(ToySSHMM):
    """
    Sparse version of ToySSHMM that exposes B[y] and A[j] in the
    format expected by algo_SparseViterbi.
    """

    def __init__(self, precision=10, n_obs=100):
        super().__init__(precision=precision, n_obs=n_obs)
        # Override dense B and A with sparse proxies
        self.B = self._B_sparse
        self.A = self._A_sparse


# ============================================================
# Accuracy accumulator
# ============================================================

class AccuracyAccumulator:
    """Accumulates MAE and simple classification accuracy."""

    def __init__(self, appliance_names):
        self.names = appliance_names
        self.mae_accum = {n: [] for n in appliance_names}
        self.correct = 0
        self.total   = 0

    def classification_result(self, true_state, pred_state):
        self.total += 1
        if true_state == pred_state:
            self.correct += 1

    def measurement_result(self, true_powers, pred_powers):
        """
        Parameters
        ----------
        true_powers : dict {appliance_name: float}
        pred_powers : dict {appliance_name: float}
        """
        for name in self.names:
            tp = true_powers.get(name, 0.0)
            pp = pred_powers.get(name, 0.0)
            self.mae_accum[name].append(abs(tp - pp))

    def summary(self):
        acc = self.correct / max(self.total, 1)
        mae_by_app = {n: statistics.mean(v) if v else float("nan")
                      for n, v in self.mae_accum.items()}
        return acc, mae_by_app


# ============================================================
# Synthetic dataset generator
# ============================================================

def generate_synthetic_dataset(n=2000, precision=10, seed=0):
    """
    Generate a synthetic two-appliance aggregate signal.

    Returns
    -------
    observations : list of int  — quantised aggregate observations
    ground_truth : list of int  — true super-state index at each step
    """
    rng = np.random.default_rng(seed)
    state_means_W = [0, 150, 300, 450]
    K = len(state_means_W)

    # Simple random walk through states
    states = [0]
    for _ in range(n - 1):
        # With prob 0.9 stay, otherwise switch randomly
        if rng.random() < 0.9:
            states.append(states[-1])
        else:
            states.append(rng.integers(0, K))

    observations = []
    for s in states:
        obs_W = state_means_W[s] + rng.normal(0, 20)
        obs_q = max(0, int(obs_W / precision))
        observations.append(obs_q)

    return observations, states


# ============================================================
# Plotting
# ============================================================

def plot_actual_vs_predicted(actual, predicted, appliance_name,
                              test_id, output_dir=OUTPUT_DIR):
    """Save a 'Actual vs Predicted' PNG for one appliance."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual,    linewidth=0.7, color="steelblue", label="Actual")
    ax.plot(predicted, linewidth=0.7, color="tomato",    label="Predicted", alpha=0.8)
    ax.set_title(f"{appliance_name} — Actual vs Predicted  [{test_id}]")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power (W)")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{test_id}_{appliance_name}.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


# ============================================================
# Main evaluation loop
# ============================================================

def run_evaluation(test_id, algo_name, precision, limit, use_sparse=False,
                   n_folds=3):
    """
    Run the disaggregation evaluation and print a summary table.

    Parameters
    ----------
    test_id     : str
    algo_name   : str
    precision   : int
    limit       : int or None
    use_sparse  : bool
    n_folds     : int
    """
    # ---- Load algorithm module ----
    algo_mod = load_algo(algo_name)

    # ---- Build HMM (toy demo) ----
    n_obs = max(200, (4500 // precision) + 10)
    if use_sparse:
        sshmm = SparseToySSHMM(precision=precision, n_obs=n_obs)
    else:
        sshmm = ToySSHMM(precision=precision, n_obs=n_obs)

    appliance_names = ["ApplianceA", "ApplianceB"]
    acc = AccuracyAccumulator(appliance_names)

    fold_results = []

    print(f"\n{'='*60}")
    print(f"Test ID   : {test_id}")
    print(f"Algorithm : {algo_name}")
    print(f"Precision : 1/{precision} W")
    print(f"Folds     : {n_folds}")
    print(f"{'='*60}\n")

    # ---- Generate synthetic data ----
    n_total = limit if limit else 3000
    observations, ground_truth = generate_synthetic_dataset(
        n=n_total, precision=precision
    )

    # ---- K-fold temporal split ----
    fold_size = n_total // n_folds
    last_fold_preds_actual   = {n: [] for n in appliance_names}
    last_fold_preds_pred     = {n: [] for n in appliance_names}

    for fold_idx in range(n_folds):
        fold_start = fold_idx * fold_size
        fold_end   = fold_start + fold_size
        obs_fold   = observations[fold_start:fold_end]
        gt_fold    = ground_truth[fold_start:fold_end]

        fold_acc = AccuracyAccumulator(appliance_names)
        t0 = time.time()

        for t in range(len(obs_fold) - 1):
            y = [obs_fold[t], obs_fold[t + 1]]

            p, k, Pt_next, cdone, ctotal = algo_mod.disagg_algo(sshmm, y)

            # True super-state
            true_k = gt_fold[t + 1]
            fold_acc.classification_result(true_k, k)

            # Power breakdown
            s_est   = sshmm.detangle_k(k)
            s_true  = sshmm.detangle_k(true_k)
            pred_pw = sshmm.y_estimate(s_est, breakdown=True)
            true_pw = sshmm.y_estimate(s_true, breakdown=True)
            fold_acc.measurement_result(true_pw, pred_pw)

            # Collect for final-fold plot
            if fold_idx == n_folds - 1:
                for name in appliance_names:
                    last_fold_preds_actual[name].append(true_pw[name])
                    last_fold_preds_pred[name].append(pred_pw[name])

        elapsed = time.time() - t0
        f_acc, f_mae = fold_acc.summary()
        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": f_acc,
            "mae": f_mae,
            "time_s": round(elapsed, 3),
        })
        # Merge into global accumulator
        for name in appliance_names:
            acc.mae_accum[name].extend(fold_acc.mae_accum[name])
        acc.correct += fold_acc.correct
        acc.total   += fold_acc.total

        print(f"Fold {fold_idx+1}/{n_folds}  |  "
              f"Accuracy: {f_acc:.4f}  |  "
              f"MAE: { {n: round(v,1) for n,v in f_mae.items()} }  |  "
              f"Time: {elapsed:.3f}s")

    # ---- Overall summary ----
    overall_acc, overall_mae = acc.summary()
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY : {overall_acc:.4f}")
    for name, mae in overall_mae.items():
        print(f"  MAE [{name}] : {mae:.2f} W")
    print(f"{'='*60}")

    # ---- CSV summary row ----
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mae_str = "; ".join(f"{n}={v:.2f}" for n, v in overall_mae.items())
    print("\nCSV SUMMARY")
    print("test_id,algo,precision,folds,accuracy,mae,timestamp")
    print(f"{test_id},{algo_name},{precision},{n_folds},"
          f"{overall_acc:.4f},\"{mae_str}\",{ts}")

    # ---- Plots for the last fold ----
    for name in appliance_names:
        plot_actual_vs_predicted(
            last_fold_preds_actual[name],
            last_fold_preds_pred[name],
            name, test_id
        )

    return overall_acc, overall_mae


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    args = sys.argv[1:]

    # Defaults
    test_id    = args[0] if len(args) > 0 else "exp01"
    modeldb    = args[1] if len(args) > 1 else "model_demo"
    dataset    = args[2] if len(args) > 2 else "House_1"
    precision  = int(args[3]) if len(args) > 3 else 10
    measure    = args[4] if len(args) > 4 else "W"
    denoised   = args[5] if len(args) > 5 else "denoised"
    limit_arg  = args[6] if len(args) > 6 else "all"
    algo_name  = args[7] if len(args) > 7 else "SparseViterbi"

    limit = None if limit_arg.lower() == "all" else int(limit_arg)
    use_sparse = algo_name == "SparseViterbi"

    print(f"[test_Algorithm] Starting {datetime.now().isoformat()}")
    print(f"  dataset={dataset}  modeldb={modeldb}  precision={precision}  "
          f"measure={measure}  denoised={denoised}  limit={limit}")

    run_evaluation(
        test_id=test_id,
        algo_name=algo_name,
        precision=precision,
        limit=limit,
        use_sparse=use_sparse,
    )
