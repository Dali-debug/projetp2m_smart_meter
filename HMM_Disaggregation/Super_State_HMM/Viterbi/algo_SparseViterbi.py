"""
algo_SparseViterbi.py — Sparse Viterbi step for SSHMM disaggregation.

Implements:
  disagg_algo(hmm, y)

Functionally equivalent to algo_Viterbi.disagg_algo but uses dictionaries
and sparse lists to skip zero-probability transitions, yielding 70–99%
fewer computations on large models.

Sparse interface contract
--------------------------
The *hmm* object must expose:
  hmm.K  : int
  hmm.P0 : array-like, shape (K,)        — initial distribution
  hmm.A  : list/dict where A[j] is a list of (i, p_a) tuples
           representing non-zero entries of the transition column for state j
           i.e.  A[j] = [(i, P(j | i)), ...]
  hmm.B  : list/dict where B[y] is a list of (j, p_b) tuples
           representing non-zero emission probabilities
           i.e.  B[y] = [(j, P(y | j)), ...]
"""

import numpy as np


# ============================================================
# Helper
# ============================================================

def dict_argmax(d):
    """
    Return the (max_value, key_of_max) pair from a dictionary.

    Parameters
    ----------
    d : dict {key: float}

    Returns
    -------
    max_val : float
    best_key : hashable
    """
    best_key = None
    max_val  = float("-inf")
    for key, val in d.items():
        if val > max_val:
            max_val  = val
            best_key = key
    return max_val, best_key


# ============================================================
# Sparse Viterbi step
# ============================================================

def disagg_algo(hmm, y):
    """
    Perform one step of the sparse Viterbi algorithm.

    Parameters
    ----------
    hmm : object
        Super-State HMM with sparse attributes (see module docstring).
    y : sequence of two ints [y0, y1]
        Two consecutive quantised power observations.

    Returns
    -------
    p : float
        Maximum probability at time t=1.
    k : int
        Index of the most probable super-state at t=1.
    Pt1 : dict {state_index: probability}
        Sparse probability dictionary at t=1.
    cdone : list of two ints [n_init_done, n_trans_done]
        Number of computations actually performed.
    ctotal : list of two ints [K, K²]
        Total possible computations (for efficiency reporting).
    """
    K  = int(hmm.K)
    P0 = np.asarray(hmm.P0, dtype=float)

    y0, y1 = int(y[0]), int(y[1])

    ctotal = [K, K * K]
    n_init_done  = 0
    n_trans_done = 0

    # ---- t = 0 : sparse initialisation ----
    Pt0 = {}
    for j, p_b in hmm.B[y0]:
        if P0[j] != 0.0:
            Pt0[j] = P0[j] * p_b
            n_init_done += 1

    # ---- t = 1 : sparse recursion ----
    Pt1 = {}
    for j, p_b in hmm.B[y1]:
        best = 0.0
        for i, p_a in hmm.A[j]:
            if i in Pt0:
                val = Pt0[i] * p_a
                if val > best:
                    best = val
                n_trans_done += 1
        if best > 0.0:
            Pt1[j] = best * p_b

    cdone = [n_init_done, n_trans_done]

    # ---- select best state ----
    if Pt1:
        p, k = dict_argmax(Pt1)
    else:
        # Fall back to the highest-probability t=0 state
        if Pt0:
            p, k = dict_argmax(Pt0)
        else:
            p, k = 0.0, 0

    return p, k, Pt1, cdone, ctotal
