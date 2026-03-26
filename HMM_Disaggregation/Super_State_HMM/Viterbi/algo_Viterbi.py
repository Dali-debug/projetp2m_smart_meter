"""
algo_Viterbi.py — Classic (dense) Viterbi step for SSHMM disaggregation.

Implements:
  disagg_algo(hmm, y)

given two consecutive observations y = [y0, y1], returns the most probable
next super-state together with performance counters.

Interface contract
------------------
The *hmm* object must expose:
  hmm.K  : int            — total number of super-states
  hmm.P0 : array-like     — initial distribution, shape (K,)
  hmm.A  : 2-D array-like — transition matrix, shape (K, K)
  hmm.B  : 2-D array-like — emission matrix B[j, y] = P(observe y | state j)
"""

import numpy as np


def disagg_algo(hmm, y):
    """
    Perform one step of the classic dense Viterbi algorithm.

    Parameters
    ----------
    hmm : object
        Super-State HMM with attributes K, P0, A, B (see module docstring).
    y : sequence of two ints [y0, y1]
        Two consecutive quantised power observations.

    Returns
    -------
    p : float
        Maximum probability at time t=1.
    k : int
        Index of the most probable super-state at t=1.
    Pt1 : np.ndarray, shape (K,)
        Full probability vector at t=1.
    cdone : list of two ints [n_init_done, n_trans_done]
        Number of computations actually performed (same as ctotal here).
    ctotal : list of two ints [K, K²]
        Total number of possible computations.
    """
    K  = int(hmm.K)
    P0 = np.asarray(hmm.P0, dtype=float)
    A  = np.asarray(hmm.A,  dtype=float)
    B  = np.asarray(hmm.B,  dtype=float)

    y0, y1 = int(y[0]), int(y[1])

    # ---- t = 0 : initialise ----
    Pt0 = np.empty(K, dtype=float)
    for j in range(K):
        Pt0[j] = P0[j] * B[j, y0]

    # ---- t = 1 : one-step Viterbi recursion ----
    Pt1 = np.empty(K, dtype=float)
    for j in range(K):
        b_val  = B[j, y1]
        best   = 0.0
        for i in range(K):
            val = Pt0[i] * A[i, j]
            if val > best:
                best = val
        Pt1[j] = best * b_val

    # ---- select best state ----
    k = int(np.argmax(Pt1))
    p = float(Pt1[k])

    # ---- performance counters ----
    cdone  = [K, K * K]   # all computations performed (dense)
    ctotal = [K, K * K]

    return p, k, Pt1, cdone, ctotal
