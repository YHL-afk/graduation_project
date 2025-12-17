
# formation_hinf.py
# H-infinity state-feedback solver utilities

from __future__ import annotations
import numpy as np

def _is_psd(M: np.ndarray, tol: float = 1e-10) -> bool:
    if not np.allclose(M, M.T, atol=1e-9): 
        return False
    w = np.linalg.eigvalsh(M)
    return np.all(w >= -tol)

def _care_via_hamiltonian(A: np.ndarray, G: np.ndarray, Q: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    H = np.block([[A, -G], [-Q, -A.T]])
    w, V = np.linalg.eig(H)
    idx = np.argsort(np.real(w))
    W = V[:, idx]
    sel = np.where(np.real(w[idx]) < 0)[0]
    if len(sel) < n:
        sel = np.arange(n)
    Vsel = W[:, sel[:n]]
    V1 = Vsel[:n, :]
    V2 = Vsel[n:, :]
    X = np.real(V2 @ np.linalg.inv(V1))
    X = 0.5 * (X + X.T)
    return X

def solve_hinf_state_feedback(A: np.ndarray, B: np.ndarray, E: np.ndarray,
                              Q: np.ndarray, R: np.ndarray,
                              gamma_init: float = 1.0, max_trials: int = 20, grow: float = 1.5):
    """Solve for K in u = -K x for the generalized H∞ problem with disturbance input E w.
    Returns (K, gamma_used).
    """
    Rinv = np.linalg.inv(R)
    gamma = gamma_init
    for _ in range(max_trials):
        G = B @ Rinv @ B.T - (1.0 / (gamma ** 2)) * (E @ E.T)
        if _is_psd(G):
            w, V = np.linalg.eigh((G + G.T) / 2.0)
            w_clipped = np.clip(w, 0.0, None)
            L = V @ np.diag(np.sqrt(w_clipped)) @ V.T
            X = _care_via_hamiltonian(A, L @ L.T, Q)
            K = Rinv @ B.T @ X
            return K, gamma
        gamma *= grow
    # Fallback
    G = B @ Rinv @ B.T
    X = _care_via_hamiltonian(A, G, Q)
    K = Rinv @ B.T @ X
    return K, float('inf')
