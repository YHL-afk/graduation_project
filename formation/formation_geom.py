
# formation_geom.py
# Geometry utilities: centroid, rotation, Hungarian assignment, best-rotation matching

from __future__ import annotations
from typing import Dict, Iterable, Tuple, List
import numpy as np
import math

def centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)

def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)

def hungarian(cost: np.ndarray) -> List[int]:
    a = cost.astype(float).copy()
    n_rows, n_cols = a.shape
    n = max(n_rows, n_cols)
    if n_rows != n_cols:
        pad = np.zeros((n, n))
        pad[:n_rows, :n_cols] = a
        a = pad
    else:
        n = n_rows
    u = np.zeros(n + 1); v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int); way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = np.full(n + 1, np.inf); used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True; i0 = p[j0]
            delta = np.inf; j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    assign = [-1] * n_rows
    for j in range(1, n + 1):
        i = p[j] - 1
        if i < n_rows and j - 1 < n_cols:
            assign[i] = j - 1
    return assign

def template_centered(template_positions: np.ndarray) -> np.ndarray:
    return template_positions - centroid(template_positions)

def targets_global(template_positions: np.ndarray, theta: float, center: np.ndarray, scale: float) -> np.ndarray:
    centered = template_centered(template_positions)
    R = rot2d(theta)
    Q_rel = (R @ centered.T).T * scale
    return Q_rel + center[None, :]

def assign_cost_for_theta(P_world: Dict[int, np.ndarray],
                          template_positions: np.ndarray,
                          theta: float, center: np.ndarray, scale: float) -> Tuple[float, Dict[int, int]]:
    ids = sorted(P_world.keys())
    Q = targets_global(template_positions, theta, center, scale)
    C = np.zeros((len(ids), Q.shape[0]))
    for r, aid in enumerate(ids):
        d = Q - P_world[aid]
        C[r, :] = (d * d).sum(axis=1)
    cols = hungarian(C)
    total = 0.0
    for r, aid in enumerate(ids):
        j = cols[r]
        total += float(np.linalg.norm(Q[j] - P_world[aid]))
    return total, {ids[r]: cols[r] for r in range(len(ids))}

def find_best_rotation_and_assignment(P_world: Dict[int, np.ndarray],
                                      template_positions: np.ndarray,
                                      center: np.ndarray, scale: float,
                                      coarse_steps: int = 360,
                                      refine_half_window_deg: float = 5,
                                      refine_steps: int = 181) -> Tuple[float, Dict[int, int]]:
    thetas = np.linspace(0.0, 2 * math.pi, num=coarse_steps, endpoint=False)
    best_cost, best_theta, best_assign = float('inf'), 0.0, {}
    for th in thetas:
        cost, assign = assign_cost_for_theta(P_world, template_positions, th, center, scale)
        if cost < best_cost:
            best_cost, best_theta, best_assign = cost, th, assign
    th0 = best_theta
    win = math.radians(refine_half_window_deg)
    thetas_ref = np.linspace(th0 - win, th0 + win, num=refine_steps)
    for th in thetas_ref:
        thn = (th + 2 * math.pi) % (2 * math.pi)
        cost, assign = assign_cost_for_theta(P_world, template_positions, thn, center, scale)
        if cost < best_cost:
            best_cost, best_theta, best_assign = cost, thn, assign
    return best_theta, best_assign
