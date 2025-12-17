
# formation_main.py
# Main script (Linux-friendly): reads JSON mappings, uses H∞ controller + geometry utils, and can run demo viz.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
import json, os
import numpy as np

from formation_hinf import solve_hinf_state_feedback
from formation_geom import centroid, rot2d, template_centered, targets_global, find_best_rotation_and_assignment

# ---- Mapping loader kept here (you can move it to a separate module if desired) ----
class TemplateInfo:
    def __init__(self, g6: str, positions: np.ndarray, n: int):
        self.g6 = g6
        self.positions = positions
        self.n = n

class MothersMappings:
    def __init__(self, nodes_by_n, order_by_n, step1, skip2, edges_by_g6):
        self.nodes_by_n = nodes_by_n
        self.order_by_n = order_by_n
        self.step1 = step1
        self.skip2 = skip2
        self.edges_by_g6 = edges_by_g6

    @staticmethod
    def load(path: str) -> "MothersMappings":
        with open(path, "r") as f:
            raw = json.load(f)
        nodes_by_n = {}
        order_by_n = {}
        for n_str, lst in raw["nodes_by_n"].items():
            n = int(n_str)
            nodes_by_n[n] = {}
            order_by_n[n] = []
            for rec in lst:
                g6 = rec["g6"]
                pos_dict = rec["pos"]
                idxs = sorted((int(k) for k in pos_dict.keys()))
                pts = np.array([pos_dict[str(i)] for i in idxs], dtype=float)
                nodes_by_n[n][g6] = TemplateInfo(g6=g6, positions=pts, n=n)
                order_by_n[n].append(g6)
        edges = raw.get("edges", {})
        step1 = edges.get("step1", [])
        skip2 = edges.get("skip2", [])
        edges_by_template = raw.get("edges_by_template", {})
        return MothersMappings(nodes_by_n, order_by_n, step1, skip2, edges_by_template)

    def any_g6_for_n(self, n: int):
        if n not in self.order_by_n or not self.order_by_n[n]:
            raise KeyError(f"No template list for n={n}")
        return list(self.order_by_n[n])

    def first_g6_for_n(self, n: int) -> str:
        return self.any_g6_for_n(n)[0]

    def get_template(self, g6: str) -> TemplateInfo:
        for d in self.nodes_by_n.values():
            if g6 in d: return d[g6]
        raise KeyError(f"g6 '{g6}' not found")

    def next_g6(self, current_g6: str, deleted_indices: Tuple[int, ...]) -> Optional[str]:
        arr = self.step1 if len(deleted_indices)==1 else self.skip2 if len(deleted_indices)==2 else None
        if arr is None: 
            raise ValueError("Only 1 or 2 deletions supported.")
        for e in arr:
            if e["parent_g6"] == current_g6 and tuple(e["deleted"]) == tuple(deleted_indices):
                return e["child_g6"]
        return None

# ---- OuterFormationController (uses hinf + geom modules) ----
class OuterFormationController:
    def __init__(self, mappings: MothersMappings, scale=10.0, alpha=1.0, arrive_eps=0.25, hold_seconds=2.0,
                 v_max=0.8, q1=10.0, q2=0.3, r_u=0.3, gamma_init=1.0, anti_windup_freeze=True):
        self.mappings = mappings
        self.scale = float(scale); self.alpha=float(alpha); self.arrive_eps=float(arrive_eps)
        self.hold_seconds=float(hold_seconds); self.v_max=float(v_max)
        self.q1=float(q1); self.q2=float(q2); self.r_u=float(r_u); self.gamma_init=float(gamma_init)
        self.anti_windup_freeze = bool(anti_windup_freeze)
        self.current_g6 = None
        self.assign_node_of_agent = {}
        self.p_ref = {}
        self.last_alive = None
        self.center_world = np.zeros(2, dtype=float)
        self.hold_until = 0.0
        self.theta_opt = 0.0
        self.Kx = np.zeros((1,2)); self.Ky = np.zeros((1,2))
        self.int_e = {}

    # ---- H-infinity gains ----
    def _init_hinf_gains(self):
        A = np.array([[0., 0.],
                      [1., 0.]], dtype=float)
        B = np.array([[-1.],
                      [ 0.]], dtype=float)
        E = np.array([[ 1., -1.],
                      [ 0.,  0.]], dtype=float)
        Q = np.diag([self.q1, self.q2])
        R = np.array([[self.r_u]], dtype=float)
        Kx, _ = solve_hinf_state_feedback(A, B, E, Q, R, gamma_init=self.gamma_init)
        Ky, _ = solve_hinf_state_feedback(A, B, E, Q, R, gamma_init=self.gamma_init)
        self.Kx = Kx.reshape(1,2); self.Ky = Ky.reshape(1,2)

    # ---- Template helpers using formation_geom ----
    def _template_centered(self, g6: str) -> np.ndarray:
        T = self.mappings.get_template(g6)
        return template_centered(T.positions)

    def _targets_global(self, g6: str, theta: float, center: np.ndarray) -> np.ndarray:
        T = self.mappings.get_template(g6)
        return targets_global(T.positions, theta, center, self.scale)

    def _find_best_rotation_and_assignment(self, P_world: Dict[int, np.ndarray],
                                           g6: str, center: np.ndarray):
        T = self.mappings.get_template(g6)
        return find_best_rotation_and_assignment(P_world, T.positions, center, self.scale)

    # ---- init / deletion detect / arrived ----
    def initialize(self, P_world_init: Dict[int, Iterable[float]], initial_g6: str):
        P_world = {i: np.asarray(p, dtype=float) for i, p in P_world_init.items()}
        self.current_g6 = initial_g6
        self.center_world = np.mean(np.stack(list(P_world.values()), axis=0), axis=0)
        theta, assign = self._find_best_rotation_and_assignment(P_world, self.current_g6, self.center_world)
        self.theta_opt = theta; self.assign_node_of_agent = assign
        self.p_ref = {i: P_world[i].copy() for i in P_world.keys()}
        self.last_alive = set(P_world.keys())
        self.hold_until = self.hold_seconds
        self._init_hinf_gains()
        self.int_e = {i: np.zeros(2) for i in P_world.keys()}

    def _detect_deletions(self, alive_flags: Dict[int, bool]):
        curr_alive = {i for i, ok in alive_flags.items() if ok}
        if self.last_dependency_unset():
            self.last_alive = curr_alive
            return False, tuple()
        lost = sorted(list(self.last_alive - curr_alive))
        self.last_alive = curr_alive
        if not lost: return False, tuple()
        if len(lost) > 2: lost = lost[:2]
        deleted_nodes = []
        for aid in lost:
            if aid in self.assign_node_of_agent:
                deleted_nodes.append(self.assign_node_of_agent[aid])
        if not deleted_nodes: return False, tuple()
        return True, tuple(sorted(deleted_nodes))

    def last_dependency_unset(self):
        return self.last_alive is None

    def _all_arrived(self, P_world: Dict[int, np.ndarray], Q: np.ndarray) -> bool:
        for i, p in P_world.items():
            j = self.assign_node_of_agent.get(i, None)
            if j is None: return False
            if np.linalg.norm(p - Q[j]) > self.arrive_eps:
                return False
        return True

    # ---- main step ----
    def step(self, dt: float, P_world_meas: Dict[int, Iterable[float]],
             alive_flags: Dict[int, bool], now: float) -> Dict[int, np.ndarray]:
        P_world = {i: np.asarray(p, dtype=float)
                   for i, p in P_world_meas.items() if alive_flags.get(i, False)}

        Q = self._targets_global(self.current_g6, self.theta_opt, self.center_world)

        if now >= self.hold_until and self._all_arrived(P_world, Q):
            self.hold_until = now + self.hold_seconds

        need_switch, deleted_indices = self._detect_deletions(alive_flags)
        if need_switch:
            self.hold_until = now
            next_g6 = self.mappings.next_g6(self.current_g6, deleted_indices)
            if next_g6 is None:
                n_alive = len(P_world)
                next_g6 = self.mappings.first_g6_for_n(n_alive)
            self.current_g6 = next_g6
            self.center_world = np.mean(np.stack(list(P_world.values()), axis=0), axis=0)
            self.theta_opt, self.assign_node_of_agent = self._find_best_rotation_and_assignment(
                P_world, self.current_g6, self.center_world
            )
            Q = self._targets_global(self.current_g6, self.theta_opt, self.center_world)

        for i in P_world:
            j = self.assign_node_of_agent.get(i, None)
            if j is None or j >= Q.shape[0]:
                assigned = set(self.assign_node_of_agent.values())
                free_nodes = [k for k in range(Q.shape[0]) if k not in assigned]
                if free_nodes:
                    d = [np.linalg.norm(P_world[i] - Q[k]) for k in free_nodes]
                    j = free_nodes[int(np.argmin(d))]
                    self.assign_node_of_agent[i] = j
                else:
                    continue
            pref = self.p_ref.get(i, P_world[i].copy())
            self.p_ref[i] = pref + self.alpha * (Q[j] - pref) * dt

        Vcmd_world: Dict[int, np.ndarray] = {}
        if now < self.hold_until:
            for i in P_world:
                Vcmd_world[i] = np.zeros(2)
        else:
            for i in P_world:
                e = self.p_ref[i] - P_world[i]
                z = self.int_e.get(i, np.zeros(2))
                ux = float(- self.Kx @ np.array([e[0], z[0]]))
                uy = float(- self.Ky @ np.array([e[1], z[1]]))
                v_world = np.array([ux, uy], dtype=float)
                nrm = float(np.linalg.norm(v_world)); saturated=False
                if nrm > self.v_max and nrm > 1e-12:
                    v_world = v_world * (self.v_max / nrm); saturated=True
                dz = np.zeros(2) if (self.anti_windup_freeze and saturated) else e
                # Python doesn't have &&; fix:
                if self.anti_windup_freeze and saturated:
                    dz = np.zeros(2)
                else:
                    dz = e
                self.int_e[i] = z + dz * dt
                Vcmd_world[i] = v_world
        return Vcmd_world

# ---- Optional demo entrypoint (uses formation_viz) ----
def run_demo(json_path: str, seed: Optional[int]=None):
    from formation_viz import run_live_demo
    mappings = MothersMappings.load(json_path)
    ctrl = OuterFormationController(mappings,
                                    scale=10.0, alpha=1.0, arrive_eps=0.25, hold_seconds=2.0,
                                    v_max=0.8, q1=10.0, q2=0.3, r_u=0.3, gamma_init=1.0,
                                    anti_windup_freeze=True)
    run_live_demo(mappings, ctrl, json_path, random_seed=seed)

if __name__ == "__main__":
    # Example usage on Linux:
    #   python3 formation_main.py /path/to/mothers_mappings.json
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 formation_main.py /path/to/mothers_mappings.json [seed]")
        sys.exit(1)
    json_path = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_demo(json_path, seed)
