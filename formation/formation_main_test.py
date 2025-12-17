#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
formation_main_test.py
适配只包含 3/4 点模板的 test.json：
- 自动选择可用的最大 n 作为起始模板
- 无 edges 信息也能画（默认连边）
- 继续复用 formation_hinf / formation_geom，控制仍为“闭环全局坐标 + 最优旋转+Hungarian 指派”
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Iterable, List
from dataclasses import dataclass
import json, sys, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from formation_hinf import solve_hinf_state_feedback
from formation_geom import (
    centroid, rot2d, template_centered, targets_global,
    find_best_rotation_and_assignment
)

# ----------------- 轻量映射装载（兼容 test.json 的精简结构） -----------------
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

        # 假定 test.json 仍然是 { "nodes_by_n": { "3":[...], "4":[...] }, ... } 的“同构/子集”结构
        nodes_by_n = {}
        order_by_n = {}
        for n_str, lst in raw["nodes_by_n"].items():
            n = int(n_str)
            nodes_by_n[n] = {}
            order_by_n[n] = []
            for rec in lst:
                g6 = rec.get("g6", f"N{n}_{len(order_by_n[n])}")
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

    def ns_available(self) -> List[int]:
        return sorted([n for n, lst in self.order_by_n.items() if lst])

    def any_g6_for_n(self, n: int) -> List[str]:
        if n not in self.order_by_n or not self.order_by_n[n]:
            raise KeyError(f"No template list for n={n}")
        return list(self.order_by_n[n])

    def first_g6_for_n(self, n: int) -> str:
        return self.any_g6_for_n(n)[0]

    def get_template(self, g6: str) -> TemplateInfo:
        for d in self.nodes_by_n.values():
            if g6 in d: return d[g6]
        raise KeyError(f"g6 '{g6}' not found")

# ----------------- 外环控制器（与之前一致，精简注释） -----------------
class OuterFormationController:
    def __init__(self, mappings: MothersMappings,
                 scale=10.0, alpha=1.0, arrive_eps=0.20, hold_seconds=1.0,
                 v_max=0.8, q1=10.0, q2=0.3, r_u=0.3, gamma_init=1.0, anti_windup_freeze=True):
        self.mappings = mappings
        self.scale=float(scale); self.alpha=float(alpha); self.arrive_eps=float(arrive_eps)
        self.hold_seconds=float(hold_seconds); self.v_max=float(v_max)
        self.q1=float(q1); self.q2=float(q2); self.r_u=float(r_u); self.gamma_init=float(gamma_init)
        self.anti_windup_freeze=bool(anti_windup_freeze)

        self.current_g6=None
        self.assign_node_of_agent={}
        self.p_ref={}
        self.center_world=np.zeros(2)
        self.hold_until=0.0
        self.theta_opt=0.0
        self.Kx=np.zeros((1,2)); self.Ky=np.zeros((1,2))
        self.int_e={}
        self.last_alive=None

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

    def _targets_global(self, g6: str, theta: float, center: np.ndarray) -> np.ndarray:
        T = self.mappings.get_template(g6)
        return targets_global(T.positions, theta, center, self.scale)

    def _find_best_rotation_and_assignment(self, P_world: Dict[int, np.ndarray],
                                           g6: str, center: np.ndarray):
        T = self.mappings.get_template(g6)
        return find_best_rotation_and_assignment(P_world, T.positions, center, self.scale)

    def initialize(self, P_world_init: Dict[int, Iterable[float]], initial_g6: str):
        P_world = {i: np.asarray(p, dtype=float) for i,p in P_world_init.items()}
        self.current_g6 = initial_g6
        self.center_world = centroid(np.stack(list(P_world.values()), axis=0))
        theta, assign = self._find_best_rotation_and_assignment(P_world, self.current_g6, self.center_world)
        self.theta_opt = theta; self.assign_node_of_agent = assign
        self.p_ref = {i: P_world[i].copy() for i in P_world.keys()}
        self._init_hinf_gains()
        self.int_e = {i: np.zeros(2) for i in P_world.keys()}
        self.last_alive = set(P_world.keys())
        self.hold_until = self.hold_seconds

    def _all_arrived(self, P_world: Dict[int,np.ndarray], Q: np.ndarray) -> bool:
        for i,p in P_world.items():
            j = self.assign_node_of_agent.get(i, None)
            if j is None: return False
            if np.linalg.norm(p - Q[j]) > self.arrive_eps: return False
        return True

    def step(self, dt: float, P_world_meas: Dict[int, Iterable[float]],
             alive_flags: Dict[int,bool], now: float) -> Dict[int,np.ndarray]:
        P_world = {i: np.asarray(p, dtype=float)
                   for i,p in P_world_meas.items() if alive_flags.get(i, False)}
        Q = self._targets_global(self.current_g6, self.theta_opt, self.center_world)

        if now >= self.hold_until and self._all_arrived(P_world, Q):
            self.hold_until = now + self.hold_seconds

        # 参考一阶滤波
        for i in P_world:
            j = self.assign_node_of_agent.get(i, None)
            if j is None: continue
            pref = self.p_ref.get(i, P_world[i].copy())
            self.p_ref[i] = pref + self.alpha * (Q[j] - pref) * dt

        Vcmd = {}
        if now < self.hold_until:
            for i in P_world: Vcmd[i] = np.zeros(2)
        else:
            for i in P_world:
                e = self.p_ref[i] - P_world[i]
                z = self.int_e.get(i, np.zeros(2))
                ux = float(- self.Kx @ np.array([e[0], z[0]]))
                uy = float(- self.Ky @ np.array([e[1], z[1]]))
                v = np.array([ux, uy], dtype=float)
                nrm = float(np.linalg.norm(v)); saturated=False
                if nrm > self.v_max and nrm > 1e-12:
                    v = v * (self.v_max / nrm); saturated=True
                if self.anti_windup_freeze and saturated:
                    dz = np.zeros(2)
                else:
                    dz = e
                self.int_e[i] = z + dz * dt
                Vcmd[i] = v
        return Vcmd

# ----------------- 简易可视化（不依赖 formation_viz，动态选 n） -----------------
def _default_edges(points: np.ndarray) -> List[tuple]:
    # 环 + 最近邻
    n = points.shape[0]
    # 极角排个环
    c = points.mean(axis=0)
    ang = np.arctan2(points[:,1]-c[1], points[:,0]-c[0])
    order = np.argsort(ang)
    E = [(int(order[k]), int(order[(k+1)%n])) for k in range(n)]
    # 再加每点的最近邻
    for i in range(n):
        d = np.linalg.norm(points - points[i], axis=1); d[i]=1e9
        j = int(np.argmin(d))
        if (i,j) not in E and (j,i) not in E: E.append((i,j))
    return E

def run_demo_dynamic(json_path: str, seed: Optional[int]=None):
    rng = np.random.default_rng(seed)
    mappings = MothersMappings.load(json_path)
    ns = mappings.ns_available()
    if not ns: raise RuntimeError("test.json 中没有任何模板")
    start_n = max(ns)                  # 用可用的最大 n 作为起点（通常是 4）
    g6 = mappings.first_g6_for_n(start_n)
    T0 = mappings.get_template(g6)
    centered = template_centered(T0.positions)
    scale = 10.0

    ctrl = OuterFormationController(mappings, scale=scale, alpha=1.0, arrive_eps=0.2, hold_seconds=1.0)

    # 初始：把无人机随机放在模板附近
    Q0 = centered * scale
    P_world = {i+1: Q0[i].copy() + rng.normal(scale=1.0, size=2) for i in range(Q0.shape[0])}
    alive = {i: True for i in P_world.keys()}
    ctrl.initialize(P_world, g6)

    pad = 8.0
    all_pts = np.stack(list(P_world.values()), axis=0)
    lo = all_pts.min(axis=0) - pad
    hi = all_pts.max(axis=0) + pad

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_title(f"Live formation (Global closed-loop; init n={start_n})")

    scat_targets = ax.scatter([], [], marker='x')
    scat_agents = ax.scatter([], [])
    lines: List = []
    text_elems: List = []
    dt = 0.05
    state = {"t": 0.0, "delete_cooldown": 0.0}

    def draw_edges(Q: np.ndarray):
        nonlocal lines
        for ln in lines: ln.remove()
        lines.clear()
        edges = _default_edges(Q)
        for (u,v) in edges:
            ln, = ax.plot([Q[u,0], Q[v,0]], [Q[u,1], Q[v,1]])
            lines.append(ln)

    def animate(_):
        state["t"] += dt

        # 控制一步
        V = ctrl.step(dt, P_world, alive, state["t"])
        for i in list(P_world.keys()):
            if alive[i] and i in V: P_world[i] = P_world[i] + dt * V[i]

        # 当前目标
        Q = ctrl._targets_global(ctrl.current_g6, ctrl.theta_opt, ctrl.center_world)

        # 到达 + 冷却 -> 删一台（直到 3）
        alive_ids = [i for i in sorted(P_world.keys()) if alive[i]]
        arrived = ctrl._all_arrived({i: P_world[i] for i in alive_ids}, Q)

        if len(alive_ids) > 3:
            if arrived:
                state["delete_cooldown"] = max(0.0, state["delete_cooldown"] - dt) if state["delete_cooldown"]>0 else ctrl.hold_seconds
                if state["delete_cooldown"] == 0.0:
                    vid = int(rng.choice(alive_ids))
                    alive[vid] = False
                    # 删完后：重新以当前活跃机的几何中心重算角度/指派（保持同一 g6 或换成 3 点 g6）
                    n_alive = len([1 for i in alive if alive[i]])
                    target_n = max(n for n in mappings.ns_available() if n <= n_alive)
                    new_g6 = mappings.first_g6_for_n(target_n)
                    ctrl.current_g6 = new_g6
                    ctrl.center_world = centroid(np.stack([P_world[i] for i in alive_ids if alive[i]], axis=0))
                    th, assign = ctrl._find_best_rotation_and_assignment(
                        {i: P_world[i] for i in alive_ids if alive[i]}, ctrl.current_g6, ctrl.center_world
                    )
                    ctrl.theta_opt = th; ctrl.assign_node_of_agent = assign
            else:
                state["delete_cooldown"] = 0.0

        # 画面
        scat_targets.set_offsets(Q); draw_edges(Q)
        for t in text_elems: t.remove()
        text_elems.clear()
        XY = np.array([P_world[i] for i in alive_ids])
        scat_agents.set_offsets(XY)
        for idx,i in enumerate(alive_ids):
            text_elems.append(ax.text(XY[idx,0], XY[idx,1], str(i), fontsize=8))

        # 终止：3 台到达
        if len(alive_ids) == 3 and arrived:
            anim.event_source.stop()

        return scat_targets, *lines, scat_agents, *text_elems

    anim = FuncAnimation(fig, animate, interval=int(dt*1000), blit=False, cache_frame_data=False)
    plt.show()

# ----------------- CLI -----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 formation_main_test.py ./test.json [seed]")
        sys.exit(1)
    json_path = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_demo_dynamic(json_path, seed)
