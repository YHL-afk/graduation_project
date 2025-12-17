
# formation_viz.py
# Visualization for live formation demo (uses a controller with .step and ._targets_global)

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def _sort_by_angle(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    ang = np.arctan2(points[:,1]-c[1], points[:,0]-c[0])
    return np.argsort(ang)

def _default_edges(points: np.ndarray) -> List[Tuple[int,int]]:
    n = points.shape[0]
    order = _sort_by_angle(points)
    E = [(int(order[k]), int(order[(k+1)%n])) for k in range(n)]
    for i in range(n):
        d = np.linalg.norm(points - points[i], axis=1)
        d[i] = 1e9
        j = int(np.argmin(d))
        if (i,j) not in E and (j,i) not in E:
            E.append((i,j))
    return E

def run_live_demo(mappings, controller, json_path: str, random_seed: Optional[int]=None,
                  use_edges_from_mapping: bool=True):
    rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng()

    g6_list = mappings.any_g6_for_n(7)
    init_g6 = str(rng.choice(g6_list))
    T0 = mappings.get_template(init_g6)
    centered = T0.positions - T0.positions.mean(axis=0)
    Q0 = centered * controller.scale

    P_world = {i+1: Q0[i].copy() + rng.normal(scale=1.0, size=2) for i in range(Q0.shape[0])}
    alive = {i: True for i in P_world.keys()}

    controller.initialize(P_world_init=P_world, initial_g6=init_g6)

    pad = 20.0
    all_pts = np.stack(list(P_world.values()), axis=0)
    xmin, ymin = (all_pts.min(axis=0) - pad)
    xmax, ymax = (all_pts.max(axis=0) + pad)

    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Live formation (Global closed-loop, best-rotation match, init: {init_g6})")

    scat_targets = ax.scatter([], [], marker='x')
    lines: List = []
    scat_agents = ax.scatter([], [])
    text_elems: List = []

    dt = 0.05
    state = {"t": 0.0, "delete_cooldown": 0.0}

    def draw_edges(Q: np.ndarray, g6: str):
        nonlocal lines
        for ln in lines: ln.remove()
        lines.clear()
        edges_raw = mappings.edges_by_g6.get(g6, None) if use_edges_from_mapping else None
        if edges_raw:
            edges = [(int(u), int(v)) for u,v in edges_raw]
        else:
            edges = _default_edges(Q)
            edges = _default_edges(Q)
        for (u,v) in edges:
            ln, = ax.plot([Q[u,0], Q[v,0]], [Q[u,1], Q[v,1]])
            lines.append(ln)

    def animate(_frame):
        state["t"] += dt

        Vcmd_world = controller.step(dt, P_world, alive, state["t"])

        for i in list(P_world.keys()):
            if alive[i] and i in Vcmd_world:
                P_world[i] = P_world[i] + dt * Vcmd_world[i]

        Q = controller._targets_global(controller.current_g6, controller.theta_opt, controller.center_world)

        alive_ids = [i for i in sorted(P_world.keys()) if alive.get(i, False)]
        arrived = controller._all_arrived({i: P_world[i] for i in alive_ids}, Q)

        if len(alive_ids) > 4:
            if arrived:
                if state["delete_cooldown"] <= 0.0:
                    state["delete_cooldown"] = controller.hold_seconds
                else:
                    state["delete_cooldown"] = max(0.0, state["delete_cooldown"] - dt)
                if state["delete_cooldown"] == 0.0:
                    import numpy as _np
                    vid = int(_np.random.choice(alive_ids))
                    alive[vid] = False
            else:
                state["delete_cooldown"] = 0.0

        scat_targets.set_offsets(Q)
        draw_edges(Q, controller.current_g6)

        for t in text_elems: t.remove()
        text_elems.clear()
        XY_world = np.array([P_world[i] for i in alive_ids]) if alive_ids else np.zeros((0,2))
        scat_agents.set_offsets(XY_world)
        for idx, i in enumerate(alive_ids):
            text_elems.append(ax.text(XY_world[idx,0], XY_world[idx,1], str(i), fontsize=8))

        if len(alive_ids) == 4 and arrived:
            anim.event_source.stop()

        return scat_targets, *lines, scat_agents, *text_elems

    anim = FuncAnimation(fig, animate, interval=int(dt*1000), blit=False, cache_frame_data=False)
    plt.show()
