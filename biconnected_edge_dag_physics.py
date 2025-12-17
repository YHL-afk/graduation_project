#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, sys, math, random, json, os, time
import numpy as np
import networkx as nx
from networkx.generators.atlas import graph_atlas_g
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from multiprocessing import Pool, cpu_count, get_start_method

# ========= 基础：取 n 点 2-连通（无割点）非同构图 =========
def get_unlabeled_biconnected(n: int):
    if n > 7:
        print("[Note] graph_atlas_g only contains graphs with ≤7 vertices.", file=sys.stderr)
        return []
    out = []
    for G in graph_atlas_g():
        if G.number_of_nodes() != n:
            continue
        if nx.is_biconnected(G):
            out.append(G)
    return out

def graph_id_g6(G: nx.Graph) -> str:
    return nx.to_graph6_bytes(G, header=False).decode().strip()

def degree_sequence(G: nx.Graph):
    return tuple(sorted((d for _, d in G.degree()), reverse=True))

# ========= 小工具：JSON 键转 int、安全读 pos =========
def _json_keys_to_int(d):
    out = {}
    for k, v in d.items():
        try:
            ik = int(k)
        except Exception:
            ik = k
        out[ik] = v
    return out

def _pos_from_json(obj):
    """obj: {node(str/int): [x,y]}, 返回 {node(int): np.array([x,y])}"""
    if obj is None:
        return None
    d = _json_keys_to_int(obj) if isinstance(obj, dict) else {}
    return {k: np.array(v, dtype=float) for k, v in d.items()}

# ========= UDG 工具：归一化 / 验收 / 细化 =========
def _scale_max_edge_to_one(G: nx.Graph, pos: dict):
    nodes = list(G.nodes())
    P = np.array([pos[v] for v in nodes], dtype=float)
    max_e = 0.0
    for (u, v) in G.edges():
        duv = np.linalg.norm(P[nodes.index(u)] - P[nodes.index(v)])
        if duv > max_e:
            max_e = duv
    if max_e <= 0:
        return None, 0.0
    P /= max_e
    return {nodes[i]: P[i] for i in range(len(nodes))}, max_e

def check_udg(G: nx.Graph, pos: dict, tol=1e-6):
    pos1, _ = _scale_max_edge_to_one(G, pos)
    if pos1 is None:
        return None
    nodes = list(G.nodes())
    P = np.array([pos1[v] for v in nodes], dtype=float)
    thr = 1.0 + tol
    for (u, v) in nx.non_edges(G):
        iu, iv = nodes.index(u), nodes.index(v)
        d = np.linalg.norm(P[iu] - P[iv])
        if d <= thr:
            return None
    return pos1  # 通过

def refine_with_margin(G: nx.Graph, pos: dict, tol=1e-6, steps=200, step=0.01, every=20):
    nodes = list(G.nodes())
    P = np.array([pos[v] for v in nodes], dtype=float)

    def renorm():
        max_e = 0.0
        for (u, v) in G.edges():
            iu, iv = nodes.index(u), nodes.index(v)
            d = np.linalg.norm(P[iu] - P[iv])
            if d > max_e:
                max_e = d
        if max_e > 0:
            P[:] /= max_e

    renorm()
    nonE = list(nx.non_edges(G))
    E = list(G.edges())
    for t in range(steps):
        F = np.zeros_like(P)
        # 过短的非边：强排斥
        for (u, v) in nonE:
            iu, iv = nodes.index(u), nodes.index(v)
            d = np.linalg.norm(P[iu] - P[iv]) + 1e-12
            if d <= 1.0 + tol:
                dir = (P[iu] - P[iv]) / d
                f = (1.0 + tol - d) * 2.5
                F[iu] += dir * f
                F[iv] -= dir * f
        # 过长的边：轻微吸引
        for (u, v) in E:
            iu, iv = nodes.index(u), nodes.index(v)
            d = np.linalg.norm(P[iu] - P[iv]) + 1e-12
            if d > 1.0:
                dir = (P[iv] - P[iu]) / d
                f = (d - 1.0) * 1.5
                F[iu] += dir * f
                F[iv] -= dir * f
        P += step * F
        if (t + 1) % every == 0:
            renorm()
    renorm()
    return {nodes[i]: P[i] for i in range(len(nodes))}

# ========= UDG 搜索：KK →(失败)→ FR(热启动) + margin 细化 =========
def find_unit_disk_embedding(G: nx.Graph, tries=80, tol=1e-5, layout="both", seed=0,
                             refine_steps=200, refine_step=0.01):
    rng = random.Random(seed)
    nodes = list(G.nodes())

    for _ in range(tries):
        init = {v: np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)], dtype=float) for v in nodes}

        # 先 KK
        if layout in ("kk", "both"):
            try:
                pos_kk = nx.kamada_kawai_layout(G, pos=init, weight=None)
                pos_ref = refine_with_margin(G, pos_kk, tol=tol, steps=refine_steps, step=refine_step)
                ok = check_udg(G, pos_ref, tol=tol)
                if ok is not None:
                    return ok
                ok = check_udg(G, pos_kk, tol=tol)
                if ok is not None:
                    return ok
            except Exception:
                pass
            # 用 KK 结果热启动 FR
            try:
                pos_fr = nx.spring_layout(G, pos=pos_kk, iterations=250, seed=rng.randrange(1 << 30))
                pos_ref = refine_with_margin(G, pos_fr, tol=tol, steps=refine_steps, step=refine_step)
                ok = check_udg(G, pos_ref, tol=tol)
                if ok is not None:
                    return ok
                ok = check_udg(G, pos_fr, tol=tol)
                if ok is not None:
                    return ok
            except Exception:
                pass

        # 只 FR（或 both 的兜底）
        if layout in ("spring", "both"):
            try:
                pos_fr2 = nx.spring_layout(G, pos=init, iterations=250, seed=rng.randrange(1 << 30))
                pos_ref = refine_with_margin(G, pos_fr2, tol=tol, steps=refine_steps, step=refine_step)
                ok = check_udg(G, pos_ref, tol=tol)
                if ok is not None:
                    return ok
                ok = check_udg(G, pos_fr2, tol=tol)
                if ok is not None:
                    return ok
            except Exception:
                pass

    return None

# ========= 多进程 worker =========
def _udg_worker(args):
    g6, G, params = args
    # 两阶段：先轻量，再重度
    pos = find_unit_disk_embedding(
        G,
        tries=params["tries_light"],
        tol=params["tol"],
        layout=params["layout"],
        seed=params["seed"] ^ (hash(g6) & 0xFFFFFFFF),
        refine_steps=params["refine_steps_light"],
        refine_step=params["refine_step_light"],
    )
    if pos is None:
        pos = find_unit_disk_embedding(
            G,
            tries=params["tries_heavy"],
            tol=params["tol"],
            layout=params["layout"],
            seed=(params["seed"] + 1234567) ^ (hash(g6) & 0xFFFFFFFF),
            refine_steps=params["refine_steps_heavy"],
            refine_step=params["refine_step_heavy"],
        )
    if pos is None:
        return (g6, None)
    # 序列化：键保持为 int，写 JSON 会变 str，读取时我们再转回 int
    pos_json = {int(k) if isinstance(k, (np.integer, int)) else k: list(map(float, v)) for k, v in pos.items()}
    return (g6, pos_json)

# ========= 构建“按边数分层 + 加一条边”的母→子关系（并判 3-连通） =========
def build_layers_and_relations(graphs):
    info = {}
    layers = {}
    for G in graphs:
        g6 = graph_id_g6(G)
        m = G.number_of_edges()
        ds = degree_sequence(G)
        try:
            is3 = (nx.node_connectivity(G) >= 3)
        except Exception:
            is3 = False
        info[g6] = {"G": G, "m": m, "degseq": ds, "is3": is3}
        layers.setdefault(m, []).append(g6)

    for m in list(layers.keys()):
        layers[m] = sorted(layers[m], key=lambda g6: (info[g6]["degseq"], g6))

    buckets = {}
    for m, ids in layers.items():
        bm = {}
        for g6 in ids:
            bm.setdefault(info[g6]["degseq"], []).append(g6)
        buckets[m] = bm

    relations = {g6: set() for g6 in info.keys()}
    for g6p, meta in info.items():
        G = meta["G"]; m = meta["m"]; target_m = m + 1
        for u, v in nx.non_edges(G):
            Htmp = G.copy(); Htmp.add_edge(u, v)
            ds = degree_sequence(Htmp)
            cand = []
            if target_m in buckets and ds in buckets[target_m]:
                cand = buckets[target_m][ds]
            for cid in cand:
                if nx.is_isomorphic(info[cid]["G"], Htmp):
                    relations[g6p].add(cid)
                    break
    return layers, info, relations

# ========= 统一编号：从上到下、从左到右 =========
def make_numbering(layers, info):
    order = []
    for m in sorted(layers.keys()):
        order.extend(layers[m])
    idx_of = {g6: i + 1 for i, g6 in enumerate(order)}
    g6_of = {i + 1: g6 for i, g6 in enumerate(order)}
    return order, idx_of, g6_of

# ========= 图工具：渲染缩略图 =========
def render_graph_thumb_rgba(G: nx.Graph, layout="kamada", px=100, node_size=16, edge_width=1.2,
                            pos_override=None, seed=0):
    if pos_override is not None:
        # 防止 pos 的键是字符串
        pos = { (int(k) if isinstance(k, str) and k.isdigit() else k): np.array(v, dtype=float)
               for k, v in pos_override.items() }
    else:
        if layout == "spring":
            pos = nx.spring_layout(G, seed=seed, iterations=60)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)

    fig = plt.Figure(figsize=(px / 100, px / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_facecolor((1, 1, 1, 0))
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, linewidths=0)
    canvas = FigureCanvas(fig); canvas.draw()
    buf, (w, h) = canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    return img

# ========= 图二：缩略图总览（保留） =========
def draw_numbered_montage(order, info, idx_of, out_png,
                          ncols=10, thumb_px=100, layout_each="kamada",
                          hgap=0.50, vgap=0.55,
                          label_gap=0.55, label_radius=0.18,
                          font_size=8, use_udg_pos=True,
                          highlight3=True, h3_lw=2.2, h3_pad=0.06):
    N = len(order)
    ncols = max(1, ncols)
    nrows = math.ceil(N / ncols)

    thumb_in = thumb_px / 100.0
    cell_w   = thumb_in + hgap
    cell_h   = thumb_in + label_gap + vgap

    fig_w = ncols * cell_w
    fig_h = nrows * cell_h + 0.2
    plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = plt.gca(); ax.axis("off")

    thumbs = {}
    for g6 in order:
        pos_override = info[g6].get("udg_pos") if use_udg_pos else None
        thumbs[g6] = render_graph_thumb_rgba(info[g6]["G"], layout=layout_each, px=thumb_px,
                                             pos_override=pos_override)

    for idx, g6 in enumerate(order, start=1):
        r = (idx - 1) // ncols
        c = (idx - 1) % ncols
        x_left = c * cell_w
        y_top  = - r * cell_h
        cx = x_left + thumb_in / 2.0
        cy = y_top  - thumb_in / 2.0

        ab = AnnotationBbox(OffsetImage(thumbs[g6], zoom=1.0),
                            (cx, cy), frameon=False, pad=0.0, xycoords="data")
        ax.add_artist(ab)

        if highlight3 and info[g6].get("is3", False):
            ring_r = thumb_in / 2.0 + h3_pad
            ring = plt.Circle((cx, cy), radius=ring_r, fill=False, ec="red", lw=h3_lw, zorder=3)
            ax.add_artist(ring)

        ly = cy - (thumb_in / 2.0) - label_gap
        circ = plt.Circle((cx, ly), radius=label_radius, fc="white", ec="black", lw=0.9, zorder=4)
        ax.add_patch(circ)
        ax.text(cx, ly, f"{idx}", ha="center", va="center", fontsize=font_size, color="0.2", zorder=5)

    ax.set_xlim(-0.4, ncols * cell_w - (hgap * 0.2) + 0.4)
    ax.set_ylim(- (nrows * cell_h + 0.4), 0.4)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

# ========= 图一：树状（编号节点） —— 带 3-连通红圈（保留） =========
def draw_dag_numbered(layers, info, relations, idx_of, out_png,
                      col_gap=1.6, row_gap=2.0, label_col_width=1.5,
                      node_radius=0.20, font_size=9,
                      edge_width=0.5, edge_alpha=0.55,
                      fig_w=None, fig_h=None, pad=1.0,
                      highlight3=True, h3_lw=2.0, h3_delta=0.08):
    ms = sorted(layers.keys())
    for m in ms:
        layers[m] = sorted(layers[m], key=lambda g6: (info[g6]["degseq"], g6))

    pos = {}
    max_cols = 0
    for r, m in enumerate(ms):
        y = -r * row_gap
        L = len(layers[m])
        max_cols = max(max_cols, L)
        for j, g6 in enumerate(layers[m]):
            x = j * col_gap
            pos[g6] = (x, y)

    width_units  = label_col_width + (max_cols - 1) * col_gap + 2 * pad
    height_units = (len(ms) - 1) * row_gap + 2 * pad
    if fig_w is None: fig_w = max(16.0, width_units)
    if fig_h is None: fig_h = max(10.0, height_units)

    plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = plt.gca(); ax.axis("off")

    for i, m in enumerate(ms):
        ly = -i * row_gap
        ax.text(-label_col_width, ly, f"m={m}", ha="right", va="center",
                fontsize=max(font_size+1, 10), color="0.25")

    for p, chs in relations.items():
        x0, y0 = pos[p]
        for c in chs:
            x1, y1 = pos[c]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='-|>',
                                        lw=edge_width, color='0.25', alpha=edge_alpha))

    for g6, (x, y) in pos.items():
        circ = plt.Circle((x, y), radius=node_radius, fc="white", ec="black", lw=0.9, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, f"{idx_of[g6]}", ha="center", va="center",
                fontsize=font_size, zorder=4)
        if highlight3 and info[g6].get("is3", False):
            ring = plt.Circle((x, y), radius=node_radius + h3_delta, fill=False, ec="red", lw=h3_lw, zorder=5)
            ax.add_patch(ring)

    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    ax.set_xlim(-label_col_width - pad, (max(xs) if xs else 0) + pad)
    ax.set_ylim((min(ys) if ys else 0) - pad, pad)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

# ========= “母图”拆分：2-连通母图 & 3-连通母图（新定义） =========
def get_mothers_split(info: dict, relations: dict):
    """
    返回：
      mothers2: 全体 2-连通 DAG 中入度为 0 的结点（旧定义）
      mothers3: 仅看“3-连通结点 + 它们之间的边”的子 DAG 中入度为 0 的结点（新定义）
    """
    # 全体入度
    indeg_all = {g6: 0 for g6 in info.keys()}
    for p, chs in relations.items():
        for c in chs:
            indeg_all[c] += 1
    mothers2 = [g6 for g6, d in indeg_all.items() if d == 0]
    mothers2.sort()

    # 3-连通子图入度：只统计“父、子都是 3-连通”的边
    is3 = {g6: bool(meta.get("is3", False)) for g6, meta in info.items()}
    indeg_3 = {g6: 0 for g6, flag in is3.items() if flag}
    for p, chs in relations.items():
        if not is3.get(p, False):
            continue
        for c in chs:
            if is3.get(c, False):
                indeg_3[c] += 1
    mothers3 = [g6 for g6 in indeg_3.keys() if indeg_3[g6] == 0]
    mothers3.sort()
    return mothers2, mothers3

# ========= 4×2 面板：按 n=4..7 × {2连通母图, 3连通母图} 绘制 =========
def draw_mothers_matrix(ns, mothers2_by_n, mothers3_by_n, infos_by_n,
                        out_png="mothers_panel.png",
                        thumb_px=90, ncols_cell=12, cell_h_pad=0.15, cell_w_pad=0.15):
    rows = len(ns); cols = 2
    thumb_in = thumb_px/100.0
    # 粗略估尺寸
    def rows_needed(k): return max(1, math.ceil(k / ncols_cell))
    max_rows_per_cell = 1
    for n in ns:
        max_rows_per_cell = max(max_rows_per_cell,
                                rows_needed(len(mothers2_by_n.get(n, []))),
                                rows_needed(len(mothers3_by_n.get(n, []))))
    cell_w_in = ncols_cell*(thumb_in) + 1.0
    cell_h_in = max_rows_per_cell*(thumb_in+0.15) + 0.8

    fig_w = cols*cell_w_in + 1.2
    fig_h = rows*cell_h_in + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_subplot(111); ax.axis("off")

    col_titles = ["2-connected mothers", "3-connected mothers"]
    for j in range(cols):
        ax.text((j+0.5)/cols, 1.0-0.03, col_titles[j],
                ha="center", va="top", transform=ax.transAxes, fontsize=11, color="0.25")

    for i, n in enumerate(ns):
        ax.text(0.01, 1.0 - ((i+0.5)/rows), f"n={n}",
                ha="left", va="center", transform=ax.transAxes, fontsize=11, color="0.35")

        for j, which in enumerate(["2", "3"]):
            g6list = mothers2_by_n[n] if which == "2" else mothers3_by_n[n]
            info = infos_by_n[n]
            left = (j)/cols + cell_w_pad/fig_w
            right = (j+1)/cols - cell_w_pad/fig_w
            bottom = 1.0 - (i+1)/rows + cell_h_pad/fig_h
            top = 1.0 - (i)/rows - cell_h_pad/fig_h
            box_w = right - left
            box_h = top - bottom
            if not g6list:
                ax.text((left+right)/2, (bottom+top)/2, "Empty",
                        ha="center", va="center", transform=ax.transAxes, color="0.6")
                continue
            ncols = min(ncols_cell, max(1, len(g6list)))
            nrows = math.ceil(len(g6list)/ncols)
            for k, g6 in enumerate(g6list):
                r = k//ncols
                c = k%ncols
                gx = left + (c+0.5)/ncols * box_w
                gy = top - (r+0.5)/nrows * box_h
                img = render_graph_thumb_rgba(info[g6]["G"], px=thumb_px,
                                              pos_override=info[g6].get("udg_pos"))
                ab = AnnotationBbox(OffsetImage(img, zoom=1.0), (gx, gy),
                                    frameon=False, pad=0.0, xycoords=ax.transAxes)
                ax.add_artist(ab)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[Output] Mothers panel -> {out_png}")

# ========= 导出母图数据表（JSON + CSV） =========
def export_mothers_tables(ns, mothers2_by_n, mothers3_by_n, infos_by_n,
                          json_path="mothers_4to7.json", csv_path="mothers_4to7_summary.csv"):
    data = {"ranges": ns, "by_n": {}}
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["n", "kind", "g6", "edges_m", "degree_sequence"])
        for n in ns:
            entry = {"n": n, "mothers2": [], "mothers3": []}
            info = infos_by_n[n]
            # 2-connected mothers
            for g6 in mothers2_by_n.get(n, []):
                meta = info[g6]
                pos = meta.get("udg_pos", None)
                pos_json = {int(k): list(map(float, v)) for k, v in pos.items()} if pos is not None else None
                entry["mothers2"].append({
                    "g6": g6, "m": meta["m"], "degseq": list(meta["degseq"]), "pos": pos_json
                })
                w.writerow([n, "2-connected", g6, meta["m"], " ".join(map(str, meta["degseq"]))])
            # 3-connected mothers（新定义）
            for g6 in mothers3_by_n.get(n, []):
                meta = info[g6]
                pos = meta.get("udg_pos", None)
                pos_json = {int(k): list(map(float, v)) for k, v in pos.items()} if pos is not None else None
                entry["mothers3"].append({
                    "g6": g6, "m": meta["m"], "degseq": list(meta["degseq"]), "pos": pos_json
                })
                w.writerow([n, "3-connected", g6, meta["m"], " ".join(map(str, meta["degseq"]))])
            data["by_n"][str(n)] = entry
    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump(data, fj, ensure_ascii=False)
    print(f"[Output] Mothers JSON -> {json_path}")
    print(f"[Output] Mothers CSV  -> {csv_path}")

# ========= 从 JSON 读表快速重绘母图面板 =========
def draw_mothers_from_json(json_path, out_png="mothers_from_json.png",
                           thumb_px=90, ncols_cell=12):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ns = data.get("ranges", [4,5,6,7])
    by_n = data.get("by_n", {})

    # 构造“伪 info”，其 G 从 g6 复原，pos 用表里 pos
    mothers2_by_n, mothers3_by_n, infos_by_n = {}, {}, {}
    for n in ns:
        entry = by_n.get(str(n), {})
        m2 = entry.get("mothers2", [])
        m3 = entry.get("mothers3", [])
        mothers2_by_n[n] = [e["g6"] for e in m2]
        mothers3_by_n[n] = [e["g6"] for e in m3]
        info = {}
        m3_set = set(mothers3_by_n[n])
        for e in m2 + m3:
            try:
                G = nx.from_graph6_bytes(e["g6"].encode())
            except Exception:
                continue
            pos = _pos_from_json(e.get("pos", None))
            info[e["g6"]] = {"G": G, "m": e.get("m", G.number_of_edges()),
                             "degseq": tuple(e.get("degseq", [])),
                             "is3": (e["g6"] in m3_set), "udg_pos": pos}
        infos_by_n[n] = info

    draw_mothers_matrix(ns, mothers2_by_n, mothers3_by_n, infos_by_n,
                        out_png=out_png, thumb_px=thumb_px, ncols_cell=ncols_cell)

# ========= CSV =========
def save_nodes_csv(order, info, idx_of, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "g6", "edges_m", "degree_sequence", "is3connected"])
        for g6 in order:
            w.writerow([idx_of[g6], g6, info[g6]["m"], " ".join(map(str, info[g6]["degseq"])),
                        int(bool(info[g6].get("is3", False)))])

def save_relations_csv(relations, idx_of, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parent_index", "child_index"])
        for p, chs in relations.items():
            for c in sorted(chs, key=lambda g: idx_of[g]):
                w.writerow([idx_of[p], idx_of[c]])

# ========= 单个 n 的完整流程（复用缓存/并行 UDG） =========
def run_for_n(n, args, cache):
    graphs = get_unlabeled_biconnected(n)

    # 并行做 UDG（使用缓存）
    tasks = []
    for G in graphs:
        g6 = graph_id_g6(G)
        if g6 in cache:
            continue
        tasks.append((g6, G, {
            "tries_light": args.tries_light,
            "tries_heavy": args.tries_heavy,
            "tol": args.udg_tol,
            "layout": args.udg_layout,
            "seed": 0,
            "refine_steps_light": args.refine_steps_light,
            "refine_step_light": args.refine_step_light,
            "refine_steps_heavy": args.refine_steps_heavy,
            "refine_step_heavy": args.refine_step_heavy,
        }))
    if tasks:
        jobs = max(1, args.jobs)
        with Pool(processes=jobs) as pool:
            for g6, pos_json in pool.imap_unordered(_udg_worker, tasks, chunksize=2):
                cache[g6] = pos_json

    passed, udg_pos = [], {}
    for G in graphs:
        g6 = graph_id_g6(G)
        pos_json = cache.get(g6, None)
        if pos_json is not None:
            passed.append(G)
            udg_pos[g6] = _pos_from_json(pos_json)   # <<< 关键修复：键转 int
    graphs = passed

    layers, info, relations = build_layers_and_relations(graphs)
    for g6, meta in info.items():
        if g6 in udg_pos:
            meta["udg_pos"] = udg_pos[g6]
    order, idx_of, g6_of = make_numbering(layers, info)

    # === 核心变更：新定义的 mothers3 ===
    mothers2, mothers3 = get_mothers_split(info, relations)
    return graphs, layers, info, relations, order, idx_of, mothers2, mothers3

# ========= CLI =========
def main():
    ap = argparse.ArgumentParser(
        description="2-connected unlabeled graphs (n≤7): UDG filter + add-one-edge DAG + mothers export/plot."
    )
    ap.add_argument("-n", type=int, default=5, help="number of vertices (≤7)")

    # 批量与输出
    ap.add_argument("--batch_4_7", action="store_true", help="process n=4,5,6,7 and build mothers panel")
    ap.add_argument("--mothers_png", type=str, default="mothers_panel.png", help="panel figure for mothers across n=4..7")
    ap.add_argument("--mothers_json", type=str, default="mothers_4to7.json", help="export JSON (mothers + positions)")
    ap.add_argument("--mothers_csv", type=str, default="mothers_4to7_summary.csv", help="export CSV summary (no positions)")
    # 读表重绘
    ap.add_argument("--plot_from_json", type=str, default=None, help="if set, skip computation and draw mothers panel from JSON")

    # 单 n 输出（保留）
    ap.add_argument("--dag_png", type=str, default="n5_dag_numbered.png")
    ap.add_argument("--montage_png", type=str, default="n5_montage_numbered.png")
    ap.add_argument("--nodes_csv", type=str, default="n5_nodes.csv")
    ap.add_argument("--relations_csv", type=str, default="n5_relations.csv")
    ap.add_argument("--ncols", type=int, default=10)
    ap.add_argument("--thumb_px", type=int, default=100)
    ap.add_argument("--layout_each", choices=["kamada","spring","circular","spectral"], default="kamada")
    ap.add_argument("--thumb_from_udg", action="store_true", default=True)

    # --- UDG 参数 / 并行与缓存 ---
    ap.add_argument("--udg_layout", choices=["kk", "spring", "both"], default="both")
    ap.add_argument("--jobs", type=int, default=max(1, cpu_count()-1))
    ap.add_argument("--udg_cache", type=str, default="udg_cache_n.json")
    ap.add_argument("--udg_tol", type=float, default=1e-5)
    ap.add_argument("--tries_light", type=int, default=40)
    ap.add_argument("--refine_steps_light", type=int, default=150)
    ap.add_argument("--refine_step_light", type=float, default=0.010)
    ap.add_argument("--tries_heavy", type=int, default=140)
    ap.add_argument("--refine_steps_heavy", type=int, default=600)
    ap.add_argument("--refine_step_heavy", type=float, default=0.012)

    # --- DAG 画图 ---
    ap.add_argument("--dag_col_gap", type=float, default=1.6)
    ap.add_argument("--dag_row_gap", type=float, default=2.0)
    ap.add_argument("--dag_label_col", type=float, default=1.5)
    ap.add_argument("--dag_node_radius", type=float, default=0.20)
    ap.add_argument("--dag_font", type=int, default=9)
    ap.add_argument("--dag_edge_width", type=float, default=0.5)
    ap.add_argument("--dag_edge_alpha", type=float, default=0.55)
    ap.add_argument("--dag_fig_w", type=float, default=None)
    ap.add_argument("--dag_fig_h", type=float, default=None)

    # --- 3-连通高亮（保留） ---
    ap.add_argument("--no_highlight3", action="store_true")
    ap.add_argument("--h3_dag_lw", type=float, default=2.0)
    ap.add_argument("--h3_dag_delta", type=float, default=0.08)
    ap.add_argument("--h3_thumb_lw", type=float, default=2.2)
    ap.add_argument("--h3_thumb_pad", type=float, default=0.06)

    args = ap.parse_args()

    # 若仅需读表重绘，直接走这条路径
    if args.plot_from_json:
        draw_mothers_from_json(args.plot_from_json, out_png=args.mothers_png,
                               thumb_px=max(80, args.thumb_px-10), ncols_cell=12)
        return

    # 载入缓存
    cache = {}
    if args.udg_cache and os.path.exists(args.udg_cache):
        try:
            with open(args.udg_cache, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"[Cache] loaded {len(cache)} entries from {args.udg_cache}")
        except Exception as e:
            print(f"[Cache] failed to read {args.udg_cache}: {e}")

    if not args.batch_4_7:
        # ==== 单个 n 的原流程（保持不变） ====
        n = args.n
        graphs, layers, info, relations, order, idx_of, mothers2, mothers3 = run_for_n(n, args, cache)

        save_nodes_csv(order, info, idx_of, args.nodes_csv)
        save_relations_csv(relations, idx_of, args.relations_csv)
        print(f"[Output] nodes -> {args.nodes_csv}")
        print(f"[Output] relations -> {args.relations_csv}")

        draw_dag_numbered(
            layers, info, relations, idx_of, args.dag_png,
            col_gap=args.dag_col_gap, row_gap=args.dag_row_gap, label_col_width=args.dag_label_col,
            node_radius=args.dag_node_radius, font_size=args.dag_font,
            edge_width=args.dag_edge_width, edge_alpha=args.dag_edge_alpha,
            fig_w=args.dag_fig_w, fig_h=args.dag_fig_h,
            highlight3=(not args.no_highlight3), h3_lw=args.h3_dag_lw, h3_delta=args.h3_dag_delta
        )
        print(f"[Output] Fig1 DAG -> {args.dag_png}")

        draw_numbered_montage(
            order, info, idx_of, args.montage_png,
            ncols=args.ncols, thumb_px=args.thumb_px, layout_each=args.layout_each,
            use_udg_pos=args.thumb_from_udg,
            highlight3=(not args.no_highlight3), h3_lw=args.h3_thumb_lw, h3_pad=args.h3_thumb_pad
        )
        print(f"[Output] Fig2 montage -> {args.montage_png}")

        # 终端打印母图（按新定义）
        print("\n[Mother graphs] (new definitions)")
        print(f"n={n}: mothers-2 (in full DAG) = {len(mothers2)}")
        print("  indices & g6:", [(idx_of[g6], g6) for g6 in mothers2])
        print(f"n={n}: mothers-3 (in 3-connected subDAG) = {len(mothers3)}")
        print("  indices & g6:", [(idx_of[g6], g6) for g6 in mothers3])

        # 回写缓存
        if args.udg_cache:
            try:
                with open(args.udg_cache, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
                print(f"[Cache] saved to {args.udg_cache}")
            except Exception as e:
                print(f"[Cache] save failed: {e}")
        return

    # ==== 批量：n=4..7 ====
    ns = [4,5,6,7]
    mothers2_by_n, mothers3_by_n, infos_by_n = {}, {}, {}

    for n in ns:
        print(f"\n=== Processing n={n} ===")
        graphs, layers, info, relations, order, idx_of, mothers2, mothers3 = run_for_n(n, args, cache)

        # 终端打印母图（按新定义）
        print(f"[Mother graphs] n={n}: mothers-2 (full DAG) = {len(mothers2)}")
        print("  indices & g6:", [(idx_of[g6], g6) for g6 in mothers2])
        print(f"[Mother graphs] n={n}: mothers-3 (3-connected subDAG) = {len(mothers3)}")
        print("  indices & g6:", [(idx_of[g6], g6) for g6 in mothers3])

        mothers2_by_n[n] = mothers2
        mothers3_by_n[n] = mothers3
        infos_by_n[n] = info  # 含 udg_pos 以便绘图/导出

    # 面板图：4 行（n=4..7）× 2 列（2连通/3连通）
    draw_mothers_matrix(ns, mothers2_by_n, mothers3_by_n, infos_by_n,
                        out_png=args.mothers_png, thumb_px=max(80, args.thumb_px-10), ncols_cell=12)

    # 导出数据表（JSON + CSV）
    export_mothers_tables(ns, mothers2_by_n, mothers3_by_n, infos_by_n,
                          json_path=args.mothers_json, csv_path=args.mothers_csv)

    # 回写缓存
    if args.udg_cache:
        try:
            with open(args.udg_cache, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            print(f"[Cache] saved to {args.udg_cache}")
        except Exception as e:
            print(f"[Cache] save failed: {e}")

if __name__ == "__main__":
    if get_start_method(allow_none=True) != "spawn":
        try:
            import multiprocessing as mp
            mp.set_start_method("spawn", force=True)
        except Exception:
            pass
    main()
