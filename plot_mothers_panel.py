#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取 mothers_4to7.json（由 biconnected_edge_dag_physics.py 导出）

功能 A：绘制“母图集合”面板图
  行： n=4,5,6,7
  列： 2-connected mothers / 3-connected mothers （--only 可选）

功能 B：构造“7→6→5→4 的最优删点路径树（分栏：左=2连通，右=3连通）”
  - 一步（删 1 点）最优映射：蓝色边，适用于全部父图；
  - 跨两层（删 2 点）最优映射：红色边，仅对 3-连通父图启用；
  - 目标只在 JSON 已存在的母图集合中寻找；
  - 并列最优的子项全部连边。

功能 C：导出映射 JSON（包含节点、带细节的边、以及所有 7→…→4 的可行路径）
  - 由 --map_json 触发导出

用法示例：
  python plot_mothers_panel.py --json mothers_4to7.json --out mothers_panel.png
  python plot_mothers_panel.py --json mothers_4to7.json --best_tree_png best_tree_7to4_split.png
  python plot_mothers_panel.py --json mothers_4to7.json --map_json mothers_mappings.json
"""

import argparse, json, math, csv, itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -------- 小工具：JSON 键转 int / 还原 pos --------
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
    """obj: {node(str/int): [x,y]} -> {node(int): np.array([x,y])}"""
    if obj is None:
        return None
    d = _json_keys_to_int(obj) if isinstance(obj, dict) else {}
    return {k: np.array(v, dtype=float) for k, v in d.items()}

# -------- 渲染单个缩略图为 RGBA --------
def render_graph_thumb_rgba(G: nx.Graph, px=100, pos_override=None, edge_width=1.2, node_size=16):
    # 优先使用 JSON 中的 UDG 见证坐标；没有则回退 KK
    if pos_override is not None:
        pos = { (int(k) if isinstance(k, str) and k.isdigit() else k): np.array(v, dtype=float)
               for k, v in pos_override.items() }
    else:
        pos = nx.kamada_kawai_layout(G)

    fig = plt.Figure(figsize=(px/100, px/100), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_facecolor((1,1,1,0))

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, linewidths=0)

    canvas = FigureCanvas(fig); canvas.draw()
    buf, (w, h) = canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    return img

# -------- 面板绘制：行 = n，列 = 2连通/3连通 --------
def draw_mothers_matrix_from_json(data, out_png="mothers_fromjson.png",
                                  thumb_px=90, ncols_cell=12, dpi=200, only="both", title=None):
    ns = data.get("ranges", [])
    by_n = data.get("by_n", {})
    ns = [n for n in ns if str(n) in by_n]
    ns.sort()

    mothers2_by_n, mothers3_by_n, infos_by_n = {}, {}, {}
    for n in ns:
        entry = by_n.get(str(n), {})
        m2 = entry.get("mothers2", [])
        m3 = entry.get("mothers3", [])
        mothers2_by_n[n] = [e["g6"] for e in m2]
        mothers3_by_n[n] = [e["g6"] for e in m3]
        info = {}
        m3set = set(mothers3_by_n[n])
        for e in (m2 + m3):
            g6 = e["g6"]
            try:
                G = nx.from_graph6_bytes(g6.encode())
            except Exception:
                continue
            pos = _pos_from_json(e.get("pos", None))
            info[g6] = {
                "G": G,
                "udg_pos": pos,
                "is3": (g6 in m3set),
                "m": int(e.get("m", G.number_of_edges())),
                "degseq": tuple(e.get("degseq", []))
            }
        infos_by_n[n] = info

    if only == "2":
        cols = 1; col_titles = ["2-connected mothers"]; which_cols = ["2"]
    elif only == "3":
        cols = 1; col_titles = ["3-connected mothers"]; which_cols = ["3"]
    else:
        cols = 2; col_titles = ["2-connected mothers", "3-connected mothers"]; which_cols = ["2", "3"]

    rows = len(ns)
    thumb_in = thumb_px/100.0

    def rows_needed(k): return max(1, math.ceil(k / ncols_cell))
    max_rows_per_cell = 1
    for n in ns:
        if "2" in which_cols:
            max_rows_per_cell = max(max_rows_per_cell, rows_needed(len(mothers2_by_n.get(n, []))))
        if "3" in which_cols:
            max_rows_per_cell = max(max_rows_per_cell, rows_needed(len(mothers3_by_n.get(n, []))))

    cell_w_in = ncols_cell*(thumb_in) + 1.0
    cell_h_in = max_rows_per_cell*(thumb_in+0.15) + 0.8
    fig_w = cols*cell_w_in + 1.2
    fig_h = rows*cell_h_in + (0.8 if title else 0.3)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111); ax.axis("off")

    if title:
        ax.text(0.5, 1.0, title, ha="center", va="top", transform=ax.transAxes, fontsize=13, color="0.2")
        top_y = 0.97
    else:
        top_y = 1.0

    for j, t in enumerate(col_titles):
        ax.text((j+0.5)/cols, top_y-0.03, t, ha="center", va="top", transform=ax.transAxes, fontsize=11, color="0.25")

    for i, n in enumerate(ns):
        ax.text(0.01, top_y - ((i+0.5)/rows), f"n={n}", ha="left", va="center",
                transform=ax.transAxes, fontsize=11, color="0.35")

        for j, which in enumerate(which_cols):
            g6list = mothers2_by_n[n] if which == "2" else mothers3_by_n[n]
            info = infos_by_n[n]

            left = (j)/cols + 0.02
            right = (j+1)/cols - 0.02
            bottom = top_y - (i+1)/rows
            top = top_y - i/rows
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
    print(f"[Output] mothers panel -> {out_png}")

# ====================== 最优删点映射：Exact（删1点 / 删2点） ======================
def _adj_matrix_bool(G: nx.Graph, order=None):
    """返回布尔邻接矩阵（上三角边位点列表一起返回）"""
    if order is None:
        order = sorted(G.nodes())
    idx = {v:i for i,v in enumerate(order)}
    k = len(order)
    A = np.zeros((k,k), dtype=np.uint8)
    for u,v in G.edges():
        i, j = idx[u], idx[v]
        if i==j: continue
        if i<j: A[i,j] = 1
        else:   A[j,i] = 1
    edges_up = [(i,j) for i in range(k) for j in range(i+1,k) if A[i,j]]
    return A, edges_up, order

def _best_overlap_cost(A_up, B, k):
    """
    给定 G' 的上三角边位点列表 A_up、H 的邻接矩阵 B，枚举所有 k! 映射，返回：
      best_overlap, best_perms(list)
    其中 overlap = |E(G') ∩ π^{-1}(E(H))|
    """
    best_overlap = -1
    best_perms = []
    for perm in itertools.permutations(range(k)):  # k<=6 时最多 720 次
        ov = 0
        for (i,j) in A_up:
            if B[perm[i], perm[j]]:
                ov += 1
        if ov > best_overlap:
            best_overlap = ov
            best_perms = [perm]
        elif ov == best_overlap:
            best_perms.append(perm)
    return best_overlap, best_perms

def _delete_vertex_and_matrix(G: nx.Graph, v_del):
    """删一个点后构造 G' 的 (A, edges_up, order)"""
    nodes_all = sorted(G.nodes())
    keep = [u for u in nodes_all if u != v_del]
    Gp = G.subgraph(keep).copy()
    A, edges_up, order = _adj_matrix_bool(Gp, keep)
    return Gp, A, edges_up, order

def _delete_two_vertices_and_matrix(G: nx.Graph, v1, v2):
    nodes_all = sorted(G.nodes())
    keep = [u for u in nodes_all if u not in (v1, v2)]
    Gp = G.subgraph(keep).copy()
    A, edges_up, order = _adj_matrix_bool(Gp, keep)
    return Gp, A, edges_up, order

def compute_best_children_step(parents_info: dict, targets_info: dict, record_details=False):
    """
    一步：删 1 点，父 n → 子 n-1
    parents_info / targets_info: {g6: {"G": Graph, ...}}
    返回：
      best_children: {parent_g6: set(child_g6,...)}
      details: list[{parent, child, deleted_vertex, overlap, cost, ...}]（仅 record_details=True 收集）
    cost = |E(G')| + |E(H)| - 2*overlap
    """
    best_children = {g6:set() for g6 in parents_info.keys()}
    details = []

    Bmats = {}
    for h6, meta in targets_info.items():
        H = meta["G"]
        B, _, _ = _adj_matrix_bool(H)
        Bmats[h6] = (B, H.number_of_edges())

    for p6, pmeta in parents_info.items():
        G = pmeta["G"]; n = G.number_of_nodes()
        best_cost = None; best_set = set()
        for v in sorted(G.nodes()):
            Gp, A, A_up, order = _delete_vertex_and_matrix(G, v)
            k = n-1
            E_Gp = Gp.number_of_edges()
            for h6, (B, E_H) in Bmats.items():
                ov, _ = _best_overlap_cost(A_up, B, k)
                cost = E_Gp + E_H - 2*ov
                if (best_cost is None) or (cost < best_cost):
                    best_cost = cost; best_set = {h6}
                elif cost == best_cost:
                    best_set.add(h6)
        best_children[p6] = best_set

        if record_details:
            for v in sorted(G.nodes()):
                Gp, A, A_up, order = _delete_vertex_and_matrix(G, v)
                k = n-1; E_Gp = Gp.number_of_edges()
                for h6, (B, E_H) in Bmats.items():
                    ov, _ = _best_overlap_cost(A_up, B, k)
                    cost = E_Gp + E_H - 2*ov
                    if h6 in best_set and cost == best_cost:
                        details.append({
                            "n_parent": n, "parent_g6": p6,
                            "n_child": n-1, "child_g6": h6,
                            "deleted": [int(v)],
                            "edges_parent_sub": int(E_Gp),
                            "edges_child": int(E_H),
                            "overlap": int(ov),
                            "cost": int(cost),
                            "mode": "step1"
                        })
    return best_children, details

def compute_best_children_skip2(parents_info3: dict, targets_info: dict, record_details=False):
    """
    跨两层：删 2 点，父 n → 子 n-2（仅对 3-连通父图调用）
    """
    best_children = {g6:set() for g6 in parents_info3.keys()}
    details = []

    Bmats = {}
    for h6, meta in targets_info.items():
        H = meta["G"]
        B, _, _ = _adj_matrix_bool(H)
        Bmats[h6] = (B, H.number_of_edges())

    for p6, pmeta in parents_info3.items():
        G = pmeta["G"]; n = G.number_of_nodes()
        best_cost = None; best_set = set()
        nodes = sorted(G.nodes())
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                v1, v2 = nodes[i], nodes[j]
                Gp, A, A_up, order = _delete_two_vertices_and_matrix(G, v1, v2)
                k = n-2; E_Gp = Gp.number_of_edges()
                for h6, (B, E_H) in Bmats.items():
                    ov, _ = _best_overlap_cost(A_up, B, k)
                    cost = E_Gp + E_H - 2*ov
                    if (best_cost is None) or (cost < best_cost):
                        best_cost = cost; best_set = {h6}
                    elif cost == best_cost:
                        best_set.add(h6)
        best_children[p6] = best_set

        if record_details:
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    v1, v2 = nodes[i], nodes[j]
                    Gp, A, A_up, order = _delete_two_vertices_and_matrix(G, v1, v2)
                    k = n-2; E_Gp = Gp.number_of_edges()
                    for h6, (B, E_H) in Bmats.items():
                        ov, _ = _best_overlap_cost(A_up, B, k)
                        cost = E_Gp + E_H - 2*ov
                        if h6 in best_set and cost == best_cost:
                            details.append({
                                "n_parent": n, "parent_g6": p6,
                                "n_child": n-2, "child_g6": h6,
                                "deleted": [int(v1), int(v2)],
                                "edges_parent_sub": int(E_Gp),
                                "edges_child": int(E_H),
                                "overlap": int(ov),
                                "cost": int(cost),
                                "mode": "skip2"
                            })
    return best_children, details

# ====================== 构造分栏树：左=2连通，右=3连通 ======================
def build_best_path_tree_split(data, record_csv=None):
    """
    读取 JSON，分层得到：
      levels2[n] = [g6,...]（该层 2-连通母图）
      levels3[n] = [g6,...]（该层 3-连通母图）
      infos[n][g6] = meta
    并计算：
      edges_step[n][parent] = set(children)      # 蓝线：n→n-1（全部父图）
      edges_skip2[n][parent] = set(children)     # 红线：n→n-2（仅 3-连通父图）

    目标层均使用“该层 2、3 母图的并集”（严格只用 JSON 存在的）。
    """
    by_n = data.get("by_n", {})
    ns = [n for n in [7,6,5,4] if str(n) in by_n]
    if len(ns) < 2:
        raise ValueError("JSON 中可用的 n 层不足以构造 7→4 树")

    levels2, levels3, infos = {}, {}, {}

    for n in ns:
        entry = by_n[str(n)]
        m2 = entry.get("mothers2", [])
        m3 = entry.get("mothers3", [])

        # 2-连通层
        arr2 = []
        for e in m2:
            g6 = e["g6"]
            try:
                G = nx.from_graph6_bytes(g6.encode())
            except Exception:
                continue
            meta = {"G": G, "udg_pos": _pos_from_json(e.get("pos", None)),
                    "m": int(e.get("m", G.number_of_edges())),
                    "degseq": tuple(e.get("degseq", [])), "is3": False}
            arr2.append((meta["degseq"], meta["m"], g6, meta))
        arr2.sort(key=lambda t: (t[0], t[1], t[2]))
        levels2[n] = [t[2] for t in arr2]

        # 3-连通层
        arr3 = []
        for e in m3:
            g6 = e["g6"]
            try:
                G = nx.from_graph6_bytes(g6.encode())
            except Exception:
                continue
            meta = {"G": G, "udg_pos": _pos_from_json(e.get("pos", None)),
                    "m": int(e.get("m", G.number_of_edges())),
                    "degseq": tuple(e.get("degseq", [])), "is3": True}
            arr3.append((meta["degseq"], meta["m"], g6, meta))
        arr3.sort(key=lambda t: (t[0], t[1], t[2]))
        levels3[n] = [t[2] for t in arr3]

        # infos 合并两侧
        infos[n] = {t[2]: t[3] for t in (arr2+arr3)}

    # 蓝线：一步 n→n-1（全体父图）
    edges_step = {}
    map_rows = []
    for i in range(len(ns)-1):
        n = ns[i]; n1 = ns[i+1]
        parents = {**{g6:infos[n][g6] for g6 in levels2[n]},
                   **{g6:infos[n][g6] for g6 in levels3[n]}}
        targets = {**{g6:infos[n1][g6] for g6 in levels2[n1]},
                   **{g6:infos[n1][g6] for g6 in levels3[n1]}}
        best_children, details = compute_best_children_step(parents, targets, record_details=True)
        edges_step[n] = best_children
        map_rows.extend(details)

    # 红线：仅 3-连通父图允许跨两层 n→n-2
    edges_skip2 = {}
    for i in range(len(ns)-2):
        n = ns[i]; n2 = ns[i+2]
        parents3 = {g6:infos[n][g6] for g6 in levels3[n]}  # 仅 3-连通父图
        targets = {**{g6:infos[n2][g6] for g6 in levels2[n2]},
                   **{g6:infos[n2][g6] for g6 in levels3[n2]}}
        best_children2, details2 = compute_best_children_skip2(parents3, targets, record_details=True)
        edges_skip2[n] = best_children2
        map_rows.extend(details2)

    # 导出映射（可选）
    if record_csv:
        with open(record_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mode","n_parent","parent_g6","n_child","child_g6","deleted","edges_parent_sub","edges_child","overlap","cost"])
            for r in map_rows:
                w.writerow([r["mode"], r["n_parent"], r["parent_g6"], r["n_child"], r["child_g6"],
                            " ".join(map(str, r["deleted"])),
                            r["edges_parent_sub"], r["edges_child"], r["overlap"], r["cost"]])

    # ★新增：返回 map_rows，供 JSON 导出使用
    return levels2, levels3, infos, edges_step, edges_skip2, map_rows

# -------- 画“分栏最优路径树”：左=2连通，右=3连通 --------
def draw_best_path_tree_split(levels2, levels3, infos, edges_step, edges_skip2,
                              out_png="best_tree_geom_split.png",
                              thumb_px=110, row_gap=2.3, col_gap=1.6,
                              mid_gap=3.0, label_col_width=1.6,
                              dpi=220, lw_blue=0.9, lw_red=1.2,
                              height_pad=6.0):
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    ns = sorted(set(list(levels2.keys())+list(levels3.keys())), reverse=True)
    left_max = max((len(levels2.get(n, [])) for n in ns), default=0)
    right_max = max((len(levels3.get(n, [])) for n in ns), default=0)

    # 布局中心点（数据坐标）
    pos = {}
    for r, n in enumerate(ns):
        y = -r * row_gap
        left = levels2.get(n, []); right = levels3.get(n, [])
        for j, g6 in enumerate(left):
            x = j * col_gap; pos[(n,g6)] = (x,y)
        right_base = left_max*col_gap + mid_gap
        for j, g6 in enumerate(right):
            x = right_base + (right_max - len(right) + j) * col_gap
            pos[(n,g6)] = (x,y)

    width_units  = label_col_width + left_max*col_gap + mid_gap + max(right_max*col_gap, 0) + 2.0
    height_units = (len(ns)-1)*row_gap + height_pad

    fig = plt.figure(figsize=(max(20.0, width_units), max(12.0, height_units)), dpi=dpi)
    ax = fig.add_subplot(111); ax.axis("off")

    # 行标签
    for r, n in enumerate(ns):
        ly = -r*row_gap
        ax.text(-label_col_width, ly, f"n={n}", ha="right", va="center", fontsize=11, color="0.35")

    # 缩略图
    thumbs = {}
    def get_thumb(n, g6):
        key=(n,g6)
        if key in thumbs: return thumbs[key]
        img = render_graph_thumb_rgba(infos[n][g6]["G"], px=thumb_px, pos_override=infos[n][g6].get("udg_pos"))
        thumbs[key]=img; return img

    boxes = {}
    for n in ns:
        for g6 in levels2.get(n, []) + levels3.get(n, []):
            x,y = pos[(n,g6)]
            ab = AnnotationBbox(OffsetImage(get_thumb(n,g6), zoom=1.0), (x,y),
                                frameon=True, pad=0.02,
                                bboxprops=dict(fc="white", ec="0.6", lw=0.6),
                                zorder=3.0)
            ax.add_artist(ab)
            boxes[(n,g6)] = ab

    # 先固定坐标范围，再绘制（保证像素↔数据映射稳定）
    xs=[x for (x,_) in pos.values()]; ys=[y for (_,y) in pos.values()]
    ax.set_xlim(-label_col_width-1.0, (max(xs) if xs else 0)+1.0)
    ax.set_ylim((min(ys) if ys else 0)-1.0, 1.0)
    ax.set_aspect("auto")  # 关键：取消等比例压扁，纵向间距真实生效

    # 计算每个框在像素坐标的包围盒，取顶/底边中心
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def top_center_data(ab):
        bb = ab.get_window_extent(renderer=renderer)
        cx = 0.5*(bb.x0+bb.x1); cy = bb.y1
        return ax.transData.inverted().transform((cx, cy))

    def bottom_center_data(ab):
        bb = ab.get_window_extent(renderer=renderer)
        cx = 0.5*(bb.x0+bb.x1); cy = bb.y0
        return ax.transData.inverted().transform((cx, cy))

    # 蓝线：n→n-1  父底→子顶
    for n, mapping in edges_step.items():
        n1 = n-1
        for p6, childs in mapping.items():
            if (n,p6) not in boxes: continue
            sx, sy = bottom_center_data(boxes[(n,p6)])
            for c6 in childs:
                if (n1,c6) not in boxes: continue
                tx, ty = top_center_data(boxes[(n1,c6)])
                ax.annotate("", xy=(tx,ty), xytext=(sx,sy),
                            arrowprops=dict(arrowstyle='-|>', lw=lw_blue, color='#1f77b4',
                                            alpha=0.9, shrinkA=0, shrinkB=0),
                            zorder=2.0)

    # 红线：n→n-2（仅3-连通） 父底→子顶
    for n, mapping in edges_skip2.items():
        n2 = n-2
        for p6, childs in mapping.items():
            if (n,p6) not in boxes: continue
            sx, sy = bottom_center_data(boxes[(n,p6)])
            for c6 in childs:
                if (n2,c6) not in boxes: continue
                tx, ty = top_center_data(boxes[(n2,c6)])
                ax.annotate("", xy=(tx,ty), xytext=(sx,sy),
                            arrowprops=dict(arrowstyle='-|>', lw=lw_red, color='#d62728',
                                            alpha=0.95, shrinkA=0, shrinkB=0),
                            zorder=2.0)

    # 图例
    ax.plot([], [], color='#1f77b4', lw=lw_blue, label='删1点（n→n-1）最小RMS位移')
    ax.plot([], [], color='#d62728', lw=lw_red,   label='删2点（n→n-2，仅3-连通）最小RMS位移')
    ax.legend(loc='upper right', frameon=False, fontsize=10)

    # 不再 tight_layout，避免再次改变布局比例
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[Output] best split path tree (geom) -> {out_png}")

# -------- ★新增：导出节点/边/路径 的 JSON --------
def _collect_nodes_by_n(levels2, levels3, infos):
    """把每层所有节点（2连通 ∪ 3连通）收集，并带上元数据。"""
    ns = sorted(set(list(levels2.keys()) + list(levels3.keys())), reverse=True)
    nodes_by_n = {}
    for n in ns:
        arr = []
        for g6 in (levels2.get(n, []) + levels3.get(n, [])):
            meta = infos[n][g6]
            pos = meta.get("udg_pos", None)
            pos_json = {int(k): list(map(float, v)) for k, v in pos.items()} if pos is not None else None
            arr.append({
                "g6": g6,
                "is3": bool(meta.get("is3", False)),
                "m": int(meta.get("m", meta["G"].number_of_edges())),
                "degseq": list(meta.get("degseq", [])),
                "pos": pos_json
            })
        nodes_by_n[str(n)] = arr
    return ns, nodes_by_n

def _build_adjacency_for_paths(levels2, levels3, edges_step, edges_skip2):
    """将蓝/红边合并成从 (n,g6) 指向 (n-1,child) 或 (n-2,child) 的邻接表。"""
    ns = sorted(set(list(levels2.keys()) + list(levels3.keys())), reverse=True)
    adj = {}  # key: (n, g6) -> list[(n_child, g6_child, mode)]
    for n in ns:
        for g6 in (levels2.get(n, []) + levels3.get(n, [])):
            adj[(n, g6)] = []
    # 蓝边 n->n-1
    for n, mapping in edges_step.items():
        n1 = n - 1
        for p6, childs in mapping.items():
            for c6 in childs:
                adj[(n, p6)].append((n1, c6, "step1"))
    # 红边 n->n-2
    for n, mapping in edges_skip2.items():
        n2 = n - 2
        for p6, childs in mapping.items():
            for c6 in childs:
                adj[(n, p6)].append((n2, c6, "skip2"))
    return adj

def _enumerate_paths(ns, levels2, levels3, adj):
    """
    枚举从最上层 n_max 的任一节点，一直降到 n_min(=4) 的所有路径。
    路径中既可能走 step1，也可能跨层走 skip2（只要边存在）。
    """
    if not ns:
        return []
    n_max, n_min = max(ns), min(ns)
    starts = (levels2.get(n_max, []) + levels3.get(n_max, []))
    paths = []

    def dfs(path_nodes):  # path_nodes: list[(n,g6)]
        n_cur, g6_cur = path_nodes[-1]
        if n_cur <= n_min:
            if n_cur == n_min:
                paths.append([{"n": n, "g6": g} for (n, g) in path_nodes])
            return
        for (n_next, g6_next, _mode) in adj.get((n_cur, g6_cur), []):
            if n_next < n_cur and n_next >= n_min:
                path_nodes.append((n_next, g6_next))
                dfs(path_nodes)
                path_nodes.pop()

    for g6s in starts:
        dfs([(n_max, g6s)])
    return paths

def export_best_tree_json(out_path, levels2, levels3, infos, edges_step, edges_skip2, map_rows):
    """
    输出 JSON，包含：
      - nodes_by_n：每层的节点及其元数据（is3/m/degseq/pos）
      - edges：step1 / skip2 两类边的细节（含 deleted/overlap/cost 等）
      - paths：从最高层 n_max 到 n_min(=4) 的所有可行路径（按 g6 序列记录）
    """
    ns, nodes_by_n = _collect_nodes_by_n(levels2, levels3, infos)

    # 整理边（用 map_rows 明细；并区分 step1/skip2）
    edges = {"step1": [], "skip2": []}
    for r in map_rows:
        item = {
            "n_parent": int(r["n_parent"]),
            "parent_g6": r["parent_g6"],
            "n_child": int(r["n_child"]),
            "child_g6": r["child_g6"],
            "deleted": list(map(int, r["deleted"])),
            "edges_parent_sub": int(r["edges_parent_sub"]),
            "edges_child": int(r["edges_child"]),
            "overlap": int(r["overlap"]),
            "cost": int(r["cost"]),
        }
        if r["mode"] == "step1":
            edges["step1"].append(item)
        else:
            edges["skip2"].append(item)

    # 枚举所有 7→…→4 的路径
    adj = _build_adjacency_for_paths(levels2, levels3, edges_step, edges_skip2)
    paths = _enumerate_paths(ns, levels2, levels3, adj)

    data_out = {
        "ranges": ns,                  # 例如 [7,6,5,4]
        "nodes_by_n": nodes_by_n,      # 每层所有节点及元数据
        "edges": edges,                # 带 deleted / cost 的边
        "paths": paths                 # 可行路径（按 g6 序列）
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=2)
    print(f"[Output] mappings JSON -> {out_path}")

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Plot mothers panel and/or build split best deletion path tree from mothers_4to7.json")
    ap.add_argument("--json", required=True, help="path to mothers_4to7.json exported previously")

    # 面板（A）
    ap.add_argument("--out", default=None, help="(A) mothers panel PNG path; skip if not set")
    ap.add_argument("--only", choices=["both","2","3"], default="both", help="(A) which set(s) to plot")
    ap.add_argument("--title", type=str, default=None, help="(A) optional panel title")

    # 分栏最优路径树（B）
    ap.add_argument("--best_tree_png", type=str, default=None, help="(B) output PNG for split best path tree")
    ap.add_argument("--map_csv", type=str, default=None, help="(B) optional CSV for parent→child mappings (step1/skip2)")
    # ★新增：导出映射 JSON（节点 + 边 + 路径）
    ap.add_argument("--map_json", type=str, default=None, help="(B) optional JSON to save parent→child mappings and all paths")

    # 画图通用
    ap.add_argument("--thumb_px", type=int, default=110, help="thumbnail size (px)")
    ap.add_argument("--ncols", type=int, default=12, help="(A) columns per cell in panel")
    ap.add_argument("--dpi", type=int, default=220, help="figure DPI")

    # 分栏布局参数（B）
    ap.add_argument("--row_gap", type=float, default=2.3)
    ap.add_argument("--col_gap", type=float, default=1.6)
    ap.add_argument("--mid_gap", type=float, default=3.0)
    ap.add_argument("--label_col", type=float, default=1.6)

    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # A) 面板（可选）
    if args.out:
        draw_mothers_matrix_from_json(
            data,
            out_png=args.out,
            thumb_px=args.thumb_px,
            ncols_cell=args.ncols,
            dpi=args.dpi,
            only=args.only,
            title=args.title
        )

    # B) 分栏最优路径树（可选） + 导出映射 JSON（可选）
    if args.best_tree_png or args.map_json or args.map_csv:
        levels2, levels3, infos, edges_step, edges_skip2, map_rows = build_best_path_tree_split(data, record_csv=args.map_csv)

        if args.best_tree_png:
            draw_best_path_tree_split(
                levels2, levels3, infos, edges_step, edges_skip2,
                out_png=args.best_tree_png,
                thumb_px=args.thumb_px, row_gap=args.row_gap, col_gap=args.col_gap,
                mid_gap=args.mid_gap, label_col_width=args.label_col, dpi=args.dpi
            )

        if args.map_json:
            export_best_tree_json(args.map_json, levels2, levels3, infos, edges_step, edges_skip2, map_rows)

if __name__ == "__main__":
    main()
