#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 mothers_mappings.json 直接绘图为“分栏最优路径树（左=2连通，右=3连通）”，
并将最上层（n_max=7）随机选取一个起点→逐层向下的一条可行路径用绿色高亮。

输入：mothers_mappings.json（由 plot_mothers_panel.py 的 --map_json 导出）
输出：PNG 图片（布局与 draw_best_path_tree_split 一致）

用法示例：
  python plot_mothers_from_mappings.py --map_json mothers_mappings.json ^
      --out_png best_tree_7to4_highlight.png --thumb_px 110 --dpi 220 --seed 2025
"""

import argparse, json, math, random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ---------- 小工具 ----------
def _pos_from_json(obj):
    if obj is None:
        return None
    out = {}
    for k, v in obj.items():
        try:
            ik = int(k)
        except Exception:
            ik = k
        out[ik] = np.array(v, dtype=float)
    return out

def render_graph_thumb_rgba(G: nx.Graph, px=100, pos_override=None,
                            edge_width=1.2, node_size=16):
    # 使用 JSON 中的 UDG 坐标；没有则回退 KK
    if pos_override is not None:
        pos = { (int(k) if isinstance(k, str) and str(k).isdigit() else k): np.array(v, dtype=float)
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

# ---------- 从 mappings JSON 重建 levels 与 edges ----------
def load_mappings(map_json_path):
    with open(map_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # nodes_by_n: { "7": [ {g6,is3,m,degseq,pos}, ... ], ... }
    nodes_by_n = data.get("nodes_by_n", {})
    ns = sorted([int(k) for k in nodes_by_n.keys()], reverse=True)

    # 重建 levels2 / levels3 / infos（与之前绘图接口一致）
    levels2, levels3, infos = {}, {}, {}
    for n in ns:
        arr = nodes_by_n.get(str(n), [])
        left, right = [], []
        info_n = {}
        for e in arr:
            g6 = e["g6"]
            is3 = bool(e.get("is3", False))
            pos = _pos_from_json(e.get("pos", None))
            try:
                G = nx.from_graph6_bytes(g6.encode())
            except Exception:
                continue
            meta = {
                "G": G,
                "udg_pos": pos,
                "m": int(e.get("m", G.number_of_edges())),
                "degseq": tuple(e.get("degseq", [])),
                "is3": is3
            }
            info_n[g6] = meta
            if is3:
                right.append(g6)
            else:
                left.append(g6)
        # 各自按 (degseq, g6) 排序
        left.sort(key=lambda g: (info_n[g]["degseq"], g))
        right.sort(key=lambda g: (info_n[g]["degseq"], g))
        levels2[n] = left
        levels3[n] = right
        infos[n] = info_n

    # edges：按 step1/skip2 两类
    edges_raw = data.get("edges", {})
    edges_step, edges_skip2 = {}, {}
    for item in edges_raw.get("step1", []):
        n = int(item["n_parent"])
        p = item["parent_g6"]
        c = item["child_g6"]
        edges_step.setdefault(n, {}).setdefault(p, set()).add(c)
    for item in edges_raw.get("skip2", []):
        n = int(item["n_parent"])
        p = item["parent_g6"]
        c = item["child_g6"]
        edges_skip2.setdefault(n, {}).setdefault(p, set()).add(c)

    return ns, levels2, levels3, infos, edges_step, edges_skip2

def build_adjacency(levels2, levels3, edges_step, edges_skip2):
    """合并为 (n,g6) -> [(n_child, g6_child, mode)] 的邻接表。"""
    ns = sorted(set(list(levels2.keys()) + list(levels3.keys())), reverse=True)
    adj = {}
    for n in ns:
        for g6 in (levels2.get(n, []) + levels3.get(n, [])):
            adj[(n, g6)] = []

    # step1: n -> n-1
    for n, mapping in edges_step.items():
        n1 = n - 1
        for p6, childs in mapping.items():
            for c6 in childs:
                adj[(n, p6)].append((n1, c6, "step1"))

    # skip2: n -> n-2
    for n, mapping in edges_skip2.items():
        n2 = n - 2
        for p6, childs in mapping.items():
            for c6 in childs:
                adj[(n, p6)].append((n2, c6, "skip2"))

    # 邻接中按优先级排序：优先 step1，再按 g6 字典序
    for k in adj:
        adj[k].sort(key=lambda t: (0 if t[2]=="step1" else 1, t[1]))
    return adj

def choose_random_top_and_path(ns, levels2, levels3, adj, target_n=4, seed=None):
    """
    在最上层 n_max 的所有候选起点中（左列∪右列）随机选择一个起点，然后找一条到 n=target_n 的路径：
      - 先贪心优先 step1（n-1），若无则用 skip2（n-2）
      - 若贪心没走到 target_n，再 DFS 找一条（通常用不到）
    seed: 随机种子（可复现）
    """
    if not ns:
        return []
    if seed is not None:
        random.seed(seed)

    n_max = max(ns)
    candidates = (levels2.get(n_max, []) + levels3.get(n_max, []))
    if not candidates:
        return []

    start = random.choice(candidates)

    # 贪心“逐层向下”
    path = [(n_max, start)]
    n_cur, g_cur = n_max, start
    while n_cur > target_n:
        # 按 step1 优先（build_adjacency 已经把 step1 优先排列）
        nexts = [t for t in adj.get((n_cur, g_cur), []) if target_n <= t[0] < n_cur]
        if not nexts:
            break
        n_next, g_next, _mode = nexts[0]
        path.append((n_next, g_next))
        n_cur, g_cur = n_next, g_next

    # 若没到 target_n，补 DFS 找一条
    if path[-1][0] != target_n:
        visited = set(path)
        best = None
        def dfs(n, g, prefix):
            nonlocal best
            if best is not None:
                return
            if n == target_n:
                best = prefix[:]
                return
            for (nn, gg, _m) in adj.get((n, g), []):
                if (nn, gg) in visited or not (target_n <= nn < n):
                    continue
                prefix.append((nn, gg))
                dfs(nn, gg, prefix)
                prefix.pop()
        dfs(path[-1][0], path[-1][1], path[:])
        if best is not None:
            path = best

    return path

# ---------- 绘图（与原版布局一致 + 绿色高亮路径） ----------
def draw_tree_with_highlight(ns, levels2, levels3, infos, edges_step, edges_skip2,
                             out_png="best_tree_7to4_highlight.png",
                             thumb_px=110, row_gap=2.3, col_gap=1.6,
                             mid_gap=3.0, label_col_width=1.6,
                             dpi=220,
                             lw_blue=0.9, lw_red=1.2,
                             # 高亮样式
                             highlight_edge_lw=2.0, highlight_edge_color="#2ca02c",
                             highlight_box_ec="#2ca02c", highlight_box_lw=1.8,
                             seed=None):
    ns = sorted(set(list(levels2.keys())+list(levels3.keys())), reverse=True)
    left_max = max((len(levels2.get(n, [])) for n in ns), default=0)
    right_max = max((len(levels3.get(n, [])) for n in ns), default=0)

    # 坐标布局（节点中心的数据坐标）
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
    height_units = (len(ns)-1)*row_gap + 6.0

    fig = plt.figure(figsize=(max(20.0, width_units), max(12.0, height_units)), dpi=dpi)
    ax = fig.add_subplot(111); ax.axis("off")

    # 行标签
    for r, n in enumerate(ns):
        ly = -r*row_gap
        ax.text(-label_col_width, ly, f"n={n}", ha="right", va="center", fontsize=11, color="0.35")

    # 连接辅助：返回顶/底中心（数据坐标）
    def connect_center(ab, top=False):
        bb = ab.get_window_extent(renderer=renderer)
        cx = 0.5*(bb.x0+bb.x1)
        cy = (bb.y1 if top else bb.y0)
        return ax.transData.inverted().transform((cx, cy))

    # 缩略图
    thumbs = {}
    def get_thumb(n, g6):
        key=(n,g6)
        if key in thumbs: return thumbs[key]
        img = render_graph_thumb_rgba(infos[n][g6]["G"], px=thumb_px, pos_override=infos[n][g6].get("udg_pos"))
        thumbs[key]=img; return img

    # 节点框（普通）
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

    # 锁定坐标范围
    xs=[x for (x,_) in pos.values()]; ys=[y for (_,y) in pos.values()]
    ax.set_xlim(-label_col_width-1.0, (max(xs) if xs else 0)+1.0)
    ax.set_ylim((min(ys) if ys else 0)-1.0, 1.0)
    ax.set_aspect("auto")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # 先画蓝/红边
    for n, mapping in edges_step.items():
        n1 = n-1
        for p6, childs in mapping.items():
            if (n,p6) not in boxes: continue
            sx, sy = connect_center(boxes[(n,p6)], top=False)
            for c6 in childs:
                if (n1,c6) not in boxes: continue
                tx, ty = connect_center(boxes[(n1,c6)], top=True)
                ax.annotate("", xy=(tx,ty), xytext=(sx,sy),
                            arrowprops=dict(arrowstyle='-|>', lw=lw_blue, color='#1f77b4',
                                            alpha=0.9, shrinkA=0, shrinkB=0),
                            zorder=2.0)
    for n, mapping in edges_skip2.items():
        n2 = n-2
        for p6, childs in mapping.items():
            if (n,p6) not in boxes: continue
            sx, sy = connect_center(boxes[(n,p6)], top=False)
            for c6 in childs:
                if (n2,c6) not in boxes: continue
                tx, ty = connect_center(boxes[(n2,c6)], top=True)
                ax.annotate("", xy=(tx,ty), xytext=(sx,sy),
                            arrowprops=dict(arrowstyle='-|>', lw=lw_red, color='#d62728',
                                            alpha=0.95, shrinkA=0, shrinkB=0),
                            zorder=2.0)

    # ===== 绿色高亮路径 =====
    adj = build_adjacency(levels2, levels3, edges_step, edges_skip2)
    path = choose_random_top_and_path(ns, levels2, levels3, adj, target_n=4, seed=seed)

    # 高亮边
    for i in range(len(path)-1):
        n0,g0 = path[i]
        n1,g1 = path[i+1]
        if (n0,g0) not in boxes or (n1,g1) not in boxes: continue
        sx, sy = connect_center(boxes[(n0,g0)], top=False)
        tx, ty = connect_center(boxes[(n1,g1)], top=True)
        ax.annotate("", xy=(tx,ty), xytext=(sx,sy),
                    arrowprops=dict(arrowstyle='-|>', lw=highlight_edge_lw,
                                    color=highlight_edge_color,
                                    alpha=0.95, shrinkA=0, shrinkB=0),
                    zorder=4.0)

    # 高亮节点（用加粗绿色边框覆盖一层）
    for (n,g6) in path:
        if (n,g6) not in boxes: continue
        x,y = pos[(n,g6)]
        ab = AnnotationBbox(OffsetImage(get_thumb(n,g6), zoom=1.0), (x,y),
                            frameon=True, pad=0.02,
                            bboxprops=dict(fc="white", ec=highlight_box_ec, lw=highlight_box_lw),
                            zorder=4.5)
        ax.add_artist(ab)

    # 图例
    ax.plot([], [], color='#1f77b4', lw=lw_blue, label='删1点（n→n-1）')
    ax.plot([], [], color='#d62728', lw=lw_red,   label='删2点（n→n-2，仅3-连通）')
    ax.plot([], [], color=highlight_edge_color, lw=highlight_edge_lw, label='高亮路径（顶层随机）')
    ax.legend(loc='upper right', frameon=False, fontsize=10)

    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[Output] highlighted best split path tree -> {out_png}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Draw split best path tree from mothers_mappings.json and highlight a random top-layer path.")
    ap.add_argument("--map_json", required=True, help="path to mothers_mappings.json (exported by plot_mothers_panel.py --map_json)")
    ap.add_argument("--out_png", default="best_tree_7to4_highlight.png", help="output PNG path")
    ap.add_argument("--thumb_px", type=int, default=110)
    ap.add_argument("--row_gap", type=float, default=2.3)
    ap.add_argument("--col_gap", type=float, default=1.6)
    ap.add_argument("--mid_gap", type=float, default=3.0)
    ap.add_argument("--label_col", type=float, default=1.6)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--seed", type=int, default=None, help="random seed for choosing the top-layer start graph (for reproducibility)")
    args = ap.parse_args()

    ns, levels2, levels3, infos, edges_step, edges_skip2 = load_mappings(args.map_json)
    draw_tree_with_highlight(
        ns, levels2, levels3, infos, edges_step, edges_skip2,
        out_png=args.out_png,
        thumb_px=args.thumb_px, row_gap=args.row_gap, col_gap=args.col_gap,
        mid_gap=args.mid_gap, label_col_width=args.label_col, dpi=args.dpi,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
