# show_fixed.py —— 可视化所有边，并用“弧线”突出显示跨层（跳层）边
import os, json, gzip, pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph

# ======= 在这里填入你的图路径（可多个）======
PATHS = [
    r"E:\DAG\src\generated5\gen_like_1754996671.gpickle",
    r"E:\PythonProject5\data\small\Tau_0_Tau_1.gpickle",
]
SAVE_PNG = False   # True 则保存 PNG 到同目录，后缀 _vis.png
SHOW_TIME = True   # 边上标注 critical_time
# ====================================

TIME_KEYS = ["critical_time", "time", "weight", "t", "C", "label"]

def normalize_edge_time_key(G: nx.DiGraph, default: float = 0.0) -> None:
    """将各种时间键规范到 G[u][v]['critical_time']"""
    for u, v, d in G.edges(data=True):
        val = None
        for k in TIME_KEYS:
            if k in d and d[k] is not None:
                val = d[k]
                break
        if val is None:
            val = default
        d["critical_time"] = float(val)

def _load_gpickle(path):
    read_gpickle = getattr(nx, "read_gpickle", None)
    if read_gpickle is not None:
        return read_gpickle(path)
    with open(path, "rb") as f:
        return pickle.load(f)

# 2) 读取 .gpickle.gz
def _load_gpickle_gz(path):
    read_gpickle = getattr(nx, "read_gpickle", None)
    with gzip.open(path, "rb") as f:
        if read_gpickle is not None:
            return read_gpickle(f)  # 支持文件对象
        return pickle.load(f)

# 3) 修正 load_graph（只改这几个分支即可）
def load_graph(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    name = p.name.lower()
    if name.endswith(".gpickle"):
        G = _load_gpickle(p)
    elif name.endswith(".gpickle.gz"):
        G = _load_gpickle_gz(p)
    elif name.endswith(".json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        G = json_graph.node_link_graph(data, directed=True, multigraph=False)
    elif name.endswith(".json.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.loads(f.read())
        G = json_graph.node_link_graph(data, directed=True, multigraph=False)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    if not isinstance(G, nx.DiGraph):
        G = G.to_directed()
    # 这里保持你原来的 normalize_edge_time_key 调用
    normalize_edge_time_key(G)
    return G

def longest_path_layers(G: nx.DiGraph) -> Dict[Any, int]:
    """按最长路径距离分层（topo DP）"""
    order = list(nx.topological_sort(G))
    L = {u: 0 for u in order}
    for u in order:
        for v in G.successors(u):
            L[v] = max(L[v], L[u] + 1)
    return L

def build_positions(L: Dict[Any, int]) -> Dict[Any, Tuple[float, float]]:
    """按层等距排布，层向下递增"""
    layers: Dict[int, List[Any]] = {}
    for n, lv in L.items():
        layers.setdefault(lv, []).append(n)
    for nodes in layers.values():
        nodes.sort(key=lambda x: str(x))
    pos: Dict[Any, Tuple[float, float]] = {}
    y_gap = 1.5
    for lv, nodes in layers.items():
        k = len(nodes)
        xs = [ (i+1)/(k+1) for i in range(k) ]
        y = -lv * y_gap
        for x, n in zip(xs, nodes):
            pos[n] = (x, y)
    return pos

def draw_graph(G: nx.DiGraph, title: str = "", show_edge_time: bool = True):
    """显示所有边；跳层（ΔL>1）画弧线并标注 ΔL"""
    L = longest_path_layers(G)
    for n, lv in L.items():
        G.nodes[n]["layer"] = lv
    pos = build_positions(L)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontsize=12)

    # 画节点（按层着不同 marker）
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=400, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    # 画边：ΔL==1 直线；ΔL>1 弧线；ΔL<=0（异常）红虚线
    def _arc(u_xy, v_xy, rad):
        from matplotlib.patches import FancyArrowPatch
        return FancyArrowPatch(
            u_xy, v_xy, arrowstyle="-|>", mutation_scale=10,
            connectionstyle=f"arc3,rad={rad}", linewidth=1.2, alpha=0.9)

    # 为了避免多条长跳重合，按 (ΔL, idx) 设不同弧度
    long_count = 0
    for u, v, d in G.edges(data=True):
        lu, lv = L[u], L[v]
        du = lu - lv
        dv = lv - lu  # ΔL
        u_xy, v_xy = pos[u], pos[v]
        t = d.get("critical_time", 0.0)
        if dv == 1:
            # 相邻层：直线
            ax.annotate(
                "", xy=v_xy, xytext=u_xy,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, alpha=0.9))
        elif dv > 1:
            # 跳层：弧线（弧度跟 ΔL 相关）
            long_count += 1
            rad = 0.25 + 0.08 * min(dv-1, 3)  # ΔL 越大弧度越大（上限 3）
            # 奇偶交替方向，减少重叠
            if long_count % 2 == 0:
                rad = -rad
            patch = _arc(u_xy, v_xy, rad)
            ax.add_patch(patch)
            # ΔL 标签
            mid = ((u_xy[0]+v_xy[0])/2, (u_xy[1]+v_xy[1])/2 + (0.6 if rad>0 else -0.6))
            ax.text(mid[0], mid[1], f"ΔL={dv}", fontsize=8, ha="center", va="center")
        else:
            # 非法方向（应当很少见）：红虚线
            ax.annotate(
                "", xy=v_xy, xytext=u_xy,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, ls="--", color="r", alpha=0.7))

    # 边时间标签（放在中点）
    if show_edge_time:
        for u, v, d in G.edges(data=True):
            u_xy, v_xy = pos[u], pos[v]
            mx, my = (u_xy[0]+v_xy[0])/2, (u_xy[1]+v_xy[1])/2
            t = d.get("critical_time", 0.0)
            ax.text(mx, my, f"{t:.1f}", fontsize=8, alpha=0.9)

    ax.axis("off")
    fig.tight_layout()
    return fig, ax

def total_time(G: nx.DiGraph) -> float:
    return sum(G[u][v].get("critical_time", 0.0) for u, v in G.edges())

def print_time_stats(G: nx.DiGraph, name: str = ""):
    # 关键路径（按边时间）最长路径
    order = list(nx.topological_sort(G))
    dist = {u: 0.0 for u in order}
    for u in order:
        for v in G.successors(u):
            w = float(G[u][v].get("critical_time", 0.0))
            dist[v] = max(dist[v], dist[u] + w)
    longest = max(dist.values()) if dist else 0.0
    print(f"[{name}] N={G.number_of_nodes()}, E={G.number_of_edges()}, "
          f"TotalT={total_time(G):.1f}, LongestPathT={longest:.1f}")

def main():
    if not PATHS:
        print("PATHS 为空，请先在文件顶部填写你的图路径。")

        return
    for p in PATHS:
        G = load_graph(p)
        name = os.path.basename(p)
        fig, ax = draw_graph(G, title=name, show_edge_time=SHOW_TIME)
        print_time_stats(G, name=name)
        if SAVE_PNG:
            out = os.path.splitext(p)[0] + "_vis.png"
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"Saved: {out}")
    if not SAVE_PNG:
        plt.show()

if __name__ == "__main__":
    main()