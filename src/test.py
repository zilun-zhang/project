# test.py —— 路径写死：REF vs GEN 的结构+时间对比
from __future__ import annotations
from pathlib import Path
import pickle, json
import networkx as nx
from networkx.readwrite import json_graph

# ======= 写死路径（改成你的）======
REF_PATH = Path(r"E:/PythonProject5/data/small/Tau_0_Tau_1.gpickle")
GEN_PATH = Path(r"E:\DAG\src\generated5\gen_like_1754996671.gpickle")
# ==================================

TIME_KEYS = ["critical_time", "time", "weight", "t", "C", "label"]


def get_edge_time(d: dict):
    for k in TIME_KEYS:
        if k in d:
            return float(d[k])
    return None

def normalize_edge_time_key(G: nx.DiGraph) -> int:
    changed = 0
    for u, v, d in G.edges(data=True):
        if "critical_time" not in d:
            t = get_edge_time(d)
            if t is not None:
                d["critical_time"] = t
                changed += 1
    return changed

def read_graph(p: Path) -> nx.DiGraph:
    if p.suffix.lower() == ".json":
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        G = json_graph.node_link_graph(data, directed=True, multigraph=False)
    else:
        try:
            from networkx.readwrite.gpickle import read_gpickle
            G = read_gpickle(p)
        except Exception:
            with open(p, "rb") as f:
                G = pickle.load(f)
    if not isinstance(G, nx.DiGraph):
        G = G.to_directed()
    normalize_edge_time_key(G)
    return G

def time_stats(G: nx.DiGraph):
    ts = [d.get("critical_time") for _, _, d in G.edges(data=True) if "critical_time" in d]
    if not ts:
        return {"has_time": False}
    total = float(sum(ts))
    mean  = total / len(ts)
    tmin, tmax = float(min(ts)), float(max(ts))
    try:
        lp = nx.algorithms.dag.dag_longest_path_length(G, weight="critical_time", default_weight=1.0)
    except Exception:
        lp = float("nan")
    non_uniform = 0
    for u in G.nodes():
        ts_u = [d["critical_time"] for _, _, d in G.out_edges(u, data=True) if "critical_time" in d]
        if len(ts_u) >= 2 and (max(ts_u) - min(ts_u) > 1e-6):
            non_uniform += 1
    return {
        "has_time": True,
        "edges_with_time": len(ts),
        "total": total, "mean": mean, "min": tmin, "max": tmax,
        "longest_path_time": float(lp),
        "non_uniform_nodes": non_uniform,
    }

def summarize(name: str, G: nx.DiGraph):
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"== {name} ==")
    print(f"DAG: {is_dag} | N={G.number_of_nodes()} | E={G.number_of_edges()}")
    ts = time_stats(G)
    if ts["has_time"]:
        print(f"time -> edges={ts['edges_with_time']} | total={ts['total']:.3f} | "
              f"min={ts['min']:.3f} | max={ts['max']:.3f} | mean={ts['mean']:.3f} | "
              f"longest={ts['longest_path_time']:.3f} | non_uniform_nodes={ts['non_uniform_nodes']}")
    else:
        print("time -> (no critical_time on edges)")
    return ts

if __name__ == "__main__":
    assert REF_PATH.exists(), f"REF 不存在: {REF_PATH}"
    assert GEN_PATH.exists(), f"GEN 不存在: {GEN_PATH}"
    print("REF:", REF_PATH)
    print("GEN:", GEN_PATH)

    G_ref = read_graph(REF_PATH)
    G_gen = read_graph(GEN_PATH)

    ts_ref = summarize("REF", G_ref)
    ts_gen = summarize("GEN", G_gen)

    if ts_ref["has_time"] and ts_gen["has_time"]:
        diff_total = ts_gen["total"] - ts_ref["total"]
        rel = diff_total / ts_ref["total"] * 100.0 if ts_ref["total"] > 0 else float("nan")
        diff_long = ts_gen["longest_path_time"] - ts_ref["longest_path_time"]
        print("-- diff (GEN - REF) --")
        print(f"total_T: {diff_total:+.3f} ({rel:+.2f}%)")
        print(f"longest_path_time: {diff_long:+.3f}")
