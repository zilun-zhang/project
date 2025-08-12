# data5.py —— read graph (A_target, T_target, widths, s_vec, total_T)
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict, Any
import json, pickle, gzip

import numpy as np
import torch
import networkx as nx
from torch.utils.data import Dataset
from networkx.readwrite import json_graph

from utils5 import topological_layers
from config5 import NORM_N, NORM_E, NORM_L, NORM_W, NORM_T


TIME_KEYS = ["critical_time", "time", "weight", "t", "C", "label"]

def edge_time_from_attr(d: dict, default: float = 1.0) -> float:
    for k in TIME_KEYS:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)


def _read_gpickle_any(p: Path):
    try:
        from networkx.readwrite.gpickle import read_gpickle as _rg  
        return _rg(p)
    except Exception:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rb") as f:
            return pickle.load(f)

def read_graph_any(p: Path) -> nx.DiGraph:
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        G = json_graph.node_link_graph(data, directed=True, multigraph=False)
    else:
        G = _read_gpickle_any(p)
    if not isinstance(G, nx.DiGraph):
        G = G.to_directed()
    return G


def graph_to_mats(G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor, float]:
    assert nx.is_directed_acyclic_graph(G), "must DAG"

    layers = topological_layers(G)
    widths = [len(Li) for Li in layers]
    order: List[int] = []
    for Li in layers:
        order += list(sorted(Li))
    idx_map = {n: i for i, n in enumerate(order)}
    N = len(order)

    A = np.zeros((N, N), dtype=np.float32)
    T = np.zeros((N, N), dtype=np.float32)
    total_T = 0.0

    for u, v, d in G.edges(data=True):
        i, j = idx_map[u], idx_map[v]
        A[i, j] = 1.0
        t = edge_time_from_attr(d, 1.0)  
        T[i, j] = float(t)
        total_T += float(t)

    E = int(A.sum())
    L = len(widths)
    W = max(widths) if widths else 0
    s = torch.tensor([N / NORM_N, E / NORM_E, L / NORM_L, W / NORM_W, total_T / NORM_T],
                     dtype=torch.float32)

    return torch.from_numpy(A), torch.from_numpy(T), widths, s, float(total_T)


def load_graph_files(data_dir: Path) -> List[Path]:
    data_dir = Path(data_dir)
    if data_dir.is_file():
        return [data_dir]
    exts = (".gpickle", ".json", ".gpickle.gz", ".gz")
    files: List[Path] = []
    for p in data_dir.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files

# ---------- Dataset ----------
class GraphStructureDataset5(Dataset):
    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        self.files = load_graph_files(data_dir)
        if not self.files:
            raise FileNotFoundError(f"No graphs found under: {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.files[idx]
        G = read_graph_any(p)
        A, T, widths, s, total_T = graph_to_mats(G)
        return {
            "s_vec": s,
            "A_target": A,
            "T_target": T,
            "meta": {"widths": widths, "path": str(p), "total_T": total_T},
        }

def graph_total_time(G: nx.DiGraph) -> float:
    return sum(edge_time_from_attr(d, 1.0) for _, _, d in G.edges(data=True))

