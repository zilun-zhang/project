from __future__ import annotations
import math
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import torch


def topological_layers(G: nx.DiGraph) -> List[List]:
    """Return nodes grouped by topological layers (levelized by longest path length).
    If G has cycles, this will raise. Ensure G is a DAG.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input graph is not a DAG (cycle detected).")

    order = list(nx.topological_sort(G))
    # Longest path distance (number of edges) from any source
    dist = {u: 0 for u in order}
    for u in order:
        for v in G.successors(u):
            dist[v] = max(dist[v], dist[u] + 1)
    max_rank = max(dist.values()) if dist else 0

    layers: List[List] = [[] for _ in range(max_rank + 1)]
    for u in order:
        layers[dist[u]].append(u)
    return layers


def layer_index_map(layers: List[List]) -> Dict:
    """Map node -> (layer_id, position_in_layer)."""
    mp = {}
    for li, layer in enumerate(layers):
        for pj, node in enumerate(layer):
            mp[node] = (li, pj)
    return mp


def distribute_nodes_across_layers(N: int, L: int, max_width: int) -> List[int]:
    """Heuristic split of N nodes into L layers with max layer width <= max_width.
    Returns list of widths per layer with sum == N.
    """
    if L <= 0:
        return [N]
    widths = [0] * L
    # Fill as evenly as possible, capped by max_width
    i = 0
    remaining = N
    while remaining > 0:
        if widths[i] < max_width:
            widths[i] += 1
            remaining -= 1
        i = (i + 1) % L
    return widths


def build_order_from_widths(widths: List[int]) -> List[Tuple[int, int]]:
    """Return a list of (layer_id, pos_in_layer) for node indices [0..N-1]."""
    order = []
    for li, w in enumerate(widths):
        for pj in range(w):
            order.append((li, pj))
    return order


def adjacency_from_graph(G: nx.DiGraph, layers: List[List]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (A_bin, T) in the layer order (layer0..k, pos inside layer).
    A_bin: NxN with {0,1}, T: NxN with >=0 times (edge attribute 'critical_time' else 1.0).
    """
    mp = layer_index_map(layers)
    # Build index mapping according to (layer, pos) order
    ordered_nodes = []
    for layer in layers:
        for node in layer:
            ordered_nodes.append(node)
    idx = {node: i for i, node in enumerate(ordered_nodes)}
    N = len(ordered_nodes)
    A = np.zeros((N, N), dtype=np.float32)
    T = np.zeros((N, N), dtype=np.float32)
    for u, v, data in G.edges(data=True):
        iu, iv = idx[u], idx[v]
        A[iu, iv] = 1.0
        t = data.get("critical_time", 1.0)
        T[iu, iv] = float(max(0.0, t))
    return A, T

def longest_path_time_from_mats(A_bin: np.ndarray, T: np.ndarray, widths: List[int]) -> float:
    """Compute longest path time given binary adjacency and edge-time matrix.
    Assumes nodes ordered by layers as per widths.
    """
    N = A_bin.shape[0]
    # Topological order is just 0..N-1 under our layer ordering
    dp = np.zeros((N,), dtype=np.float32)
    for j in range(N):
        # incoming i -> j
        incoming = np.where(A_bin[:, j] > 0.5)[0]
        if incoming.size > 0:
            dp[j] = np.max(dp[incoming] + T[incoming, j])
    return float(dp.max() if dp.size else 0.0)


def mask_allowed_pairs(widths: List[int]) -> np.ndarray:
    """Mask of shape NxN with 1 for allowed directed pairs (earlier layer -> later layer)."""
    order = build_order_from_widths(widths)
    N = len(order)
    mask = np.zeros((N, N), dtype=np.float32)
    for i, (li, _pi) in enumerate(order):
        for j, (lj, _pj) in enumerate(order):
            if lj > li:  # strictly later layer
                mask[i, j] = 1.0
    return mask


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device)
