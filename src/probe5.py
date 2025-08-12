# probe5.py
from pathlib import Path
import networkx as nx
import pickle
from data5 import load_graph_files, layer_widths
from utils5 import topological_layers

from config5 import DATA_DIR

def read_any_gpickle(path):
    try:
        from networkx.readwrite.gpickle import read_gpickle as nx_read_gpickle
        return nx_read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

files = load_graph_files(DATA_DIR)
print(f"Found {len(files)} files under {DATA_DIR}")

bad = 0
for i, p in enumerate(files[:50]):   # 先看前50个
    G = read_any_gpickle(p)
    if not nx.is_directed(G):
        G = G.to_directed()
    E = G.number_of_edges()
    N = G.number_of_nodes()
    try:
        layers = topological_layers(G)
        L = len(layers)
        W = max(layer_widths(layers)) if layers else 0
    except Exception as e:
        L = -1
        W = -1
    print(f"[{i:03d}] {p.name} | N={N}, E={E}, L={L}, W={W}")
    if E == 0 or L <= 1:
        bad += 1

print(f"Potentially unusable (E==0 or L<=1): {bad}/{len(files[:50])}")
