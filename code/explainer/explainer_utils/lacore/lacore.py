from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from array import array
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

_C_LIB = None


def _build_shared_lib(src_path: Path, build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    system = platform.system().lower()
    if system == "darwin":
        lib_name = "lacore.dylib"
        cmd = ["cc", "-O3", "-std=c99", "-fPIC", "-dynamiclib", "-o"]
    else:
        lib_name = "lacore.so"
        cmd = ["cc", "-O3", "-std=c99", "-fPIC", "-shared", "-o"]
    lib_path = build_dir / lib_name
    if lib_path.exists() and lib_path.stat().st_mtime >= src_path.stat().st_mtime:
        return lib_path
    env = os.environ.copy()
    env.setdefault("TMPDIR", str(build_dir))
    subprocess.run(cmd + [str(lib_path), str(src_path)], check=True, env=env)
    return lib_path


def _load_c_lacore() -> Optional[ctypes.CDLL]:
    global _C_LIB
    if _C_LIB is not None:
        return _C_LIB
    here = Path(__file__).resolve().parent
    src_path = here / "lacore.c"
    if not src_path.exists():
        return None
    try:
        lib_path = _build_shared_lib(src_path, here / "build")
        lib = ctypes.CDLL(str(lib_path))
    except Exception:
        return None
    lib.lacore_run.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.lacore_run.restype = ctypes.c_int
    lib.lacore_last_best_sl.argtypes = []
    lib.lacore_last_best_sl.restype = ctypes.c_double
    _C_LIB = lib
    return lib


def _edges_to_adj1(
    edges: List[Tuple[int, int]], num_nodes: Optional[int] = None
) -> Tuple[List[List[int]], int]:
    max_node = -1
    seen = set()
    for u, v in edges:
        if u == v:
            continue
        if u > max_node:
            max_node = u
        if v > max_node:
            max_node = v
    if num_nodes is None:
        n = max_node + 1 if max_node >= 0 else 0
    else:
        n = int(num_nodes)
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        aa, bb = a + 1, b + 1
        if aa <= n and bb <= n:
            adj[aa].append(bb)
            adj[bb].append(aa)
    return adj, n


def _build_csr(adj: List[List[int]]) -> Tuple[array, array]:
    n = len(adj) - 1
    offsets = array("i", [0] * (n + 2))
    total = 0
    for i in range(1, n + 1):
        offsets[i] = total
        total += len(adj[i])
    offsets[n + 1] = total
    neighbors = array("i", [0] * total)
    pos = 0
    for i in range(1, n + 1):
        for v in adj[i]:
            neighbors[pos] = v
            pos += 1
    return offsets, neighbors


def _lacore_c_edges(
    edges: List[Tuple[int, int]], eps: float, num_nodes: Optional[int] = None
) -> Tuple[List[int], float]:
    adj, n = _edges_to_adj1(edges, num_nodes)
    if n == 0:
        return [], 0.0
    lib = _load_c_lacore()
    if lib is None:
        raise RuntimeError("LaCore C library unavailable")
    offsets, neighbors = _build_csr(adj)
    offsets_c = (ctypes.c_int * len(offsets)).from_buffer(offsets)
    neighbors_c = (ctypes.c_int * len(neighbors)).from_buffer(neighbors)
    out_c = (ctypes.c_int * n)()
    count = lib.lacore_run(n, offsets_c, neighbors_c, ctypes.c_double(eps), out_c, n)
    if count < 0:
        raise RuntimeError(f"lacore_run failed with code {count}")
    nodes = [out_c[i] - 1 for i in range(count)]
    nodes.sort()
    score = float(lib.lacore_last_best_sl())
    return nodes, score


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n + 1))
        self.size = [0] * (n + 1)
        self.made = [False] * (n + 1)
        self.Q = [0.0] * (n + 1)

    def make_if_needed(self, v: int):
        if not self.made[v]:
            self.made[v] = True
            self.parent[v] = v
            self.size[v] = 1
            self.Q[v] = 0.0

    def find(self, v: int) -> int:
        if not self.made[v]:
            return v
        while self.parent[v] != v:
            self.parent[v] = self.parent[self.parent[v]]
            v = self.parent[v]
        return v

    def union(self, a: int, b: int) -> int:
        self.make_if_needed(a)
        self.make_if_needed(b)
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.Q[ra] += self.Q[rb]
        return ra


def _rmc_single_cluster_from_adj(
    adj1: List[List[int]], epsilon: float
) -> Tuple[Set[int], float]:
    import heapq

    n = len(adj1) - 1

    deg0 = [0] * (n + 1)
    for i in range(1, n + 1):
        deg0[i] = len(adj1[i])
    pq = [(deg0[i], i) for i in range(1, n + 1)]
    heapq.heapify(pq)
    peel_stack: List[int] = []

    while pq:
        d, u = heapq.heappop(pq)
        if d != deg0[u]:
            continue
        peel_stack.append(u)
        for v in adj1[u]:
            if deg0[v] > 0:
                deg0[v] -= 1
                heapq.heappush(pq, (deg0[v], v))
        deg0[u] = 0

    add_order = [0] * n
    idx = [0] * (n + 1)
    for t in range(n):
        u = peel_stack.pop()
        add_order[t] = u
        idx[u] = t

    succ = [[] for _ in range(n + 1)]
    pred = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        for v in adj1[u]:
            if u < v:
                if idx[u] < idx[v]:
                    succ[u].append(v)
                    pred[v].append(u)
                else:
                    succ[v].append(u)
                    pred[u].append(v)
    for v in range(1, n + 1):
        if len(succ[v]) > 1:
            succ[v].sort(key=lambda w: idx[w])

    dsu = _DSU(n)
    deg = [0] * (n + 1)
    pred_sum = [0] * (n + 1)

    best_sl = 0.0
    best_root = 0
    best_component: Set[int] = set()

    def sum_succ_until(v: int, T: int) -> int:
        s = 0
        for w in succ[v]:
            if idx[w] >= T:
                break
            s += deg[w]
        return s

    def snapshot_component(root: int) -> Set[int]:
        comp = set()
        for i in range(1, n + 1):
            if dsu.made[i] and dsu.find(i) == root:
                comp.add(i)
        return comp

    for u in add_order:
        dsu.make_if_needed(u)

        ru = dsu.find(u)
        sl = dsu.size[ru] / (dsu.Q[ru] + epsilon)
        if sl > best_sl:
            best_sl = sl
            best_root = ru
            best_component = snapshot_component(ru)

        Su = 0
        Tu = idx[u]

        for v in pred[u]:
            a = deg[u]
            b = deg[v]

            Sv = pred_sum[v] + sum_succ_until(v, Tu)

            dQu = 2 * a * a - 2 * Su + a
            dQv = 2 * b * b - 2 * Sv + b
            edge_term = (a - b) * (a - b)

            ru = dsu.find(u)
            rv = dsu.find(v)

            dsu.Q[ru] += float(dQu)
            dsu.Q[rv] += float(dQv)

            if ru != rv:
                r = dsu.union(ru, rv)
                dsu.Q[r] += float(edge_term)
            else:
                r = ru
                dsu.Q[r] += float(edge_term)

            sl = dsu.size[r] / (dsu.Q[r] + epsilon)
            if sl > best_sl:
                best_sl = sl
                best_component = snapshot_component(r)

            deg[u] += 1
            deg[v] += 1

            for y in succ[u]:
                pred_sum[y] += 1
            for y in succ[v]:
                pred_sum[y] += 1

            Su += deg[v]

    return best_component, float(best_sl)


def _lacore_py_edges(
    edges: List[Tuple[int, int]], eps: float, num_nodes: Optional[int] = None
) -> Tuple[List[int], float]:
    adj, n = _edges_to_adj1(edges, num_nodes)
    if n == 0:
        return [], 0.0
    nodes_1b, best_sl = _rmc_single_cluster_from_adj(adj, eps)
    nodes = sorted([u - 1 for u in nodes_1b])
    return nodes, best_sl


def generate_lacore_cluster(
    edges: List[Tuple[int, int]],
    epsilon: float,
    num_nodes: Optional[int] = None,
) -> Dict:
    try:
        eps = float(epsilon)
    except Exception:
        eps = float(str(epsilon))

    if not edges and (num_nodes is None or num_nodes <= 0):
        return {"seed_nodes": [], "score": 0.0}

    try:
        seed_nodes_0b, best_sl = _lacore_c_edges(edges, eps, num_nodes)
    except Exception:
        seed_nodes_0b, best_sl = _lacore_py_edges(edges, eps, num_nodes)
    return {"seed_nodes": seed_nodes_0b, "score": best_sl}
