"""Value-guided search: batched weighted A* (DeepCubeA) and greedy best-first.

A *value_fn* maps a torch uint8 tensor of states (N, 54), on the search device, to (N,) cost-to-go estimates h(s)
(any float dtype; ``rubiks.value_fn.make_value_fn`` wraps a trained value network, clamped at 0). NaN is an error.

:func:`astar_solve` is batched weighted A*:

* priority ``f = weight * g + h(s)`` (``weight`` < 1 trades optimality for fewer expansions; ``weight`` = 1 is
  plain A*; ``weight`` = 0 is pure best-first on h);
* the open list is a binary heap (``heapq``) of ``(f, g, node_id)`` tuples; equal f pops the smaller g first;
* ``closed`` is a dict ``state bytes -> best g found so far`` over EVERY generated state (open or expanded). A
  child is kept only if its state is new or reached with a strictly smaller g than before (re-opening: the
  better copy is pushed as a new node, and the old heap entry becomes stale and is skipped when popped);
* each iteration pops up to ``batch`` non-stale nodes, generates all their children under the move filter
  (numpy gather over ``cube.MOVE_PERMS``), and stops if any child is the solved state (the solved child with the
  smallest g in that batch is returned). Otherwise it drops the children that ``closed`` rejects and evaluates h
  on all the remaining children in ONE value_fn call on the device, then pushes them;
* it gives up (``moves=None``) after ``max_nodes`` expansions, or if the open list empties;
* nodes live in growable numpy arrays (states, parent, move, g, last face, lock axis); paths are rebuilt from the
  parent pointers. There is no Python object per node apart from the heap tuple and the ``closed`` entry.

Move filter (the dataset's canonical scramble filter, as ``rubiks.search``): a child's move is never on the face
of its node's last move, and after two moves on opposite faces (``U D``) neither face of that axis may follow.
Every sequence it removes has a strictly shorter equivalent (``X X'`` -> nothing, ``X X`` -> ``X2``, ``U D U'``
-> ``D``), so it stays compatible with the duplicate detection: a pruned child's state is always reachable with a
smaller g through a branch that is not pruned.

Every returned path is verified with ``cube.apply_sequence`` + ``cube.is_solved`` (``rubiks.search._verify``); a
non-solving path raises ``AssertionError`` (an explicit raise, active under ``python -O``).

Counts (:class:`SearchResult`): ``iterations`` is the number of batches popped (A*) or steps taken (greedy), and
nothing more. ``value_calls`` is the number of value_fn calls actually made, ``<= iterations``: no call is made
by the batch or step that generates the solved child (it returns first) nor, in A*, by a batch whose children
the closed dict all rejects.

:func:`greedy_best_first` is the "network alone" baseline: from the current state take the allowed child with the
lowest h (a solved child is taken at once: J(solved) := 0), until solved or ``max_depth`` moves. It keeps no
visited set, so it can cycle; that is part of what the baseline measures.
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from . import cube
from .search import MOVE_FACE, _check_state, _verify

ValueFn = Callable[[torch.Tensor], torch.Tensor]

MOVE_AXIS = MOVE_FACE % 3
_SOLVED = np.array(cube.SOLVED, dtype=np.uint8)


@dataclass
class SearchResult:
    moves: list[int] | None  # the verified solution, or None (node cap reached / open list empty / max_depth)
    nodes_expanded: int      # nodes popped and expanded (A*), or states stepped from (greedy)
    nodes_generated: int     # children produced by those expansions, after the move filter, before deduplication
    ms: float                # wall-clock milliseconds for the whole call
    nodes_evaluated: int = 0  # children whose h was computed (new or improved states; A*) / all children (greedy)
    iterations: int = 0       # batches popped (A*) / steps taken (greedy)
    value_calls: int = 0      # value_fn calls actually made, <= iterations (see "Counts" in the module docstring)

    @property
    def solved(self) -> bool:
        return self.moves is not None


def _move_perms() -> np.ndarray:
    """The (18, 54) move permutations the searches expand with (a hook for the tests)."""
    return cube.MOVE_PERMS


def _resolve(value_fn: ValueFn, device) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device(getattr(value_fn, "device", "cpu"))


def _allowed_np(last_face: np.ndarray, lock_axis: np.ndarray) -> np.ndarray:
    """(B, 18) bool: the moves each node may take (see the module docstring); -1 = none."""
    ok = MOVE_FACE[None, :] != last_face[:, None]
    ok &= (lock_axis[:, None] < 0) | (MOVE_AXIS[None, :] != lock_axis[:, None])
    return ok


def _next_lock(parent_face: np.ndarray, new_face: np.ndarray) -> np.ndarray:
    """Axis locked after a move: the move's axis if it follows a move on the opposite face, else -1."""
    opposite = (parent_face >= 0) & ((parent_face % 3) == (new_face % 3))
    return np.where(opposite, new_face % 3, -1).astype(np.int8)


def _heuristic(value_fn: ValueFn, states: np.ndarray, dev: torch.device) -> np.ndarray:
    """h for uint8 (M, 54) numpy states: one value_fn call; float64 (M,) on the CPU. Raises on a bad shape or NaN."""
    with torch.inference_mode():
        out = value_fn(torch.from_numpy(np.ascontiguousarray(states)).to(dev))
    h = out.detach().reshape(-1).cpu().numpy().astype(np.float64)  # CPU first, then widen (MPS has no float64)
    if h.shape != (states.shape[0],):
        raise ValueError(f"value_fn returned {h.shape[0]} values for {states.shape[0]} states")
    if np.isnan(h).any():
        raise ValueError("value_fn returned NaN")
    return h


class _Nodes:
    """Growable node arrays: states, parent, move, g, last face, lock axis (index = node id)."""

    def __init__(self, capacity: int = 4096):
        self.n = 0
        self.states = np.empty((capacity, cube.N_FACELETS), dtype=np.uint8)
        self.parent = np.empty(capacity, dtype=np.int64)
        self.move = np.empty(capacity, dtype=np.int8)
        self.g = np.empty(capacity, dtype=np.int32)
        self.face = np.empty(capacity, dtype=np.int8)
        self.lock = np.empty(capacity, dtype=np.int8)

    def _grow(self, need: int) -> None:
        cap = len(self.parent)
        if need <= cap:
            return
        new_cap = max(need, 2 * cap)
        for name in ("states", "parent", "move", "g", "face", "lock"):
            old = getattr(self, name)
            arr = np.empty((new_cap, *old.shape[1:]), dtype=old.dtype)
            arr[:self.n] = old[:self.n]
            setattr(self, name, arr)

    def add(self, states, parent, move, g, face, lock) -> np.ndarray:
        k = len(parent)
        self._grow(self.n + k)
        sl = slice(self.n, self.n + k)
        self.states[sl] = states
        self.parent[sl] = parent
        self.move[sl] = move
        self.g[sl] = g
        self.face[sl] = face
        self.lock[sl] = lock
        ids = np.arange(self.n, self.n + k, dtype=np.int64)
        self.n += k
        return ids

    def path_to(self, node: int) -> list[int]:
        path = []
        while self.parent[node] >= 0:
            path.append(int(self.move[node]))
            node = int(self.parent[node])
        path.reverse()
        return path


def astar_solve(value_fn: ValueFn, state, weight: float = 0.6, batch: int = 200, max_nodes: int = 200000,
                device: torch.device | str | None = None) -> SearchResult:
    """Batched weighted A* from ``state`` (uint8 (54,)). Returns a verified solution, or ``moves=None`` after
    ``max_nodes`` expansions (or an empty open list)."""
    t0 = time.perf_counter()
    s0 = _check_state(state)
    weight = float(weight)
    if not (math.isfinite(weight) and weight >= 0) or batch < 1 or max_nodes < 0:
        raise ValueError("need a finite weight >= 0, batch >= 1 and max_nodes >= 0")
    if cube.is_solved(s0):
        return SearchResult([], 0, 0, (time.perf_counter() - t0) * 1e3)
    dev = _resolve(value_fn, device)
    perms = _move_perms()
    n_f = cube.N_FACELETS

    nodes = _Nodes()
    nodes.add(s0[None], np.array([-1]), np.array([-1]), np.array([0]), np.array([-1]), np.array([-1]))
    closed: dict[bytes, int] = {s0.tobytes(): 0}
    heap: list[tuple[float, int, int]] = [(0.0, 0, 0)]  # the root is the only open node: its h is not needed
    expanded = generated = evaluated = iterations = calls = 0

    while heap and expanded < max_nodes:
        take = min(batch, max_nodes - expanded)
        popped: list[int] = []
        while heap and len(popped) < take:
            _, g, nid = heapq.heappop(heap)
            if closed[nodes.states[nid].tobytes()] < g:  # stale: the state was re-opened with a smaller g
                continue
            popped.append(nid)
        if not popped:
            break
        iterations += 1
        ids = np.array(popped, dtype=np.int64)
        expanded += len(ids)

        pi, mv = np.nonzero(_allowed_np(nodes.face[ids], nodes.lock[ids]))  # parent-major order
        par = ids[pi]
        children = np.take_along_axis(nodes.states[par], perms[mv], axis=1)
        cg = nodes.g[par].astype(np.int64) + 1
        generated += len(mv)

        solved = np.flatnonzero(np.all(children == _SOLVED, axis=1))
        if solved.size:
            j = int(solved[np.argmin(cg[solved])])  # first solved child with the smallest g
            path = nodes.path_to(int(par[j])) + [int(mv[j])]
            _verify(s0, path)
            return SearchResult(path, expanded, generated, (time.perf_counter() - t0) * 1e3, evaluated, iterations,
                                calls)

        # Duplicate detection, smallest g first, so a state repeated within the batch keeps its best copy only.
        order = np.argsort(cg, kind="stable")
        raw = children.tobytes()
        keep = []
        for j in order.tolist():
            key = raw[j * n_f:(j + 1) * n_f]
            gj = int(cg[j])
            old = closed.get(key)
            if old is not None and old <= gj:
                continue
            closed[key] = gj
            keep.append(j)
        if not keep:
            continue
        kp = np.array(keep, dtype=np.int64)
        kstates = children[kp]
        h = _heuristic(value_fn, kstates, dev)
        calls += 1
        evaluated += len(kp)
        kmv = mv[kp]
        kface = (kmv // 3).astype(np.int8)
        kpar = par[kp]
        kg = cg[kp]
        new_ids = nodes.add(kstates, kpar, kmv, kg, kface, _next_lock(nodes.face[kpar], kface))
        f = weight * kg + h
        for fi, gi, ni in zip(f.tolist(), kg.tolist(), new_ids.tolist()):
            heapq.heappush(heap, (fi, gi, ni))

    return SearchResult(None, expanded, generated, (time.perf_counter() - t0) * 1e3, evaluated, iterations, calls)


def greedy_best_first(value_fn: ValueFn, state, max_depth: int = 40,
                      device: torch.device | str | None = None) -> SearchResult:
    """Follow the allowed child with the lowest h from ``state`` until solved or ``max_depth`` moves (same move
    filter as A*; ties: the lowest move index). Returns a verified solution or ``moves=None``."""
    t0 = time.perf_counter()
    s0 = _check_state(state)
    if max_depth < 0:
        raise ValueError("need max_depth >= 0")
    if cube.is_solved(s0):
        return SearchResult([], 0, 0, (time.perf_counter() - t0) * 1e3)
    dev = _resolve(value_fn, device)
    perms = _move_perms()
    cur = s0.copy()
    path: list[int] = []
    last_face, lock_axis = -1, -1
    generated = evaluated = calls = 0
    for depth in range(1, max_depth + 1):
        ok = _allowed_np(np.array([last_face]), np.array([lock_axis]))[0]
        allowed = np.flatnonzero(ok)
        children = cur[perms[allowed]]
        generated += len(allowed)
        solved = np.flatnonzero(np.all(children == _SOLVED, axis=1))
        if solved.size:
            k = int(solved[0])
        else:
            h = _heuristic(value_fn, children, dev)
            calls += 1
            evaluated += len(allowed)
            k = int(np.argmin(h))
        m = int(allowed[k])
        face = m // 3
        lock_axis = face % 3 if last_face >= 0 and last_face % 3 == face % 3 else -1
        last_face = face
        cur = children[k]
        path.append(m)
        if solved.size:
            _verify(s0, path)
            return SearchResult(path, depth, generated, (time.perf_counter() - t0) * 1e3, evaluated, depth, calls)
    return SearchResult(None, max_depth, generated, (time.perf_counter() - t0) * 1e3, evaluated, max_depth, calls)
