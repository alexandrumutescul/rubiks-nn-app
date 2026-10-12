"""Policy-guided search: batched beam search and a greedy baseline.

A *policy_fn* maps a torch uint8 tensor of states (N, 54), on the search device, to torch log-probabilities
(N, 18) over the moves of ``rubiks.cube.MOVE_NAMES``. :func:`make_policy_fn` wraps a trained model.

Beam search keeps all states on the device (torch): each layer expands every beam state by its allowed moves
with ``torch.gather`` over ``cube.MOVE_PERMS``, scores each child by the cumulative log-probability of its path,
removes duplicate states within the layer (keeping the best-scored copy), keeps the ``width`` best, and stops at
the first layer that contains a solved child. Paths are rebuilt from per-layer parent pointers.

Move filter (the dataset's canonical scramble filter, applied to search paths):

* always: a child's move is never on the same face as its parent's last move (``R R'``, ``R R2`` ...);
* if ``no_backtrack`` (default): when the parent's last two moves are on opposite faces (``U D``), the child
  may use neither face of that axis (no ``U D U'``, ``U D D2``).

Both rules only remove paths that have an equally short or shorter equivalent, so the filtered search can still
find a minimal solution; ``no_backtrack=False`` keeps only the first rule (for ablations).

Every solution is verified with ``cube.apply_sequence`` + ``cube.is_solved`` before it is returned; a path that
fails raises ``AssertionError`` (an explicit raise, active under ``python -O``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from . import cube

PolicyFn = Callable[[torch.Tensor], torch.Tensor]

MOVE_FACE = np.arange(cube.N_MOVES, dtype=np.int64) // 3
NON_CENTRE = np.array([i for i in range(cube.N_FACELETS) if i not in set(cube.CENTRES.tolist())], dtype=np.int64)
# 48 non-centre stickers, colours 0..5: two base-6 words of 24 digits each; 6**24 - 1 < 2**63 - 1, so the pair of
# int64 keys identifies a state exactly (the centres never move).
_KEY_DIGITS = 24
_KEY_POW = 6 ** np.arange(_KEY_DIGITS, dtype=np.int64)
if not (len(NON_CENTRE) == 2 * _KEY_DIGITS and int(_KEY_POW[-1]) * 6 - 1 < 2 ** 63 - 1):
    raise AssertionError("state key packing does not fit in int64")


@dataclass
class SearchResult:
    moves: list[int] | None  # the verified solution, or None if none was found within max_depth
    nodes_expanded: int      # states the policy was evaluated on (each expanded into its allowed children)
    depth_reached: int       # number of layers expanded (= len(moves) when solved)
    ms: float                # wall-clock milliseconds for the whole call

    @property
    def solved(self) -> bool:
        return self.moves is not None


class _Tables:
    def __init__(self, device: torch.device):
        self.perms = torch.from_numpy(cube.MOVE_PERMS.astype(np.int64)).to(device)  # (18, 54)
        self.move_face = torch.from_numpy(MOVE_FACE).to(device)                       # (18,)
        self.non_centre = torch.from_numpy(NON_CENTRE).to(device)                     # (48,)
        self.key_pow = torch.from_numpy(_KEY_POW).to(device)                          # (24,)
        self.solved = torch.from_numpy(np.array(cube.SOLVED)).to(device)              # (54,) uint8


_TABLE_CACHE: dict[str, _Tables] = {}


def _tables(device: torch.device) -> _Tables:
    key = str(device)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = _Tables(device)
    return _TABLE_CACHE[key]


def _move_perms(device: torch.device) -> torch.Tensor:
    """The (18, 54) move permutations the search expands with (a hook for the tests)."""
    return _tables(device).perms


def make_policy_fn(model: torch.nn.Module, device: torch.device | str | None = None) -> PolicyFn:
    """Wrap a policy model (with ``logits_from_uint8``) as a policy_fn returning log-softmax (N, 18).

    The model is put in eval mode. The returned function carries ``.device``, which the searches use by default.
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()

    def policy_fn(states: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits = model.logits_from_uint8(states.to(dev))
            return torch.log_softmax(logits.float(), dim=1)

    policy_fn.device = dev  # type: ignore[attr-defined]
    return policy_fn


def _resolve(policy_fn: PolicyFn, device) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device(getattr(policy_fn, "device", "cpu"))


def _check_state(state) -> np.ndarray:
    s = np.asarray(state)
    if s.shape != (cube.N_FACELETS,):
        raise ValueError(f"state must have shape (54,), got {s.shape}")
    if s.size and (s.min() < 0 or s.max() > 5):
        raise ValueError("colours must be in 0..5")
    return s.astype(np.uint8)


def _verify(state: np.ndarray, path: list[int]) -> list[int]:
    """Never trust the search: the path must solve ``state`` under the reference move model."""
    if not cube.is_solved(cube.apply_sequence(state, path)):
        raise AssertionError(f"search produced a non-solving path ({len(path)} moves): {cube.format_moves(path)}")
    return path


def _allowed(last_face: torch.Tensor, lock_axis: torch.Tensor, move_face: torch.Tensor,
             no_backtrack: bool) -> torch.Tensor:
    """(B, 18) bool: which moves each beam state may take (see the module docstring)."""
    ok = move_face[None, :] != last_face[:, None]
    if no_backtrack:
        ok &= (lock_axis[:, None] < 0) | ((move_face[None, :] % 3) != lock_axis[:, None])
    return ok


def _next_lock(parent_face: torch.Tensor, new_face: torch.Tensor) -> torch.Tensor:
    """Axis locked after a move: the move's axis if it follows a move on the opposite face, else -1."""
    opposite = (parent_face >= 0) & ((parent_face % 3) == (new_face % 3))
    return torch.where(opposite, new_face % 3, torch.full_like(new_face, -1))


def state_keys(states: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Exact (M, 2) int64 keys of uint8 (M, 54) states: equal keys <=> equal states."""
    t = _tables(states.device if device is None else device)
    x = states.index_select(1, t.non_centre).to(torch.int64).reshape(states.shape[0], 2, _KEY_DIGITS)
    return (x * t.key_pow).sum(dim=2)


def _dedupe_best(children: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Indices of one child per distinct state, the best-scored one (ties: the earliest index)."""
    keys = state_keys(children)
    order = torch.sort(scores, descending=True, stable=True).indices
    order = order[torch.sort(keys[order, 1], stable=True).indices]
    order = order[torch.sort(keys[order, 0], stable=True).indices]  # lexicographic (k0, k1), score desc
    k = keys[order]
    first = torch.ones(k.shape[0], dtype=torch.bool, device=k.device)
    first[1:] = (k[1:] != k[:-1]).any(dim=1)
    return order[first]


def beam_search(policy_fn: PolicyFn, state, width: int = 4096, max_depth: int = 40, no_backtrack: bool = True,
                device: torch.device | str | None = None) -> SearchResult:
    """Beam search from ``state`` (uint8 (54,)). Returns a verified solution, or ``moves=None`` at max_depth."""
    t0 = time.perf_counter()
    s0 = _check_state(state)
    if width < 1 or max_depth < 0:
        raise ValueError("need width >= 1 and max_depth >= 0")
    if cube.is_solved(s0):
        return SearchResult([], 0, 0, (time.perf_counter() - t0) * 1e3)
    dev = _resolve(policy_fn, device)
    t = _tables(dev)
    perms = _move_perms(dev)
    n_moves = cube.N_MOVES

    with torch.inference_mode():
        beam = torch.from_numpy(s0[None].copy()).to(dev)                 # (B, 54) uint8
        score = torch.zeros(1, dtype=torch.float32, device=dev)          # cumulative log-prob
        last_face = torch.full((1,), -1, dtype=torch.int64, device=dev)
        lock_axis = torch.full((1,), -1, dtype=torch.int64, device=dev)
        hist_parent: list[torch.Tensor] = []                             # per kept layer: index into previous beam
        hist_move: list[torch.Tensor] = []
        nodes = 0
        for depth in range(1, max_depth + 1):
            b = beam.shape[0]
            nodes += b
            logp = policy_fn(beam)
            if tuple(logp.shape) != (b, n_moves):
                raise ValueError(f"policy_fn returned shape {tuple(logp.shape)}, expected ({b}, {n_moves})")
            logp = logp.to(device=dev, dtype=torch.float32)
            allowed = _allowed(last_face, lock_axis, t.move_face, no_backtrack).reshape(-1)
            cand = allowed.nonzero().squeeze(1)                          # parent-major order
            parent = cand // n_moves
            move = cand % n_moves
            children = torch.gather(beam.index_select(0, parent), 1, perms.index_select(0, move))
            cscore = (score[:, None] + logp).reshape(-1).index_select(0, cand)
            if bool(torch.isnan(cscore).any()):
                raise ValueError("policy_fn returned NaN log-probabilities")

            # Solved children are checked BEFORE any truncation, and the choice is made among their explicit indices,
            # so a solved child is returned even if every solved child scores -inf (a saturated policy).
            solved_idx = (children == t.solved).all(dim=1).nonzero().squeeze(1)
            if solved_idx.numel() > 0:
                j = int(solved_idx[torch.argmax(cscore.index_select(0, solved_idx))])
                path = [int(move[j])]
                bi = int(parent[j])
                for k in range(len(hist_parent) - 1, -1, -1):
                    path.append(int(hist_move[k][bi]))
                    bi = int(hist_parent[k][bi])
                path.reverse()
                _verify(s0, path)
                return SearchResult(path, nodes, depth, (time.perf_counter() - t0) * 1e3)
            if depth == max_depth:
                break

            uniq = _dedupe_best(children, cscore)
            top = torch.sort(cscore.index_select(0, uniq), descending=True, stable=True).indices[:width]
            sel = uniq.index_select(0, top)
            beam = children.index_select(0, sel)
            score = cscore.index_select(0, sel)
            sel_parent = parent.index_select(0, sel)
            sel_move = move.index_select(0, sel)
            new_face = t.move_face.index_select(0, sel_move)
            lock_axis = _next_lock(last_face.index_select(0, sel_parent), new_face)
            last_face = new_face
            hist_parent.append(sel_parent)
            hist_move.append(sel_move)

    return SearchResult(None, nodes, max_depth, (time.perf_counter() - t0) * 1e3)


def greedy_solve(policy_fn: PolicyFn, state, max_depth: int = 40, no_backtrack: bool = True,
                 device: torch.device | str | None = None) -> SearchResult:
    """Follow the policy's best allowed move from ``state`` until solved or ``max_depth`` moves (same move filter
    as the beam search). Returns a verified solution or ``moves=None``."""
    t0 = time.perf_counter()
    s0 = _check_state(state)
    if max_depth < 0:
        raise ValueError("need max_depth >= 0")
    if cube.is_solved(s0):
        return SearchResult([], 0, 0, (time.perf_counter() - t0) * 1e3)
    dev = _resolve(policy_fn, device)
    cur = s0.copy()
    path: list[int] = []
    last_face, lock_axis = -1, -1
    with torch.inference_mode():
        for depth in range(1, max_depth + 1):
            logp = policy_fn(torch.from_numpy(cur[None].copy()).to(dev))
            if tuple(logp.shape) != (1, cube.N_MOVES):
                raise ValueError(f"policy_fn returned shape {tuple(logp.shape)}, expected (1, {cube.N_MOVES})")
            lp = logp[0].detach().cpu().numpy().astype(np.float64)  # CPU first: MPS has no float64
            if np.isnan(lp).any():
                raise ValueError("policy_fn returned NaN log-probabilities")
            ok = MOVE_FACE != last_face
            if no_backtrack and lock_axis >= 0:
                ok &= (MOVE_FACE % 3) != lock_axis
            allowed_idx = np.flatnonzero(ok)  # argmax over the allowed moves only: a forbidden move is never chosen,
            m = int(allowed_idx[np.argmax(lp[allowed_idx])])  # even when every allowed move scores -inf
            face = m // 3
            lock_axis = face % 3 if last_face >= 0 and last_face % 3 == face % 3 else -1
            last_face = face
            cur = cur[cube.MOVE_PERMS[m]]
            path.append(m)
            if cube.is_solved(cur):
                _verify(s0, path)
                return SearchResult(path, depth, depth, (time.perf_counter() - t0) * 1e3)
    return SearchResult(None, max_depth, max_depth, (time.perf_counter() - t0) * 1e3)
