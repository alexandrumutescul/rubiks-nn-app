"""Solver API around a trained policy checkpoint (for rubiks.serve and notebooks).

    solver = NNSolver("runs/policy-v1/best.pt")          # loads once
    solver.solve(state)  # -> {"moves": [...], "notation": "R U' ...", "source": "policy-beam", "nodes": n, "ms": t}
                         #    or {"error": "...", ...}

``solve`` never raises for bad input: an invalid state (wrong shape, colours outside 0..5, unreachable cube)
returns ``{"error": ...}``. Every returned solution has been verified by the search (``rubiks.search``) and is
checked once more here with the cubie model. Calls are serialised with a lock, so one instance can be shared by
the threads of an HTTP server.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from . import cube, cubie
from .model import load_checkpoint, resolve_device
from .search import beam_search, make_policy_fn

_SOLVED_CUBIE = cubie.CubieCube()


class NNSolver:
    source = "policy-beam"

    def __init__(self, checkpoint_path: str | Path, device: str = "auto", width: int = 4096, max_depth: int = 40):
        if width < 1 or max_depth < 1:
            raise ValueError("need width >= 1 and max_depth >= 1")
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        self.width = int(width)
        self.max_depth = int(max_depth)
        self.model, ckpt = load_checkpoint(self.checkpoint_path, self.device)
        if self.model.out_features != cube.N_MOVES:
            raise ValueError(f"{self.checkpoint_path}: not a policy checkpoint "
                             f"({self.model.out_features} outputs, expected {cube.N_MOVES})")
        self.config = ckpt["config"]
        self.epoch = ckpt.get("epoch")
        self._policy = make_policy_fn(self.model, self.device)
        self._lock = threading.Lock()

    def _validate(self, state) -> tuple[np.ndarray | None, str | None]:
        try:
            s = np.asarray(state)
        except Exception:  # noqa: BLE001 - any unconvertible input is a client error
            return None, "state is not an array"
        if s.shape != (cube.N_FACELETS,):
            return None, f"state must have 54 entries, got shape {list(s.shape)}"
        if not np.issubdtype(s.dtype, np.integer):
            if not np.issubdtype(s.dtype, np.floating) or not np.all(np.isfinite(s)) or np.any(s != np.round(s)):
                return None, "state entries must be integers"
        if s.min() < 0 or s.max() > 5:
            return None, "state colours must be in 0..5"
        s = s.astype(np.uint8)
        try:
            valid = cubie.from_facelets(s).is_valid()
        except ValueError as exc:
            return None, f"not a valid cube: {exc}"
        if not valid:
            return None, "not a reachable cube state (twisted corner, flipped edge or parity)"
        return s, None

    def solve(self, state) -> dict:
        t0 = time.perf_counter()
        s, err = self._validate(state)
        if err is not None:
            return {"error": err, "source": self.source, "ms": (time.perf_counter() - t0) * 1e3}
        try:
            with self._lock:
                res = beam_search(self._policy, s, width=self.width, max_depth=self.max_depth, device=self.device)
        except AssertionError as exc:  # the search's own verification failed: a solver bug, reported, not hidden
            return {"error": f"internal: {exc}", "source": self.source, "ms": (time.perf_counter() - t0) * 1e3}
        ms = (time.perf_counter() - t0) * 1e3
        if res.moves is None:
            return {"error": f"no solution found within {self.max_depth} moves at beam width {self.width}",
                    "source": self.source, "nodes": res.nodes_expanded, "ms": ms}
        if cubie.from_facelets(s).apply_sequence(res.moves) != _SOLVED_CUBIE:  # second, independent model
            return {"error": "internal: solution failed the cubie-model check", "source": self.source,
                    "nodes": res.nodes_expanded, "ms": ms}
        return {"moves": [int(m) for m in res.moves], "notation": cube.format_moves(res.moves),
                "source": self.source, "nodes": res.nodes_expanded, "ms": ms}
