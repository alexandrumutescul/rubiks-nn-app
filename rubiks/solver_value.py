"""Solver API around a trained cost-to-go checkpoint (the value-method twin of ``rubiks.solver_nn.NNSolver``).

    solver = ValueSolver("models/cost-to-go-v1.pt")        # loads once
    solver.solve(state)  # -> {"moves": [...], "notation": "R U' ...", "source": "value-astar", "nodes": n,
                         #     "nodes_generated": m, "ms": t}   or   {"error": "...", "source": ..., "ms": t, ...}

``solve`` never raises for bad input: the input validation is ``NNSolver``'s own (wrong shape, non-integers,
colours outside 0..5, unreachable cube -> ``{"error": ...}``). The search is ``rubiks.astar.astar_solve`` (batched
weighted A*), which verifies every solution; ``solve`` checks it once more with the cubie model. ``nodes`` is the
number of nodes expanded. Calls are serialised with a lock, so one instance can be shared by the threads of an
HTTP server.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

from . import cube, cubie
from .astar import astar_solve
from .model import VALUE_OUT, load_checkpoint, resolve_device
from .solver_nn import NNSolver
from .value_fn import make_value_fn

_SOLVED_CUBIE = cubie.CubieCube()


class ValueSolver:
    source = "value-astar"

    # The same input validation as the policy solver (a plain function of the input; it uses no instance state).
    _validate = NNSolver._validate

    def __init__(self, checkpoint_path: str | Path, device: str = "auto", weight: float = 0.6, batch: int = 1000,
                 max_nodes: int = 200000):
        if not (math.isfinite(float(weight)) and weight >= 0) or batch < 1 or max_nodes < 1:
            raise ValueError("need a finite weight >= 0, batch >= 1 and max_nodes >= 1")
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        self.weight = float(weight)
        self.batch = int(batch)
        self.max_nodes = int(max_nodes)
        self.model, ckpt = load_checkpoint(self.checkpoint_path, self.device)
        if self.model.out_features != VALUE_OUT:
            raise ValueError(f"{self.checkpoint_path}: not a value checkpoint "
                             f"({self.model.out_features} outputs, expected {VALUE_OUT})")
        self.config = ckpt["config"]
        self.iteration = ckpt.get("iteration")
        self._value = make_value_fn(self.model, self.device)
        self._lock = threading.Lock()

    def solve(self, state) -> dict:
        t0 = time.perf_counter()
        s, err = self._validate(state)
        if err is not None:
            return {"error": err, "source": self.source, "ms": (time.perf_counter() - t0) * 1e3}
        try:
            with self._lock:
                res = astar_solve(self._value, s, weight=self.weight, batch=self.batch, max_nodes=self.max_nodes,
                                  device=self.device)
        except AssertionError as exc:  # the search's own verification failed: a solver bug, reported, not hidden
            return {"error": f"internal: {exc}", "source": self.source, "ms": (time.perf_counter() - t0) * 1e3}
        ms = (time.perf_counter() - t0) * 1e3
        if res.moves is None:
            return {"error": f"no solution found within {self.max_nodes} node expansions (weight {self.weight:g}, "
                             f"batch {self.batch})",
                    "source": self.source, "nodes": res.nodes_expanded, "nodes_generated": res.nodes_generated,
                    "ms": ms}
        if cubie.from_facelets(s).apply_sequence(res.moves) != _SOLVED_CUBIE:  # second, independent model
            return {"error": "internal: solution failed the cubie-model check", "source": self.source,
                    "nodes": res.nodes_expanded, "nodes_generated": res.nodes_generated, "ms": ms}
        return {"moves": [int(m) for m in res.moves], "notation": cube.format_moves(res.moves),
                "source": self.source, "nodes": res.nodes_expanded, "nodes_generated": res.nodes_generated, "ms": ms}
