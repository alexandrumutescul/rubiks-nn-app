"""Cost-to-go A* followed, on exhaustion, by policy beam search."""

from __future__ import annotations

import time
from pathlib import Path

from . import cube, cubie
from .astar import astar_solve
from .model import load_checkpoint
from .search import beam_search, make_policy_fn
from .solver_value import ValueSolver


class HybridSolver(ValueSolver):
    """Load both models once and serialise both searches with one lock.

    ``nodes`` totals ``astar_expanded`` and ``beam_expanded``. ``astar_ms`` and
    ``beam_ms`` measure each search separately; beam fields are zero when A* succeeds.
    """

    source = "hybrid: value-astar"

    def __init__(self, value_checkpoint: str | Path, policy_checkpoint: str | Path,
                 device: str = "auto", weight: float = 0.6, batch: int = 1000,
                 max_nodes: int = 200000, beam: int = 1024, max_depth: int = 40):
        if beam < 1 or max_depth < 1:
            raise ValueError("need beam >= 1 and max_depth >= 1")
        super().__init__(value_checkpoint, device, weight, batch, max_nodes)
        self.value_checkpoint = self.checkpoint_path
        self.policy_checkpoint = Path(policy_checkpoint)
        self.beam = int(beam)
        self.max_depth = int(max_depth)
        self.policy_model, _ = load_checkpoint(self.policy_checkpoint, self.device)
        if self.policy_model.out_features != cube.N_MOVES:
            raise ValueError(f"{self.policy_checkpoint}: not a policy checkpoint "
                             f"({self.policy_model.out_features} outputs, expected {cube.N_MOVES})")
        self._policy = make_policy_fn(self.policy_model, self.device)

    def solve(self, state) -> dict:
        t0 = time.perf_counter()
        s, err = self._validate(state)
        details = {"source": self.source, "nodes": 0, "astar_expanded": 0, "beam_expanded": 0,
                   "astar_ms": 0.0, "beam_ms": 0.0}

        def answer(**fields):
            return {**details, **fields, "ms": (time.perf_counter() - t0) * 1e3}

        if err is not None:
            return answer(error=err)
        try:
            with self._lock:
                res = astar_solve(self._value, s, weight=self.weight, batch=self.batch,
                                  max_nodes=self.max_nodes, device=self.device)
                details.update(astar_ms=res.ms, astar_expanded=res.nodes_expanded, nodes=res.nodes_expanded)
                if res.moves is None:
                    details["source"] = "hybrid: policy-beam (fallback)"
                    res = beam_search(self._policy, s, width=self.beam, max_depth=self.max_depth,
                                      device=self.device)
                    details.update(beam_ms=res.ms, beam_expanded=res.nodes_expanded,
                                   nodes=details["nodes"] + res.nodes_expanded)
        except (AssertionError, ValueError) as exc:
            return answer(error=f"internal: {exc}")
        if res.moves is None:
            return answer(error=f"no solution found within {self.max_nodes} A* node expansions then "
                                f"{self.max_depth} moves at beam width {self.beam}")
        if cubie.from_facelets(s).apply_sequence(res.moves) != cubie.CubieCube():
            return answer(error="internal: solution failed the cubie-model check")
        return answer(moves=[int(m) for m in res.moves], notation=cube.format_moves(res.moves))
