import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
import torch

from rubiks import cube, model, serve, solver_hybrid


@pytest.fixture
def hybrid(tmp_path):
    torch.manual_seed(2)
    value, policy = tmp_path / "value.pt", tmp_path / "policy.pt"
    model.save_checkpoint(value, model.ResMLP(stem=(16, 8), blocks=1), {})
    model.save_checkpoint(policy, model.MLP(hidden=(16, 8)), {})
    return solver_hybrid.HybridSolver(value, policy, device="cpu", max_nodes=1, beam=18, max_depth=2)


def test_exhausted_value_falls_back_and_reports_both_stages(hybrid):
    seen = []

    def hopeless(states):
        seen.append(len(states))
        return torch.ones(len(states)) * 100

    hybrid._value = hopeless
    state = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U"))
    original = state.copy()
    result = hybrid.solve(state)
    assert seen == [18]
    assert result["source"] == "hybrid: policy-beam (fallback)"
    assert len(result["moves"]) == 2 and cube.is_solved(cube.apply_sequence(state, result["moves"]))
    assert result["nodes"] == result["astar_expanded"] + result["beam_expanded"] == 20
    assert 0 < result["astar_ms"] and 0 < result["beam_ms"]
    assert result["ms"] >= result["astar_ms"] + result["beam_ms"]
    np.testing.assert_array_equal(state, original)


def test_value_success_never_calls_beam_and_labels_are_per_call(hybrid):
    state = cube.apply_move(cube.SOLVED, 3)
    hybrid._policy = lambda s: pytest.fail("A* succeeded")
    for _ in range(2):
        result = hybrid.solve(state)
        assert result["source"] == "hybrid: value-astar" and result["moves"] == [4]
        assert result["beam_ms"] == result["beam_expanded"] == 0
    assert hybrid.solve(cube.SOLVED)["moves"] == []


def test_one_lock_covers_both_searches(hybrid, monkeypatch):
    states = [cube.apply_sequence(cube.SOLVED, cube.parse_moves(s)) for s in ("R U", "F L")]
    calls = {state.tobytes(): i for i, state in enumerate(states)}
    events = []
    ready = Barrier(2)

    def checked(stage, fn):
        def run(network, state, **kwargs):
            call = calls[state.tobytes()]
            events.append((call, stage, "enter"))
            assert hybrid._lock.locked()
            # Yield to the competing request while this stage holds the lock.
            time.sleep(0.01)
            try:
                return fn(network, state, **kwargs)
            finally:
                assert hybrid._lock.locked()
                events.append((call, stage, "exit"))
        return run

    monkeypatch.setattr(solver_hybrid, "astar_solve", checked("astar", solver_hybrid.astar_solve))
    monkeypatch.setattr(solver_hybrid, "beam_search", checked("beam", solver_hybrid.beam_search))

    def solve(state):
        ready.wait(timeout=5)
        return hybrid.solve(state)

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(solve, states))
    assert all("moves" in answer for answer in answers)
    first = events[0][0]
    assert events == [(call, stage, event) for call in (first, 1 - first)
                      for stage in ("astar", "beam") for event in ("enter", "exit")]


@pytest.mark.parametrize("failure", ["assertion", "nan"])
def test_astar_exception_is_internal_500_and_never_calls_beam(hybrid, monkeypatch, failure):
    state = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U"))
    if failure == "assertion":
        def broken_search(*args, **kwargs):
            raise AssertionError("non-solving path")
        monkeypatch.setattr(solver_hybrid, "astar_solve", broken_search)
        message = "non-solving path"
    else:
        hybrid._value = lambda states: torch.full((len(states),), float("nan"))
        message = "value_fn returned NaN"
    hybrid._policy = lambda states: pytest.fail("policy must not run after an A* exception")
    monkeypatch.setattr(solver_hybrid, "beam_search",
                        lambda *args, **kwargs: pytest.fail("beam must not run after an A* exception"))

    result = hybrid.solve(state)
    assert result["error"] == f"internal: {message}"
    assert result["source"] == "hybrid: value-astar"
    assert result["beam_expanded"] == result["beam_ms"] == 0
    status, result = serve.handle_solve(json.dumps({"state": state.tolist()}).encode(),
                                       serve.Backend("hybrid", solver=hybrid))
    assert status == 500 and f"internal: {message}" in result["error"]


def test_invalid_and_exhausted_inputs(hybrid):
    assert "error" in hybrid.solve([0] * 53)
    hybrid.max_depth = 1
    state = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U"))
    result = hybrid.solve(state)
    assert result["error"].startswith("no solution found within")
    assert result["source"] == "hybrid: policy-beam (fallback)"


def test_cubie_check_rejects_bad_result(hybrid, monkeypatch):
    from rubiks.astar import SearchResult

    monkeypatch.setattr(solver_hybrid, "astar_solve", lambda *a, **kw: SearchResult([0], 1, 18, 1.0))
    result = hybrid.solve(cube.apply_move(cube.SOLVED, 3))
    assert result["error"] == "internal: solution failed the cubie-model check"
    assert result["beam_ms"] == 0


def test_loads_each_checkpoint_once(hybrid, monkeypatch):
    monkeypatch.setattr(solver_hybrid, "load_checkpoint", lambda *a, **kw: pytest.fail("reloaded a checkpoint"))
    import rubiks.solver_value as value_module
    monkeypatch.setattr(value_module, "load_checkpoint", lambda *a, **kw: pytest.fail("reloaded a checkpoint"))
    for move in (3, 6):
        assert hybrid.solve(cube.apply_move(cube.SOLVED, move))["moves"] == [cube.inverse_move(move)]
