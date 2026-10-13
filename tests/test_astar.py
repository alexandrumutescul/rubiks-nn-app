import numpy as np
import pytest
import torch

from rubiks import astar, cube, sampling


@pytest.fixture(scope="module")
def distances():
    """Independent breadth-first distances through depth five in the half-turn metric."""
    dist = {cube.SOLVED.tobytes(): 0}
    frontier = np.array(cube.SOLVED)[None]
    for depth in range(1, 6):
        new = []
        for row in frontier[:, cube.MOVE_PERMS].reshape(-1, 54):
            key = row.tobytes()
            if key not in dist:
                dist[key] = depth
                new.append(key)
        frontier = np.frombuffer(b"".join(new), dtype=np.uint8).reshape(-1, 54)
    assert np.bincount(list(dist.values())).tolist() == [1, 18, 243, 3240, 43239, 574908]
    return dist


def oracle(dist):
    return lambda states: torch.tensor([dist.get(s.tobytes(), 6) for s in states.cpu().numpy()], dtype=torch.float32)


@pytest.mark.parametrize("batch", [1, 7, 100])
def test_exact_bfs_optimal_through_depth_five(distances, batch):
    states, _, _ = sampling.random_scrambles(np.random.default_rng(42), 25, 5, depths=np.repeat([1, 2, 3, 4, 5], 5))
    for state in states:
        result = astar.astar_solve(oracle(distances), state, weight=1, batch=batch, max_nodes=5000)
        assert result.solved and len(result.moves) == distances[state.tobytes()]
        assert cube.is_solved(cube.apply_sequence(state, result.moves))
        faces = [m // 3 for m in result.moves]
        for i in range(1, len(faces)):
            assert faces[i] != faces[i - 1]
            if i >= 2 and faces[i - 2] % 3 == faces[i - 1] % 3:
                assert faces[i] % 3 != faces[i - 1] % 3


def test_filter_matches_canonical_rules():
    faces = np.array([-1, 0, 3, 4, 2])
    locks = np.array([-1, -1, 0, 1, -1])
    allowed = astar._allowed_np(faces, locks)
    for i, (face, lock) in enumerate(zip(faces, locks)):
        assert np.flatnonzero(allowed[i]).tolist() == [m for m in range(18)
                                                      if m // 3 != face and (lock < 0 or m // 3 % 3 != lock)]
    assert astar._next_lock(np.array([-1, 0, 1, 2]), np.array([3, 3, 4, 0])).tolist() == [-1, 0, 1, -1]


@pytest.mark.parametrize("cap", [0, 1, 20])
def test_exhaustion_respects_cap(cap):
    state = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U F L B"))
    result = astar.astar_solve(lambda s: torch.zeros(len(s)), state, batch=7, max_nodes=cap)
    assert result.moves is None and result.nodes_expanded == cap


@pytest.mark.parametrize("solve", [astar.astar_solve, astar.greedy_best_first])
def test_verification_catches_broken_move_table(monkeypatch, solve):
    state = cube.apply_move(cube.SOLVED, 1)
    bad = cube.MOVE_PERMS.copy()
    bad[[0, 1]] = bad[[1, 0]]
    monkeypatch.setattr(astar, "_move_perms", lambda: bad)
    with pytest.raises(AssertionError, match="non-solving path"):
        solve(lambda s: pytest.fail("solved child must be checked before h"), state)


def test_solved_input_and_solved_child_skip_heuristic():
    fn = lambda s: pytest.fail("no heuristic call needed")
    assert astar.astar_solve(fn, cube.SOLVED).moves == []
    result = astar.astar_solve(fn, cube.apply_move(cube.SOLVED, 3))
    assert result.moves == [4] and result.nodes_expanded == 1 and result.value_calls == 0


@pytest.mark.parametrize("fn, message", [(lambda s: torch.zeros(len(s) + 1), "values"),
                                         (lambda s: torch.full((len(s),), float("nan")), "NaN")])
def test_bad_heuristic(fn, message):
    state = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U"))
    with pytest.raises(ValueError, match=message):
        astar.astar_solve(fn, state)


def test_reopens_a_state_and_returns_the_shorter_path(monkeypatch):
    # This identity gives two routes to the same cube: seven moves forward,
    # or six moves backward. The heuristic expands the longer route first.
    identity = cube.parse_moves("R L U2 R L' B2 U2 R2 F2 L2 D2 L2 F2")
    assert len(identity) == 13 and cube.is_solved(cube.apply_sequence(cube.SOLVED, identity))
    long_path, short_path = identity[:7], cube.inverse_sequence(identity[7:])
    tail = cube.parse_moves("B U")
    state = cube.apply_sequence(cube.SOLVED, cube.inverse_sequence(short_path + tail))
    meeting = cube.apply_sequence(state, short_path)
    np.testing.assert_array_equal(cube.apply_sequence(state, long_path), meeting)

    estimates = {}
    for path, value in ((short_path, 20.0), (long_path, 0.0)):
        for end in range(1, len(path) + 1):
            estimates[cube.apply_sequence(state, path[:end]).tobytes()] = value
    # Delay the last-but-one state until the six-move route can re-open both
    # the meeting state and its child with a smaller g.
    estimates[cube.apply_move(meeting, tail[0]).tobytes()] = 100.0
    meeting_costs = []
    add = astar._Nodes.add

    def recording_add(self, states, parent, move, g, face, lock):
        meeting_costs.extend(int(cost) for row, cost in zip(states, g)
                             if np.array_equal(row, meeting))
        return add(self, states, parent, move, g, face, lock)

    monkeypatch.setattr(astar._Nodes, "add", recording_add)

    def heuristic(states):
        return torch.tensor([estimates.get(row.tobytes(), 1000.0) for row in states.cpu().numpy()])

    result = astar.astar_solve(heuristic, state, weight=0.6, batch=1, max_nodes=30)
    assert meeting_costs == [7, 6]
    assert result.solved and len(result.moves) == len(short_path + tail) == 8
    assert cube.is_solved(cube.apply_sequence(state, result.moves))


def test_weight_point_six_scales_g_not_h():
    solution = cube.parse_moves("R U F")
    state = cube.apply_sequence(cube.SOLVED, cube.inverse_sequence(solution))
    estimates = {
        cube.apply_move(state, solution[0]).tobytes(): 0.0,
        cube.apply_sequence(state, solution[:2]).tobytes(): 0.0,
        cube.apply_sequence(state, cube.parse_moves("L")).tobytes(): 1.0,
    }

    def heuristic(states):
        return torch.tensor([estimates.get(row.tobytes(), 100.0) for row in states.cpu().numpy()])

    # After R, (g=2, h=0) must precede L's (g=1, h=1): 1.2 < 1.6.
    # The other convention, g + 0.6*h, reverses them: 2 > 1.6, exhausting
    # the three-expansion cap before it can generate the solved child.
    result = astar.astar_solve(heuristic, state, weight=0.6, batch=1, max_nodes=3)
    assert result.moves == solution and result.nodes_expanded == 3


def test_greedy_bfs(distances):
    state = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U F L B"))
    result = astar.greedy_best_first(oracle(distances), state)
    assert result.solved and len(result.moves) == distances[state.tobytes()]
