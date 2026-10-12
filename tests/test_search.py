import math

import numpy as np
import pytest
import torch

from rubiks import cube, model, sampling, search


def _bfs_distances(max_dist: int) -> dict[bytes, int]:
    """Exact distance to solved for every state within ``max_dist`` moves (half-turn metric)."""
    dist = {np.array(cube.SOLVED).tobytes(): 0}
    frontier = np.array(cube.SOLVED)[None]
    for d in range(1, max_dist + 1):
        children = np.unique(frontier[:, cube.MOVE_PERMS].reshape(-1, 54), axis=0)
        new = []
        for row in children:
            k = row.tobytes()
            if k not in dist:
                dist[k] = d
                new.append(row)
        frontier = np.array(new)
    return dist


@pytest.fixture(scope="module")
def dist4():
    d = _bfs_distances(4)
    counts = np.bincount(list(d.values()))
    assert counts.tolist() == [1, 18, 243, 3240, 43239]  # known half-turn-metric layer sizes
    return d


def uniform_policy(states: torch.Tensor) -> torch.Tensor:
    return torch.full((states.shape[0], 18), -math.log(18.0))


def oracle_policy(dist: dict[bytes, int]):
    """Log-probabilities that prefer moves to a state closer to solved (unknown states count as distance 5)."""
    def fn(states: torch.Tensor) -> torch.Tensor:
        s = states.cpu().numpy()
        children = s[:, cube.MOVE_PERMS]  # (N, 18, 54)
        d = np.array([[dist.get(children[i, m].tobytes(), 5) for m in range(18)] for i in range(len(s))],
                     dtype=np.float32)
        return torch.log_softmax(torch.from_numpy(-4.0 * d), dim=1)
    return fn


def _check_canonical(path):
    faces = [m // 3 for m in path]
    for t in range(1, len(faces)):
        assert faces[t] != faces[t - 1]
        if t >= 2 and faces[t - 2] % 3 == faces[t - 1] % 3:
            assert faces[t] % 3 != faces[t - 1] % 3


@pytest.fixture(scope="module")
def shallow():
    rng = np.random.default_rng(5)
    states, moves, depth = sampling.random_scrambles(rng, 60, 3, depths=np.repeat([1, 2, 3], 20))
    return states, depth


def test_uniform_policy_brute_force_solves_depth_1_to_3(shallow, dist4):
    states, depth = shallow
    for s, d in zip(states, depth):
        r = search.beam_search(uniform_policy, s, width=18 ** 3, max_depth=3, device="cpu")
        assert r.solved, f"depth {d} unsolved"
        assert cube.is_solved(cube.apply_sequence(s, r.moves))   # independent re-check
        assert len(r.moves) == dist4[s.tobytes()] == r.depth_reached  # exhaustive -> minimal
        assert len(r.moves) <= d
        _check_canonical(r.moves)


def test_no_solution_within_max_depth(dist4):
    states, _, _ = sampling.random_scrambles(np.random.default_rng(9), 40, 6, depths=6)
    far = [s for s in states if s.tobytes() not in dist4]  # true distance >= 5
    assert len(far) >= 5
    for s in far[:5]:
        r = search.beam_search(uniform_policy, s, width=18 ** 3, max_depth=2, device="cpu")
        assert r.moves is None and not r.solved
        assert r.depth_reached == 2 and r.nodes_expanded == 1 + 18


def test_solved_input():
    for fn in (search.beam_search, search.greedy_solve):
        r = fn(uniform_policy, np.array(cube.SOLVED), max_depth=5, device="cpu")
        assert r.moves == [] and r.nodes_expanded == 0 and r.depth_reached == 0


def test_greedy_with_bfs_oracle_solves_depth_up_to_4(dist4):
    rng = np.random.default_rng(13)
    states, _, depth = sampling.random_scrambles(rng, 120, 4, depths=np.repeat([1, 2, 3, 4], 30))
    fn = oracle_policy(dist4)
    for s in states:
        true_d = dist4[s.tobytes()]
        r = search.greedy_solve(fn, s, max_depth=10, device="cpu")
        assert r.solved and len(r.moves) == true_d
        assert cube.is_solved(cube.apply_sequence(s, r.moves))
        rb = search.beam_search(fn, s, width=1, max_depth=10, device="cpu")
        assert rb.solved and len(rb.moves) == true_d


def test_search_never_returns_an_unverified_path(monkeypatch):
    s = cube.apply_move(cube.SOLVED, cube.parse_moves("U'")[0])
    good = search._move_perms(torch.device("cpu"))
    bad = good.clone()
    bad[[0, 1]] = good[[1, 0]]  # the search's expansion table now swaps U and U'
    monkeypatch.setattr(search, "_move_perms", lambda device: bad)
    with pytest.raises(AssertionError, match="non-solving path"):
        search.beam_search(uniform_policy, s, width=64, max_depth=3, device="cpu")


def test_verify_rejects_wrong_path():
    s = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U"))
    assert search._verify(s, cube.parse_moves("U' R'")) == cube.parse_moves("U' R'")
    with pytest.raises(AssertionError):
        search._verify(s, cube.parse_moves("R' U'"))


def test_state_keys_are_exact(dist4):
    states = np.frombuffer(b"".join(dist4.keys()), dtype=np.uint8).reshape(-1, 54)
    keys = search.state_keys(torch.from_numpy(states.copy())).numpy()
    assert len(np.unique(keys, axis=0)) == len(states)


def test_dedupe_keeps_best_score():
    a = np.array(cube.SOLVED)
    b = cube.apply_move(a, 3)
    children = torch.from_numpy(np.stack([a, b, a, b, a]))
    scores = torch.tensor([-3.0, -1.0, -0.5, -2.0, -0.5])
    kept = sorted(search._dedupe_best(children, scores).tolist())
    assert kept == [1, 2]  # best a is row 2 (tie with row 4 -> earliest), best b is row 1


def test_filter_and_ablation_with_untrained_model():
    torch.manual_seed(0)
    fn = search.make_policy_fn(model.MLP(hidden=(32, 16)), "cpu")
    states, _, _ = sampling.random_scrambles(np.random.default_rng(21), 10, 5)
    for s in states:
        for nb in (True, False):
            r = search.beam_search(fn, s, width=512, max_depth=5, no_backtrack=nb)
            if r.solved:
                assert cube.is_solved(cube.apply_sequence(s, r.moves))
                if nb:
                    _check_canonical(r.moves)
                assert all(r.moves[t] // 3 != r.moves[t - 1] // 3 for t in range(1, len(r.moves)))


def test_bad_inputs():
    with pytest.raises(ValueError):
        search.beam_search(uniform_policy, np.zeros(53, dtype=np.uint8), device="cpu")
    with pytest.raises(ValueError):
        search.beam_search(uniform_policy, np.full(54, 7, dtype=np.uint8), device="cpu")
    bad_shape = lambda s: torch.zeros(s.shape[0], 17)  # noqa: E731
    with pytest.raises(ValueError):
        search.beam_search(bad_shape, cube.apply_move(cube.SOLVED, 0), device="cpu")


# --- policies with saturated / -inf outputs, and beam-ordering properties (review fix round 1) -------------------

def _table_policy(table: dict[bytes, dict[int, float]], default: float):
    """Log-probabilities from a lookup: ``table[state][move]``; unlisted states/moves get ``default``."""
    def fn(states: torch.Tensor) -> torch.Tensor:
        s = states.cpu().numpy()
        out = np.full((len(s), 18), default, dtype=np.float32)
        for i, row in enumerate(s):
            for m, v in table.get(row.tobytes(), {}).items():
                out[i, m] = v
        return torch.from_numpy(out)
    return fn


def anti_oracle_policy(states: torch.Tensor) -> torch.Tensor:
    """-inf exactly on every move that solves the state, 0 elsewhere (a policy saturated the wrong way)."""
    s = states.cpu().numpy()
    solves = np.all(s[:, cube.MOVE_PERMS] == np.array(cube.SOLVED), axis=2)  # (N, 18)
    return torch.from_numpy(np.where(solves, -np.inf, 0.0).astype(np.float32))


def test_solved_child_with_minus_inf_score_is_still_returned():
    for scramble in ("R", "F2", "B'", "R U", "U D'"):
        s = cube.apply_sequence(cube.SOLVED, cube.parse_moves(scramble))
        r = search.beam_search(anti_oracle_policy, s, width=18 ** 2, max_depth=3, device="cpu")
        assert r.solved and cube.is_solved(cube.apply_sequence(s, r.moves)), scramble
        assert len(r.moves) == len(scramble.split())


def test_fully_saturated_policy_minus_inf_everywhere():
    all_inf = lambda st: torch.full((st.shape[0], 18), -float("inf"))  # noqa: E731
    for scramble in ("L'", "D2 F"):
        s = cube.apply_sequence(cube.SOLVED, cube.parse_moves(scramble))
        r = search.beam_search(all_inf, s, width=18 ** 2, max_depth=3, device="cpu")
        assert r.solved and cube.is_solved(cube.apply_sequence(s, r.moves))


def test_minus_1e9_logits_on_the_right_move():
    def fn(states):
        logits = torch.zeros(states.shape[0], 18)
        logits[torch.from_numpy(np.isinf(anti_oracle_policy(states).numpy()))] = -1e9
        return torch.log_softmax(logits, dim=1)
    s = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U"))
    r = search.beam_search(fn, s, width=64, max_depth=4, device="cpu")
    assert r.solved and cube.is_solved(cube.apply_sequence(s, r.moves))


def test_greedy_never_picks_a_forbidden_move_when_all_scores_are_minus_inf():
    seen = []

    def recording(states):
        seen.append(states.cpu().numpy()[0].copy())
        return torch.full((states.shape[0], 18), -float("inf"))

    s = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R F"))
    r = search.greedy_solve(recording, s, max_depth=8, device="cpu")
    path = []
    for a, b in zip(seen, seen[1:]):
        hits = [m for m in range(18) if np.array_equal(cube.apply_move(a, m), b)]
        assert len(hits) == 1
        path.append(hits[0])
    assert len(path) == len(seen) - 1 >= 6
    _check_canonical(path)
    if r.solved:
        _check_canonical(r.moves)


def test_width_1_finds_a_solved_child_with_the_lowest_score():
    a_, b_ = cube.parse_moves("R U")
    s = cube.apply_sequence(cube.SOLVED, [a_, b_])
    mid = cube.apply_move(s, cube.inverse_move(b_))
    table = {s.tobytes(): {cube.inverse_move(b_): 0.0},                       # root: U' is the clear favourite
             mid.tobytes(): {m: -1.0 for m in range(18)} | {cube.inverse_move(a_): -50.0}}  # solver ranked last
    r = search.beam_search(_table_policy(table, -10.0), s, width=1, max_depth=4, device="cpu")
    assert r.moves == [cube.inverse_move(b_), cube.inverse_move(a_)]
    assert r.nodes_expanded == 2  # root, then the single beam state


def test_beam_ranks_by_cumulative_not_local_score():
    """Layer 1 keeps P2 (score -0.1) and P1 (-3). P1's children score -1 locally (cumulative -4); P2's child on the
    solution path scores -2 locally (cumulative -2.1). Ranking by local score would keep two P1 children (width 2)
    and lose the solution; the beam must keep P2's child by its better cumulative score."""
    s = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U F"))
    f_i, u_i, r_i = cube.parse_moves("F' U' R'")
    x = cube.parse_moves("L")[0]
    p2 = cube.apply_move(s, f_i)
    p1 = cube.apply_move(s, x)
    table = {s.tobytes(): {m: -20.0 for m in range(18)} | {f_i: -0.1, x: -3.0},
             p2.tobytes(): {m: -30.0 for m in range(18)} | {u_i: -2.0},
             p1.tobytes(): {m: -1.0 for m in range(18)}}
    r = search.beam_search(_table_policy(table, -25.0), s, width=2, max_depth=3, device="cpu")
    assert r.moves == [f_i, u_i, r_i]
