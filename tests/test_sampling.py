import numpy as np
import pytest

from rubiks import cube, cubie, sampling

N = 20_000
MAXD = 20


@pytest.fixture(scope="module")
def scrambles():
    return sampling.random_scrambles(np.random.default_rng(42), N, MAXD)


def test_shapes_and_dtypes(scrambles):
    states, moves, depth = scrambles
    assert states.shape == (N, 54) and states.dtype == np.uint8
    assert moves.shape == (N, MAXD) and moves.dtype == np.int8
    assert depth.shape == (N,) and depth.dtype == np.uint8


def test_depth_range_and_padding(scrambles):
    _, moves, depth = scrambles
    assert depth.min() >= 1 and depth.max() <= MAXD
    assert set(np.unique(depth).tolist()) == set(range(1, MAXD + 1))
    within = np.arange(MAXD)[None, :] < depth[:, None]
    assert np.all(moves[~within] == -1)
    assert np.all((moves[within] >= 0) & (moves[within] < 18))


def test_canonical_filter(scrambles):
    _, moves, depth = scrambles
    faces = np.where(moves >= 0, moves // 3, -1).astype(np.int64)
    valid = faces >= 0
    # no equal consecutive faces
    pair = valid[:, 1:] & valid[:, :-1]
    assert not np.any(pair & (faces[:, 1:] == faces[:, :-1]))
    # no A B A, nor A B B', with A, B opposite (once t-2, t-1 are opposite, t is off that axis)
    tri = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    opposite = (faces[:, :-2] % 3) == (faces[:, 1:-1] % 3)
    assert not np.any(tri & opposite & ((faces[:, 2:] % 3) == (faces[:, 1:-1] % 3)))


def test_filter_allows_everything_else_uniformly(scrambles):
    """Every allowed (t-2, t-1, t) face triple occurs, and the choice at step t is uniform over allowed faces."""
    _, moves, _ = scrambles
    faces = np.where(moves >= 0, moves // 3, -1).astype(np.int64)
    a, b, c = faces[:, :-2].ravel(), faces[:, 1:-1].ravel(), faces[:, 2:].ravel()
    ok = (a >= 0) & (b >= 0) & (c >= 0)
    triples = set(zip(a[ok].tolist(), b[ok].tolist(), c[ok].tolist()))
    allowed = {(x, y, z) for x in range(6) for y in range(6) for z in range(6)
               if y != x and (z % 3 != y % 3 if x % 3 == y % 3 else z != y)}
    assert len(allowed) == 144 and triples == allowed
    for x, y in [(0, 3), (0, 1)]:  # an opposite pair (4 choices), a non-opposite pair (5 choices)
        sel = ok & (a == x) & (b == y)
        freq = np.bincount(c[sel], minlength=6) / sel.sum()
        nz = freq[freq > 0]
        expect = 1 / len(nz)
        assert len(nz) == (4 if x % 3 == y % 3 else 5)
        assert np.all(np.abs(nz - expect) < 0.035)
    kinds = moves[moves >= 0] % 3
    assert np.all(np.abs(np.bincount(kinds, minlength=3) / kinds.size - 1 / 3) < 0.01)


def test_kind_independent_of_face(scrambles):
    """Per face, each kind (clockwise, ', 2) is ~1/3: a uniform aggregate alone would not catch kind = f(face)."""
    _, moves, _ = scrambles
    valid = moves[moves >= 0].astype(np.int64)
    faces, kinds = valid // 3, valid % 3
    for f in range(6):
        k = kinds[faces == f]
        assert k.size > 10_000
        freq = np.bincount(k, minlength=3) / k.size
        assert np.all(np.abs(freq - 1 / 3) < 0.02), (f, freq)


def test_states_match_moves_and_inverse_solves(scrambles):
    states, moves, depth = scrambles
    # forward: replay each row's moves on SOLVED (vectorised) and compare
    s = np.tile(cube.SOLVED, (N, 1))
    for t in range(MAXD):
        act = moves[:, t] >= 0
        s[act] = cube.apply_moves(s[act], moves[act, t])
    assert np.array_equal(s, states)
    # inverse: undo from the last move back to the first
    s = states.copy()
    for t in range(MAXD - 1, -1, -1):
        act = moves[:, t] >= 0
        inv = np.array([cube.inverse_move(m) for m in range(18)])[moves[act, t]]
        s[act] = cube.apply_moves(s[act], inv)
    assert cube.is_solved_batch(s).all()
    # spot check against the sequential API
    for i in range(0, N, 997):
        seq = moves[i, :depth[i]].tolist()
        assert np.array_equal(cube.apply_sequence(cube.SOLVED, seq), states[i])
        assert cube.is_solved(cube.apply_sequence(states[i], cube.inverse_sequence(seq)))


def test_seed_reproducibility():
    a = sampling.random_scrambles(np.random.default_rng(5), 3000, MAXD)
    b = sampling.random_scrambles(np.random.default_rng(5), 3000, MAXD)
    c = sampling.random_scrambles(np.random.default_rng(6), 3000, MAXD)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    for x, y in zip(a, c):
        assert not np.array_equal(x, y)


def test_given_depths():
    states, moves, depth = sampling.random_scrambles(np.random.default_rng(1), 500, 12, depths=7)
    assert np.all(depth == 7) and np.all(moves[:, 7:] == -1) and np.all(moves[:, :7] >= 0)
    d = np.arange(500) % 12 + 1
    _, moves, depth = sampling.random_scrambles(np.random.default_rng(1), 500, 12, depths=d)
    assert np.array_equal(depth, d)
    with pytest.raises(ValueError):
        sampling.random_scrambles(np.random.default_rng(1), 10, 12, depths=0)
    with pytest.raises(ValueError):
        sampling.random_scrambles(np.random.default_rng(1), 10, 12, depths=13)
    states, moves, depth = sampling.random_scrambles(np.random.default_rng(1), 0, 12)
    assert states.shape == (0, 54) and moves.shape == (0, 12) and depth.shape == (0,)


def test_uniform_random_states():
    n = 3000
    states = sampling.uniform_random_states(np.random.default_rng(77), n)
    assert states.shape == (n, 54) and states.dtype == np.uint8
    counts = np.stack([np.bincount(s, minlength=6) for s in states])
    assert np.all(counts == 9)
    assert np.all(states[:, cube.CENTRES] == np.arange(6))
    cp = np.empty((n, 8), dtype=np.int64)
    for i, s in enumerate(states):
        c = cubie.from_facelets(s)
        assert c.is_valid()
        cp[i] = c.cp
    freq = np.stack([(cp == j).mean(axis=0) for j in range(8)])  # freq[cubie, position]
    assert freq.min() >= 0.09 and freq.max() <= 0.16, (freq.min(), freq.max())
    assert not cube.is_solved_batch(states).any()
