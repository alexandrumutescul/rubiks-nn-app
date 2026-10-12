import numpy as np
import pytest

from rubiks import cube
from rubiks.cube import MOVE_PERMS, SOLVED

IDENT = np.arange(54)


def compose(*perms):
    """Permutation of applying perms[0], then perms[1], ... (state[p0][p1] == state[p0[p1]])."""
    p = IDENT.copy()
    for q in perms:
        p = p[q]
    return p


def power(p, k):
    return compose(*([p] * k)) if k else IDENT.copy()


def seq_perm(text):
    return cube.sequence_perm(cube.parse_moves(text))


def test_solved_and_constants():
    assert SOLVED.dtype == np.uint8 and SOLVED.shape == (54,)
    assert np.array_equal(SOLVED, np.arange(54) // 9)
    assert MOVE_PERMS.shape == (18, 54) and MOVE_PERMS.dtype == np.int64
    assert cube.MOVE_NAMES == ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2",
                               "D", "D'", "D2", "L", "L'", "L2", "B", "B'", "B2"]
    for p in MOVE_PERMS:
        assert sorted(p.tolist()) == list(range(54))


@pytest.mark.parametrize("face", range(6))
def test_quarter_turn_order_and_powers(face):
    p = MOVE_PERMS[3 * face]
    for k in (1, 2, 3):
        assert not np.array_equal(power(p, k), IDENT)
    assert np.array_equal(power(p, 4), IDENT)
    assert np.array_equal(MOVE_PERMS[3 * face + 1], power(p, 3))
    assert np.array_equal(MOVE_PERMS[3 * face + 2], power(p, 2))


@pytest.mark.parametrize("face", range(6))
def test_quarter_turn_moves_20_facelets(face):
    assert int(np.count_nonzero(MOVE_PERMS[3 * face] != IDENT)) == 20


def test_centres_fixed_under_all_moves():
    for m in range(18):
        assert np.array_equal(MOVE_PERMS[m][cube.CENTRES], cube.CENTRES)


@pytest.mark.parametrize("face", range(6))
def test_face_facelets_rotate_clockwise(face):
    p = MOVE_PERMS[3 * face]
    for r in range(3):
        for c in range(3):
            assert p[face * 9 + c * 3 + (2 - r)] == face * 9 + r * 3 + c


def test_face_views_match_convention():
    """Each border statement of the convention, checked on the 3D sticker geometry."""
    pos = cube.STICKER_POS

    def borders(face, idx, other):
        # every facelet at idx shares its cubie with a facelet of `other`
        other_pos = {tuple(pos[other * 9 + k]) for k in range(9)}
        return all(tuple(pos[face * 9 + i]) in other_pos for i in idx)

    row = {0: [0, 1, 2], 2: [6, 7, 8]}
    col = {0: [0, 3, 6], 2: [2, 5, 8]}
    U, R, F, D, L, B = range(6)
    statements = [
        (U, row[0], B), (U, row[2], F), (U, col[0], L), (U, col[2], R),
        (D, row[0], F), (D, row[2], B), (D, col[0], L), (D, col[2], R),
        (F, col[0], L), (F, col[2], R), (R, col[0], F), (R, col[2], B),
        (B, col[0], R), (B, col[2], L), (L, col[0], B), (L, col[2], F),
    ]
    for f in (F, R, B, L):
        statements += [(f, row[0], U), (f, row[2], D)]
    for face, idx, other in statements:
        assert borders(face, idx, other), (cube.FACE_NAMES[face], idx, cube.FACE_NAMES[other])


def test_single_move_effect_R():
    # R clockwise: F's right column goes up to U, U's right column goes to B's left column (upside down).
    s = cube.apply_move(SOLVED, cube.parse_moves("R")[0])
    assert cube.to_facelet_string(s) == (
        "UUFUUFUUF" "RRRRRRRRR" "FFDFFDFFD" "DDBDDBDDB" "LLLLLLLLL" "UBBUBBUBB")


@pytest.mark.parametrize("a,b", [("U", "D"), ("R", "L"), ("F", "B")])
def test_opposite_faces_commute(a, b):
    assert np.array_equal(seq_perm(f"{a} {b}"), seq_perm(f"{b} {a}"))
    assert not np.array_equal(seq_perm("U R"), seq_perm("R U"))


@pytest.mark.parametrize("text,order", [("R U", 105), ("R U R' U'", 6), ("R2 U2", 6)])
def test_sequence_orders(text, order):
    p = seq_perm(text)
    assert cube.perm_order(p) == order
    s = SOLVED.copy()
    seq = cube.parse_moves(text)
    for k in range(1, order + 1):
        s = cube.apply_sequence(s, seq)
        assert cube.is_solved(s) == (k == order)


def test_known_identity_sequence():
    seq = cube.parse_moves("R L U2 R L' B2 U2 R2 F2 L2 D2 L2 F2")
    assert cube.is_solved(cube.apply_sequence(SOLVED, seq))


def test_inverse_sequence_undoes_random_sequences():
    rng = np.random.default_rng(123)
    for _ in range(100):
        seq = rng.integers(0, 18, size=25).tolist()
        s = cube.apply_sequence(SOLVED, seq)
        assert cube.is_solved(cube.apply_sequence(s, cube.inverse_sequence(seq)))
    for m in range(18):
        assert cube.inverse_move(cube.inverse_move(m)) == m
        assert np.array_equal(compose(MOVE_PERMS[m], MOVE_PERMS[cube.inverse_move(m)]), IDENT)


def test_apply_moves_batch_matches_single():
    rng = np.random.default_rng(7)
    states = np.stack([cube.apply_sequence(SOLVED, rng.integers(0, 18, 10).tolist()) for _ in range(64)])
    moves = rng.integers(0, 18, size=64)
    batch = cube.apply_moves(states, moves)
    for i in range(64):
        assert np.array_equal(batch[i], cube.apply_move(states[i], moves[i]))
    with pytest.raises(ValueError):
        cube.apply_moves(states, np.full(64, 18))
    with pytest.raises(ValueError):
        cube.apply_moves(states, moves[:10])


def test_is_solved_batch():
    states = np.stack([SOLVED, cube.apply_move(SOLVED, 0), SOLVED])
    assert cube.is_solved_batch(states).tolist() == [True, False, True]
    assert cube.is_solved(SOLVED) and not cube.is_solved(states[1])


def test_parse_format_round_trip():
    text = "U U' U2 R R' R2 F F' F2 D D' D2 L L' L2 B B' B2"
    assert cube.parse_moves(text) == list(range(18))
    assert cube.format_moves(range(18)) == text
    rng = np.random.default_rng(5)
    for _ in range(50):
        seq = rng.integers(0, 18, size=rng.integers(0, 30)).tolist()
        assert cube.parse_moves(cube.format_moves(seq)) == seq
    assert cube.parse_moves("  R   U'\tF2 ") == [3, 1, 8]
    assert cube.parse_moves("") == []
    for bad in ("X", "R3", "r", "U''"):
        with pytest.raises(ValueError):
            cube.parse_moves(bad)


def test_facelet_string_round_trip():
    assert cube.to_facelet_string(SOLVED) == "".join(ch * 9 for ch in "URFDLB")
    rng = np.random.default_rng(9)
    for _ in range(50):
        s = cube.apply_sequence(SOLVED, rng.integers(0, 18, 25).tolist())
        t = cube.to_facelet_string(s)
        assert len(t) == 54 and set(t) <= set("URFDLB")
        assert np.array_equal(cube.from_facelet_string(t), s)
        assert cube.from_facelet_string(t).dtype == np.uint8
    with pytest.raises(ValueError):
        cube.from_facelet_string("U" * 53)
    with pytest.raises(ValueError):
        cube.from_facelet_string("X" * 54)


def test_print_net():
    net = cube.print_net(SOLVED).splitlines()
    assert len(net) == 9
    assert net[0].strip() == "U U U" and net[8].strip() == "D D D"
    assert net[4] == "L L L  F F F  R R R  B B B"
