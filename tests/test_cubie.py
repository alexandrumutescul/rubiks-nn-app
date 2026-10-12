import numpy as np
import pytest

from rubiks import cube, cubie
from rubiks.cube import MOVE_PERMS, SOLVED
from rubiks.cubie import MOVE_CUBES, SOLVED_CUBE, CubieCube


def test_solved_cube_facelets():
    assert np.array_equal(SOLVED_CUBE.to_facelets(), SOLVED)
    assert cubie.from_facelets(SOLVED) == SOLVED_CUBE
    assert cubie.MOVE_NAMES == cube.MOVE_NAMES


def test_facelet_tables_match_geometry():
    """The transcribed corner/edge facelet tables name facelets of one cubie each (checked on cube.py's 3D model)."""
    pos = cube.STICKER_POS
    seen = set()
    for table, n_nonzero in ((cubie.CORNER_FACELET, 3), (cubie.EDGE_FACELET, 2)):
        for facelets in table:
            p = {tuple(pos[i]) for i in facelets}
            assert len(p) == 1
            assert np.count_nonzero(np.array(next(iter(p)))) == n_nonzero
            assert facelets[0] // 9 in (cube.U, cube.D) or (n_nonzero == 2 and facelets[0] // 9 in (cube.F, cube.B))
            seen.update(int(i) for i in facelets)
    assert seen == set(range(54)) - set(cube.CENTRES.tolist())


@pytest.mark.parametrize("m", range(18))
def test_models_agree_on_every_move(m):
    assert np.array_equal(MOVE_CUBES[m].to_facelets(), SOLVED[MOVE_PERMS[m]])


def test_move_cube_powers():
    for face in range(6):
        c = MOVE_CUBES[3 * face]
        c2 = cubie.multiply(c, c)
        c4 = cubie.multiply(c2, c2)
        assert c4 == SOLVED_CUBE and c2 == MOVE_CUBES[3 * face + 2]
        assert cubie.multiply(c2, c) == MOVE_CUBES[3 * face + 1]


def test_models_agree_on_random_sequences():
    rng = np.random.default_rng(2026)
    for _ in range(200):
        seq = rng.integers(0, 18, size=30).tolist()
        c = SOLVED_CUBE
        for m in seq:
            c = cubie.multiply(c, MOVE_CUBES[m])
        assert np.array_equal(c.to_facelets(), cube.apply_sequence(SOLVED, seq))
        assert c.is_valid()


def test_from_facelets_round_trip():
    rng = np.random.default_rng(11)
    for _ in range(300):
        c = cubie.random_cube(rng)
        f = c.to_facelets()
        assert f.dtype == np.uint8 and f.shape == (54,)
        assert np.array_equal(np.bincount(f, minlength=6), [9] * 6)
        assert cubie.from_facelets(f) == c
        assert CubieCube.from_facelets(f) == c


def test_random_cube_is_valid():
    rng = np.random.default_rng(12)
    for _ in range(500):
        assert cubie.random_cube(rng).is_valid()


def test_invalid_cubes():
    rng = np.random.default_rng(13)
    for _ in range(20):
        c = cubie.random_cube(rng)
        twist = c.copy()
        twist.co[3] = (twist.co[3] + 1) % 3
        assert not twist.is_valid()
        flip = c.copy()
        flip.eo[5] ^= 1
        assert not flip.is_valid()
        swap = c.copy()
        swap.cp[[0, 1]] = swap.cp[[1, 0]]
        assert not swap.is_valid()
        # each is still made of possible cubies, so it survives the facelet round trip
        for bad in (twist, flip, swap):
            assert cubie.from_facelets(bad.to_facelets()) == bad
        dup = c.copy()
        dup.cp[0] = dup.cp[1]
        assert not dup.is_valid()


def test_from_facelets_rejects_impossible_cubies():
    f = SOLVED.copy()
    f[8] = cube.R  # URF corner now R R F: no U/D sticker
    with pytest.raises(ValueError):
        cubie.from_facelets(f)
    f = SOLVED.copy()
    f[[9, 18]] = f[[18, 9]]  # URF corner U F R: mirror image, no such cubie
    with pytest.raises(ValueError):
        cubie.from_facelets(f)
    f = SOLVED.copy()
    f[8] = cube.D  # URF corner D R F: no such cubie
    with pytest.raises(ValueError):
        cubie.from_facelets(f)
    f = SOLVED.copy()
    f[7] = cube.B  # UF edge B F
    with pytest.raises(ValueError):
        cubie.from_facelets(f)
    f = SOLVED.copy()
    f[4] = cube.R  # a centre
    with pytest.raises(ValueError):
        cubie.from_facelets(f)
