"""Random states: scrambles from the solved cube, and a uniformly random set.

``random_scrambles`` is a vectorised random walk from SOLVED in the half-turn metric with a canonical filter
(see :func:`random_scrambles`). ``uniform_random_states`` draws uniformly over all reachable states via
:func:`rubiks.cubie.random_cube`.
"""

from __future__ import annotations

import numpy as np

from . import cube, cubie

# _NEXT_FACE[key, k]: the k-th allowed face at step t >= 1, with key = prev_face + 6 * opposite_pair.
#   key 0..5  : the previous two faces are not an opposite pair -> 5 allowed faces (all but prev_face).
#   key 6..11 : faces t-2, t-1 are opposite (same axis)          -> 4 allowed faces (all but that axis).
# A face's axis is face % 3 (U/D -> 0, R/L -> 1, F/B -> 2).
_NEXT_FACE = np.full((12, 5), -1, dtype=np.int8)
for _f in range(6):
    _NEXT_FACE[_f, :5] = [g for g in range(6) if g != _f]
    _NEXT_FACE[6 + _f, :4] = [g for g in range(6) if g % 3 != _f % 3]
_NEXT_FACE.setflags(write=False)

_CHUNK_ROWS = 32768  # rows per chunk when applying moves (keeps the gather index ~14 MB)


def _sample_faces(rng: np.random.Generator, n: int, max_depth: int) -> np.ndarray:
    """Faces (n, max_depth) int8 of canonical walks; the choice at each step is uniform over allowed faces."""
    faces = np.empty((n, max_depth), dtype=np.int8)
    faces[:, 0] = rng.integers(0, 6, size=n, dtype=np.int8)
    for t in range(1, max_depth):
        prev = faces[:, t - 1]
        if t >= 2:
            opposite = (faces[:, t - 2] % 3) == (prev % 3)  # consecutive faces always differ: same axis == opposite
        else:
            opposite = np.zeros(n, dtype=bool)
        x = rng.integers(0, 20, size=n, dtype=np.int8)  # 20 = lcm(4, 5): x % 5 and x % 4 are both uniform
        k = np.where(opposite, x % 4, x % 5)
        faces[:, t] = _NEXT_FACE[prev + 6 * opposite, k]
    return faces


def _apply_padded(moves: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """States reached from SOLVED by moves[i, :depth[i]] (vectorised; rows past their depth do not move)."""
    n, max_depth = moves.shape
    order = np.argsort(depth, kind="stable")[::-1]          # deepest rows first
    mv = moves[order]
    dp = depth[order].astype(np.int64)
    out = np.empty((n, cube.N_FACELETS), dtype=np.uint8)
    for start in range(0, n, _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, n)
        s = np.tile(cube.SOLVED, (stop - start, 1))
        d = dp[start:stop]                                   # non-increasing within the chunk
        for t in range(max_depth):
            active = int(np.count_nonzero(d > t))            # rows still moving are a prefix
            if active == 0:
                break
            s[:active] = np.take_along_axis(s[:active], cube.MOVE_PERMS[mv[start:start + active, t]], axis=1)
        out[start:stop] = s
    states = np.empty_like(out)
    states[order] = out
    return states


def random_scrambles(rng: np.random.Generator, n: int, max_depth: int, depths=None):
    """Random scrambles from SOLVED.

    Returns ``(states, moves, depth)``: ``states`` uint8 (n, 54), ``moves`` int8 (n, max_depth) padded with -1
    beyond each row's depth, ``depth`` uint8 (n,). The last move of row i is ``moves[i, depth[i] - 1]``.

    ``depth`` is uniform in [1, max_depth] unless ``depths`` (an int or an (n,) array in [1, max_depth]) is given.

    Canonical filter, per row: the face at step t differs from the face at step t-1; and when the faces at t-2
    and t-1 are opposite (same axis: U/D, R/L, F/B), the face at t differs from both. Each step picks uniformly
    among the allowed faces; the kind (clockwise, ', 2) is uniform. Every row is a prefix of a length-max_depth
    walk, which has the same distribution as a walk of its own length (the process is a forward Markov chain).

    RNG draw order (fixed): depth (only if ``depths`` is None), face at step 0, faces at steps 1..max_depth-1
    (one integer in [0, 20) per row and step), then the kinds (n, max_depth).
    """
    n = int(n)
    max_depth = int(max_depth)
    if n < 0:
        raise ValueError("n must be >= 0")
    if not 1 <= max_depth <= 255:
        raise ValueError("max_depth must be in [1, 255]")
    if depths is None:
        depth = rng.integers(1, max_depth + 1, size=n, dtype=np.int64)
    else:
        depth = np.broadcast_to(np.asarray(depths, dtype=np.int64), (n,)).copy()
        if n and (depth.min() < 1 or depth.max() > max_depth):
            raise ValueError("depths must be in [1, max_depth]")
    depth = depth.astype(np.uint8)

    faces = _sample_faces(rng, n, max_depth)
    kinds = rng.integers(0, 3, size=(n, max_depth), dtype=np.int8)
    moves = (faces * 3 + kinds).astype(np.int8)
    moves[np.arange(max_depth)[None, :] >= depth[:, None]] = -1

    states = _apply_padded(moves, depth)
    return states, moves, depth


def uniform_random_states(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` states (uint8 (n, 54)) drawn uniformly over all reachable states (``cubie.random_cube`` per row)."""
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    out = np.empty((n, cube.N_FACELETS), dtype=np.uint8)
    for i in range(n):
        out[i] = cubie.random_cube(rng).to_facelets()
    return out
