"""Facelet-level model of the 3x3x3 Rubik's Cube (Kociemba facelet convention).

A state is a ``uint8`` array of shape ``(54,)`` (or ``(N, 54)`` for a batch): entry ``i`` is the colour of
facelet ``i``. Faces are U=0, R=1, F=2, D=3, L=4, B=5 and ``facelet = face * 9 + row * 3 + col``. In the
solved state facelet ``i`` has colour ``i // 9``; the six centres (4, 13, 22, 31, 40, 49) never move.

Face views (``row 0`` is the top of the view, ``col 0`` its left edge):

* U is seen from above with B at the top; D from below with F at the top.
* F, R, B, L are seen straight on with U at the top.

The 18 half-turn-metric moves are permutations of facelet positions. Applying move ``m`` to a state ``s`` is
``s[MOVE_PERMS[m]]`` (``new[i] = old[perm[i]]``).

The six clockwise quarter turns are DERIVED FROM GEOMETRY here, not copied from a cubie table: every facelet
is given a 3D cubie position and an outward normal from the face views above, the turned layer is rotated by
-90 degrees about the face's outward normal (clockwise as seen looking at that face from outside), and the
permutation is read off where each facelet lands. The derivation is checked at import: the 8 non-centre
facelets of the turned face must move ``(r, c) -> (c, 2 - r)`` and exactly 20 facelets must move.
``rubiks.cubie`` is an independent second model built from Kociemba's cubie tables; the tests check that the
two agree on every move.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

FACE_NAMES = "URFDLB"
U, R, F, D, L, B = range(6)
N_FACELETS = 54
N_MOVES = 18

MOVE_NAMES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2",
              "D", "D'", "D2", "L", "L'", "L2", "B", "B'", "B2"]
_MOVE_INDEX = {name: i for i, name in enumerate(MOVE_NAMES)}

CENTRES = np.array([4, 13, 22, 31, 40, 49], dtype=np.int64)
CENTRES.setflags(write=False)

SOLVED = (np.arange(N_FACELETS) // 9).astype(np.uint8)
SOLVED.setflags(write=False)


# --------------------------------------------------------------------------------------------------------------
# Geometry. Axes: +x points to R, +y points to U, +z points to F (a right-handed frame).
# For each face: (outward normal, view "up" = towards row 0, view "right" = towards col 2).
# These three vectors encode exactly the face views of the convention:
#   U: from above, B at the top     -> up = -z (B), right = +x (col 2 borders R)
#   D: from below, F at the top     -> up = +z (F), right = +x (col 2 borders R)
#   F: U at top, col 0 L, col 2 R   -> up = +y,     right = +x
#   R: U at top, col 0 F, col 2 B   -> up = +y,     right = -z
#   B: U at top, col 0 R, col 2 L   -> up = +y,     right = -x
#   L: U at top, col 0 B, col 2 F   -> up = +y,     right = +z
# --------------------------------------------------------------------------------------------------------------
_EX = np.array([1, 0, 0])
_EY = np.array([0, 1, 0])
_EZ = np.array([0, 0, 1])

FACE_FRAMES: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    U: (+_EY, -_EZ, +_EX),
    R: (+_EX, +_EY, -_EZ),
    F: (+_EZ, +_EY, +_EX),
    D: (-_EY, +_EZ, +_EX),
    L: (-_EX, +_EY, +_EZ),
    B: (-_EZ, +_EY, -_EX),
}


def _build_sticker_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Return (cubie position, outward normal) for each of the 54 facelets, both int arrays of shape (54, 3)."""
    pos = np.zeros((N_FACELETS, 3), dtype=np.int64)
    nrm = np.zeros((N_FACELETS, 3), dtype=np.int64)
    for face, (normal, up, right) in FACE_FRAMES.items():
        # Every view must be a proper (non-mirrored) view from outside: right x up == outward normal.
        if not np.array_equal(np.cross(right, up), normal):
            raise AssertionError(f"face {FACE_NAMES[face]}: view frame is mirrored")
        for r in range(3):
            for c in range(3):
                i = face * 9 + r * 3 + c
                pos[i] = normal + right * (c - 1) + up * (1 - r)
                nrm[i] = normal
    return pos, nrm


STICKER_POS, STICKER_NORMAL = _build_sticker_geometry()
STICKER_POS.setflags(write=False)
STICKER_NORMAL.setflags(write=False)

_STICKER_INDEX = {(tuple(p), tuple(n)): i for i, (p, n) in enumerate(zip(STICKER_POS.tolist(),
                                                                         STICKER_NORMAL.tolist()))}
if len(_STICKER_INDEX) != N_FACELETS:  # every facelet must have a distinct (position, normal)
    raise AssertionError("sticker geometry is not injective")


def _clockwise_rotation(normal: np.ndarray) -> np.ndarray:
    """Integer 3x3 matrix of the rotation by -90 degrees about ``normal`` (clockwise seen from outside).

    Rodrigues with theta = -90 deg: R v = (n . v) n - n x v.
    """
    cols = [int(np.dot(normal, e)) * normal - np.cross(normal, e) for e in (_EX, _EY, _EZ)]
    return np.stack(cols, axis=1).astype(np.int64)


def _quarter_turn_perm(face: int) -> np.ndarray:
    """Facelet permutation of the clockwise quarter turn of ``face`` (new[i] = old[perm[i]])."""
    normal = FACE_FRAMES[face][0]
    rot = _clockwise_rotation(normal)
    perm = np.arange(N_FACELETS, dtype=np.int64)
    in_layer = STICKER_POS @ normal == 1
    for j in np.flatnonzero(in_layer):
        new_pos = tuple((rot @ STICKER_POS[j]).tolist())
        new_nrm = tuple((rot @ STICKER_NORMAL[j]).tolist())
        perm[_STICKER_INDEX[(new_pos, new_nrm)]] = j  # the facelet that was at j now sits there
    return perm


def _derive_move_perms() -> np.ndarray:
    perms = np.empty((N_MOVES, N_FACELETS), dtype=np.int64)
    for face in range(6):
        p = _quarter_turn_perm(face)
        # Checks of the derivation against the stated convention.
        if sorted(p.tolist()) != list(range(N_FACELETS)):
            raise AssertionError(f"{FACE_NAMES[face]}: not a permutation")
        if int(np.count_nonzero(p != np.arange(N_FACELETS))) != 20:
            raise AssertionError(f"{FACE_NAMES[face]}: a quarter turn must move exactly 20 facelets")
        for r in range(3):
            for c in range(3):
                # the facelet at (r, c) of the turned face moves to (c, 2 - r)
                if p[face * 9 + c * 3 + (2 - r)] != face * 9 + r * 3 + c:
                    raise AssertionError(f"{FACE_NAMES[face]}: face facelets do not rotate clockwise")
        p2 = p[p]          # apply p twice
        p3 = p2[p]         # apply p three times
        perms[3 * face + 0] = p
        perms[3 * face + 1] = p3
        perms[3 * face + 2] = p2
    return perms


MOVE_PERMS = _derive_move_perms()
MOVE_PERMS.setflags(write=False)


# --------------------------------------------------------------------------------------------------------------
# Moves on states
# --------------------------------------------------------------------------------------------------------------
def _check_move(m: int) -> int:
    m = int(m)
    if not 0 <= m < N_MOVES:
        raise ValueError(f"move index out of range: {m}")
    return m


def apply_move(state: np.ndarray, m: int) -> np.ndarray:
    """Apply move ``m`` to one state (54,) or to every row of a batch (N, 54). Returns a new array."""
    state = np.asarray(state)
    if state.shape[-1] != N_FACELETS:
        raise ValueError(f"last axis must have length {N_FACELETS}, got shape {state.shape}")
    return state[..., MOVE_PERMS[_check_move(m)]]


def apply_moves(states: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """Apply a possibly different move to each row: ``states`` (N, 54), ``moves`` (N,) ints in [0, 18)."""
    states = np.asarray(states)
    moves = np.asarray(moves)
    if states.ndim != 2 or states.shape[1] != N_FACELETS:
        raise ValueError(f"states must have shape (N, {N_FACELETS}), got {states.shape}")
    if moves.shape != (states.shape[0],):
        raise ValueError(f"moves must have shape ({states.shape[0]},), got {moves.shape}")
    if moves.size and (moves.min() < 0 or moves.max() >= N_MOVES):
        raise ValueError("move index out of range")
    return np.take_along_axis(states, MOVE_PERMS[moves], axis=1)


def apply_sequence(state: np.ndarray, seq: Iterable[int]) -> np.ndarray:
    """Apply the moves of ``seq`` in order to one state (or to every row of a batch)."""
    out = np.asarray(state)
    for m in seq:
        out = apply_move(out, m)
    return out.copy() if out is state else out


def sequence_perm(seq: Iterable[int]) -> np.ndarray:
    """The single facelet permutation equivalent to applying ``seq`` in order."""
    p = np.arange(N_FACELETS, dtype=np.int64)
    for m in seq:
        p = p[MOVE_PERMS[_check_move(m)]]
    return p


def inverse_move(m: int) -> int:
    """Inverse of move ``m``: X <-> X', X2 is its own inverse."""
    m = _check_move(m)
    face, kind = divmod(m, 3)
    return 3 * face + (1, 0, 2)[kind]


def inverse_sequence(seq: Iterable[int]) -> list[int]:
    """The sequence that undoes ``seq``: reversed, each move inverted."""
    return [inverse_move(m) for m in reversed(list(seq))]


def is_solved(state: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(state), SOLVED))


def is_solved_batch(states: np.ndarray) -> np.ndarray:
    """Boolean (N,) array: which rows of ``states`` (N, 54) are solved."""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != N_FACELETS:
        raise ValueError(f"states must have shape (N, {N_FACELETS}), got {states.shape}")
    return np.all(states == SOLVED, axis=1)


# --------------------------------------------------------------------------------------------------------------
# Text formats
# --------------------------------------------------------------------------------------------------------------
def parse_moves(text: str) -> list[int]:
    """Parse whitespace-separated moves such as ``"R U' F2"`` into move indices. Unknown tokens raise."""
    out = []
    for tok in text.split():
        if tok not in _MOVE_INDEX:
            raise ValueError(f"unknown move token: {tok!r}")
        out.append(_MOVE_INDEX[tok])
    return out


def format_moves(seq: Iterable[int]) -> str:
    return " ".join(MOVE_NAMES[_check_move(m)] for m in seq)


_FACE_CHARS = np.array(list(FACE_NAMES))
_CHAR_TO_FACE = {ch: i for i, ch in enumerate(FACE_NAMES)}


def to_facelet_string(state: np.ndarray) -> str:
    """54-character string over "URFDLB" in Kociemba's facelet order (solved = "UUUUUUUUURRRRRRRRR...")."""
    state = np.asarray(state)
    if state.shape != (N_FACELETS,):
        raise ValueError(f"state must have shape ({N_FACELETS},), got {state.shape}")
    if state.min() < 0 or state.max() > 5:
        raise ValueError("colours must be in 0..5")
    return "".join(_FACE_CHARS[state.astype(np.int64)])


def from_facelet_string(s: str) -> np.ndarray:
    """Inverse of :func:`to_facelet_string`. Checks length and alphabet only (not cube validity)."""
    if len(s) != N_FACELETS:
        raise ValueError(f"facelet string must have {N_FACELETS} characters, got {len(s)}")
    try:
        return np.array([_CHAR_TO_FACE[ch] for ch in s], dtype=np.uint8)
    except KeyError as exc:
        raise ValueError(f"invalid facelet character: {exc.args[0]!r}") from None


def print_net(state: np.ndarray) -> str:
    """ASCII net for debugging (returns the text). Each face is drawn in its own view::

              U
            L F R B
              D
    """
    state = np.asarray(state)
    if state.shape != (N_FACELETS,):
        raise ValueError(f"state must have shape ({N_FACELETS},), got {state.shape}")

    def row(face: int, r: int) -> str:
        return " ".join(FACE_NAMES[int(state[face * 9 + r * 3 + c])] for c in range(3))

    pad = " " * 8
    lines = [pad + row(U, r) for r in range(3)]
    lines += ["  ".join(row(f, r) for f in (L, F, R, B)) for r in range(3)]
    lines += [pad + row(D, r) for r in range(3)]
    return "\n".join(lines)


def perm_order(perm: Sequence[int] | np.ndarray) -> int:
    """Order of a permutation (lcm of its cycle lengths)."""
    perm = np.asarray(perm)
    seen = np.zeros(len(perm), dtype=bool)
    order = 1
    for start in range(len(perm)):
        if seen[start]:
            continue
        length = 0
        j = start
        while not seen[j]:
            seen[j] = True
            j = int(perm[j])
            length += 1
        order = int(np.lcm(order, length))
    return order
