"""Cubie-level model of the 3x3x3 cube (Kociemba's representation): the independent second model.

A :class:`CubieCube` holds ``cp`` (8,), ``co`` (8,), ``ep`` (12,), ``eo`` (12,): ``cp[i]`` is the corner cubie
now at corner position ``i`` and ``co[i]`` in 0..2 its orientation; ``ep``/``eo`` likewise for edges with ``eo``
in {0, 1}. Corner order URF UFL ULB UBR DFR DLF DBL DRB; edge order UR UF UL UB DR DF DL DB FR FL BL BR.

The six basic move cubes are Kociemba's reference tables, transcribed here. This module deliberately does NOT
import ``rubiks.cube``: its move permutations are derived from geometry, these come from the cubie tables, and the
tests check that the two models agree on every move and on random sequences.
"""

from __future__ import annotations

import numpy as np

# Faces and facelet numbering (duplicated from the convention on purpose; see the module docstring).
FACE_NAMES = "URFDLB"
U, R, F, D, L, B = range(6)
N_FACELETS = 54
CENTRES = np.array([4, 13, 22, 31, 40, 49], dtype=np.int64)
MOVE_NAMES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2",
              "D", "D'", "D2", "L", "L'", "L2", "B", "B'", "B2"]

CORNER_NAMES = ["URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB"]
EDGE_NAMES = ["UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR"]
URF, UFL, ULB, UBR, DFR, DLF, DBL, DRB = range(8)
UR, UF, UL, UB, DR, DF, DL, DB, FR, FL, BL, BR = range(12)


def _facelet(name: str) -> int:
    """Kociemba 1-based facelet name such as "U9" -> 0-based facelet index."""
    return FACE_NAMES.index(name[0]) * 9 + int(name[1:]) - 1


# Facelet positions of each corner / edge position; the first entry is the U/D facelet (or the F/B facelet for
# the four middle-layer edges). Orientation k means the cubie's sticker list is rotated by k along this list.
CORNER_FACELET = np.array([[_facelet(x) for x in row] for row in [
    ["U9", "R1", "F3"],  # URF
    ["U7", "F1", "L3"],  # UFL
    ["U1", "L1", "B3"],  # ULB
    ["U3", "B1", "R3"],  # UBR
    ["D3", "F9", "R7"],  # DFR
    ["D1", "L9", "F7"],  # DLF
    ["D7", "B9", "L7"],  # DBL
    ["D9", "R9", "B7"],  # DRB
]], dtype=np.int64)
EDGE_FACELET = np.array([[_facelet(x) for x in row] for row in [
    ["U6", "R2"],  # UR
    ["U8", "F2"],  # UF
    ["U4", "L2"],  # UL
    ["U2", "B2"],  # UB
    ["D6", "R8"],  # DR
    ["D2", "F8"],  # DF
    ["D4", "L8"],  # DL
    ["D8", "B8"],  # DB
    ["F6", "R4"],  # FR
    ["F4", "L6"],  # FL
    ["B6", "L4"],  # BL
    ["B4", "R6"],  # BR
]], dtype=np.int64)
# Colours of each cubie's stickers in reference orientation (the solved colour of those facelets).
CORNER_COLOR = CORNER_FACELET // 9
EDGE_COLOR = EDGE_FACELET // 9
for _a in (CORNER_FACELET, EDGE_FACELET, CORNER_COLOR, EDGE_COLOR):
    _a.setflags(write=False)

_CORNER_BY_COLORS = {tuple(c): j for j, c in enumerate(CORNER_COLOR.tolist())}
_EDGE_BY_COLORS = {tuple(c): j for j, c in enumerate(EDGE_COLOR.tolist())}


class CubieCube:
    """A cube as corner/edge permutation + orientation (Kociemba's cubie level)."""

    __slots__ = ("cp", "co", "ep", "eo")

    def __init__(self, cp=None, co=None, ep=None, eo=None):
        self.cp = np.arange(8, dtype=np.int64) if cp is None else np.array(cp, dtype=np.int64)
        self.co = np.zeros(8, dtype=np.int64) if co is None else np.array(co, dtype=np.int64)
        self.ep = np.arange(12, dtype=np.int64) if ep is None else np.array(ep, dtype=np.int64)
        self.eo = np.zeros(12, dtype=np.int64) if eo is None else np.array(eo, dtype=np.int64)
        for name, arr, size in (("cp", self.cp, 8), ("co", self.co, 8), ("ep", self.ep, 12), ("eo", self.eo, 12)):
            if arr.shape != (size,):
                raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")

    def copy(self) -> "CubieCube":
        return CubieCube(self.cp, self.co, self.ep, self.eo)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CubieCube):
            return NotImplemented
        return (np.array_equal(self.cp, other.cp) and np.array_equal(self.co, other.co)
                and np.array_equal(self.ep, other.ep) and np.array_equal(self.eo, other.eo))

    __hash__ = None  # mutable

    def __repr__(self) -> str:
        return (f"CubieCube(cp={self.cp.tolist()}, co={self.co.tolist()}, "
                f"ep={self.ep.tolist()}, eo={self.eo.tolist()})")

    # -- moves ---------------------------------------------------------------------------------------------
    def apply_move(self, m: int) -> "CubieCube":
        """Return this cube with move ``m`` applied (``multiply(self, MOVE_CUBES[m])``)."""
        return multiply(self, MOVE_CUBES[int(m)])

    def apply_sequence(self, seq) -> "CubieCube":
        c = self
        for m in seq:
            c = multiply(c, MOVE_CUBES[int(m)])
        return c if c is not self else self.copy()

    # -- facelets ------------------------------------------------------------------------------------------
    def to_facelets(self) -> np.ndarray:
        """Facelet state (54,) uint8, as Kociemba's toFaceCube:
        ``f[CORNER_FACELET[i][(n + co[i]) % 3]] = CORNER_COLOR[cp[i]][n]`` and likewise for edges mod 2."""
        f = np.zeros(N_FACELETS, dtype=np.uint8)
        f[CENTRES] = np.arange(6, dtype=np.uint8)
        k3 = (np.arange(3)[None, :] + self.co[:, None]) % 3
        f[CORNER_FACELET[np.arange(8)[:, None], k3]] = CORNER_COLOR[self.cp]
        k2 = (np.arange(2)[None, :] + self.eo[:, None]) % 2
        f[EDGE_FACELET[np.arange(12)[:, None], k2]] = EDGE_COLOR[self.ep]
        return f

    @classmethod
    def from_facelets(cls, state) -> "CubieCube":
        """Inverse of :meth:`to_facelets`.

        Raises ValueError if a centre is wrong or a corner/edge position holds an impossible cubie (a colour
        combination that is no cubie in any orientation). A cube made of possible cubies can still be
        unreachable (duplicated cubie, twisted corner, parity); :meth:`is_valid` reports that.
        """
        f = np.asarray(state)
        if f.shape != (N_FACELETS,):
            raise ValueError(f"state must have shape ({N_FACELETS},), got {f.shape}")
        f = f.astype(np.int64)
        if not np.array_equal(f[CENTRES], np.arange(6)):
            raise ValueError("centres are not U R F D L B")
        cp = np.empty(8, dtype=np.int64)
        co = np.empty(8, dtype=np.int64)
        for i in range(8):
            stickers = f[CORNER_FACELET[i]]
            ud = np.flatnonzero((stickers == U) | (stickers == D))
            if len(ud) != 1:
                raise ValueError(f"corner position {CORNER_NAMES[i]}: impossible cubie {stickers.tolist()}")
            ori = int(ud[0])
            ref = tuple(int(stickers[(n + ori) % 3]) for n in range(3))  # sticker n of the cubie
            if ref not in _CORNER_BY_COLORS:
                raise ValueError(f"corner position {CORNER_NAMES[i]}: impossible cubie {stickers.tolist()}")
            cp[i] = _CORNER_BY_COLORS[ref]
            co[i] = ori
        ep = np.empty(12, dtype=np.int64)
        eo = np.empty(12, dtype=np.int64)
        for i in range(12):
            a, b = (int(x) for x in f[EDGE_FACELET[i]])
            if (a, b) in _EDGE_BY_COLORS:
                ep[i], eo[i] = _EDGE_BY_COLORS[(a, b)], 0
            elif (b, a) in _EDGE_BY_COLORS:
                ep[i], eo[i] = _EDGE_BY_COLORS[(b, a)], 1
            else:
                raise ValueError(f"edge position {EDGE_NAMES[i]}: impossible cubie {[a, b]}")
        return cls(cp, co, ep, eo)

    # -- validity ------------------------------------------------------------------------------------------
    def is_valid(self) -> bool:
        """True iff the cube is reachable: permutations, orientation sums 0, equal corner/edge parity."""
        if sorted(self.cp.tolist()) != list(range(8)) or sorted(self.ep.tolist()) != list(range(12)):
            return False
        if self.co.min() < 0 or self.co.max() > 2 or self.eo.min() < 0 or self.eo.max() > 1:
            return False
        if int(self.co.sum()) % 3 != 0 or int(self.eo.sum()) % 2 != 0:
            return False
        return parity(self.cp) == parity(self.ep)


def from_facelets(state) -> CubieCube:
    """Module-level alias of :meth:`CubieCube.from_facelets`."""
    return CubieCube.from_facelets(state)


def parity(perm) -> int:
    """Parity of a permutation: 0 even, 1 odd (inversion count mod 2)."""
    p = [int(x) for x in perm]
    inv = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            inv += p[i] > p[j]
    return inv % 2


def multiply(a: CubieCube, b: CubieCube) -> CubieCube:
    """Kociemba's cube product: apply ``a`` then ``b``.

    ``cp[i] = a.cp[b.cp[i]]``, ``co[i] = (a.co[b.cp[i]] + b.co[i]) % 3``; edges likewise mod 2.
    """
    return CubieCube(a.cp[b.cp], (a.co[b.cp] + b.co) % 3, a.ep[b.ep], (a.eo[b.ep] + b.eo) % 2)


SOLVED_CUBE = CubieCube()

# Kociemba's six basic clockwise move cubes.
_BASIC = {
    U: CubieCube(cp=[UBR, URF, UFL, ULB, DFR, DLF, DBL, DRB], co=[0] * 8,
                 ep=[UB, UR, UF, UL, DR, DF, DL, DB, FR, FL, BL, BR], eo=[0] * 12),
    R: CubieCube(cp=[DFR, UFL, ULB, URF, DRB, DLF, DBL, UBR], co=[2, 0, 0, 1, 1, 0, 0, 2],
                 ep=[FR, UF, UL, UB, BR, DF, DL, DB, DR, FL, BL, UR], eo=[0] * 12),
    F: CubieCube(cp=[UFL, DLF, ULB, UBR, URF, DFR, DBL, DRB], co=[1, 2, 0, 0, 2, 1, 0, 0],
                 ep=[UR, FL, UL, UB, DR, FR, DL, DB, UF, DF, BL, BR], eo=[0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]),
    D: CubieCube(cp=[URF, UFL, ULB, UBR, DLF, DBL, DRB, DFR], co=[0] * 8,
                 ep=[UR, UF, UL, UB, DF, DL, DB, DR, FR, FL, BL, BR], eo=[0] * 12),
    L: CubieCube(cp=[URF, ULB, DBL, UBR, DFR, UFL, DLF, DRB], co=[0, 1, 2, 0, 0, 2, 1, 0],
                 ep=[UR, UF, BL, UB, DR, DF, FL, DB, FR, UL, DL, BR], eo=[0] * 12),
    B: CubieCube(cp=[URF, UFL, UBR, DRB, DFR, DLF, ULB, DBL], co=[0, 0, 1, 2, 0, 0, 2, 1],
                 ep=[UR, UF, UL, BR, DR, DF, DL, BL, FR, FL, UB, DB], eo=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]),
}


def _build_move_cubes() -> list[CubieCube]:
    cubes: list[CubieCube] = []
    for face in range(6):
        c1 = _BASIC[face]
        c2 = multiply(c1, c1)
        c3 = multiply(c2, c1)
        cubes += [c1, c3, c2]  # X, X', X2
    return cubes


MOVE_CUBES = _build_move_cubes()


def random_cube(rng: np.random.Generator) -> CubieCube:
    """A cube drawn uniformly from the 43,252,003,274,489,856,000 reachable states.

    Draw order (fixed, for reproducibility): cp = permutation(8), ep = permutation(12), co[0:7], eo[0:11].
    If the parities of cp and ep differ, ep[0] and ep[1] are swapped (a bijection between the parity classes,
    so ep stays uniform given cp). The last orientation entries are fixed so the sums are 0 mod 3 / mod 2.
    """
    cp = rng.permutation(8).astype(np.int64)
    ep = rng.permutation(12).astype(np.int64)
    if parity(cp) != parity(ep):
        ep[[0, 1]] = ep[[1, 0]]
    co = np.zeros(8, dtype=np.int64)
    co[:7] = rng.integers(0, 3, size=7)
    co[7] = (-int(co[:7].sum())) % 3
    eo = np.zeros(12, dtype=np.int64)
    eo[:11] = rng.integers(0, 2, size=11)
    eo[11] = (-int(eo[:11].sum())) % 2
    return CubieCube(cp, co, ep, eo)
