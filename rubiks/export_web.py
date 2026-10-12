"""Export the cube model to the web page: ``python -m rubiks.export_web --out web/cube-tables.js --seed 20261012``.

Writes one classic-script JS file that sets ``window.CUBE_TABLES = {...};``. It is a script, not JSON fetched at
runtime, because ``fetch`` of a local file fails when the page is opened as ``file://``. The page renders and turns
the cube ONLY from these tables; nothing about the cube is re-derived in JavaScript. Every table comes from
``rubiks.cube`` (move permutations derived from geometry, sticker geometry) and ``rubiks.sampling`` (uniform states).

File layout: two ``//`` comment lines naming the generator, then ``window.CUBE_TABLES = {`` ... ``};`` whose body is
strict JSON. The file carries no timestamp: with the same seed, code and numpy version two runs give byte-identical
files, so ``git diff`` after a regeneration shows only real changes. Provenance is the git commit of the file and
the seed (plus the package and numpy versions recorded in ``generated``).

Seeding: ``np.random.SeedSequence(seed).spawn(2)`` gives two child sequences used, in this order, for the test
vectors and the uniform states, each through its own ``np.random.default_rng(child)``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__ as PACKAGE_VERSION
from . import cube, sampling

EXPORT_VERSION = "1.0.0"
JS_PREFIX = "window.CUBE_TABLES = "
JS_SUFFIX = ";"

N_TEST_VECTORS = 50
TEST_VECTOR_LENGTH = 30
N_UNIFORM_STATES = 1000

# Default sticker colour per face index (U R F D L B).
COLOURS = ["#ffffff", "#c41e3a", "#009e60", "#ffd500", "#ff5800", "#0051ba"]

# The dataset generator's canonical filter (rubiks.sampling.random_scrambles); a face's axis is face % 3.
SCRAMBLE_FILTER = {
    "rule": "face(t) != face(t-1); if face(t-2) and face(t-1) are opposite, face(t) differs from both",
    "opposite": [[0, 3], [1, 4], [2, 5]],
    "first_face": "uniform over the 6 faces",
    "next_face": "uniform over the allowed faces (5, or 4 after an opposite pair)",
    "kind": "uniform over the 3 moves of the chosen face",
}

IDENTITY_SEQUENCE = "R L U2 R L' B2 U2 R2 F2 L2 D2 L2 F2"
SEQUENCE_ORDERS = {"R U": 105, "R U R' U'": 6, "R2 U2": 6}

# Top-left cell (row, col) of each face in the 2D net of rubiks.cube.print_net: U on top, then L F R B, D below.
NET_ORIGIN = {"U": (0, 3), "L": (3, 0), "F": (3, 3), "R": (3, 6), "B": (3, 9), "D": (6, 3)}
NET_ROWS, NET_COLS = 9, 12

# Signed clockwise quarter turns per move-name suffix: X = 1, X' = -1, X2 = 2.
_SUFFIX_QUARTER_TURNS = {"": 1, "'": -1, "2": 2}


def _move_face_and_turns() -> tuple[list[int], list[int]]:
    """Face index and signed clockwise quarter turns of each move, from its name, checked against MOVE_PERMS."""
    faces, turns = [], []
    ident = np.arange(cube.N_FACELETS)
    for m, name in enumerate(cube.MOVE_NAMES):
        face = cube.FACE_NAMES.index(name[0])
        q = _SUFFIX_QUARTER_TURNS[name[1:]]
        p = ident.copy()
        for _ in range(q % 4):
            p = p[cube.MOVE_PERMS[3 * face]]  # the clockwise quarter turn of that face, q times
        if not np.array_equal(p, cube.MOVE_PERMS[m]):
            raise AssertionError(f"move {name}: name does not match its permutation")
        faces.append(face)
        turns.append(q)
    return faces, turns


def _net_pos() -> list[list[int]]:
    """(net row, net col) of each facelet: each face drawn in its own view, as in rubiks.cube.print_net."""
    out = []
    for i in range(cube.N_FACELETS):
        face, k = divmod(i, 9)
        r, c = divmod(k, 3)
        r0, c0 = NET_ORIGIN[cube.FACE_NAMES[face]]
        out.append([r0 + r, c0 + c])
    return out


def _rotation(normal: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Integer matrix of the rotation by -90 deg * quarter_turns about ``normal`` (right-hand rule)."""
    n = np.asarray(normal, dtype=np.int64)
    k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]], dtype=np.int64)  # k @ v == n x v
    theta = -np.pi / 2 * quarter_turns
    c, s = int(round(np.cos(theta))), int(round(np.sin(theta)))
    return c * np.eye(3, dtype=np.int64) + s * k + (1 - c) * np.outer(n, n)


def check_turn_geometry(face_normal, move_face, move_quarter_turns, move_perms, sticker_pos, sticker_normal) -> None:
    """Check the animation rule against the permutations; raises AssertionError on any mismatch.

    Rule: move m rotates the layer {stickers with sticker_pos . n == 1}, n = face_normal[move_face[m]], by
    -90 deg * move_quarter_turns[m] about n; the sticker that starts at facelet j must end on the facelet i
    with move_perms[m][i] == j (new[i] = old[perm[i]]), and every other sticker must stay put.
    """
    pos = np.asarray(sticker_pos, dtype=np.int64)
    nrm = np.asarray(sticker_normal, dtype=np.int64)
    index = {(tuple(p), tuple(q)): i for i, (p, q) in enumerate(zip(pos.tolist(), nrm.tolist()))}
    for m, perm in enumerate(move_perms):
        n = np.asarray(face_normal[move_face[m]], dtype=np.int64)
        rot = _rotation(n, move_quarter_turns[m])
        for j in range(len(pos)):
            if int(pos[j] @ n) == 1:
                i = index.get((tuple((rot @ pos[j]).tolist()), tuple((rot @ nrm[j]).tolist())))
                if i is None:
                    raise AssertionError(f"move {cube.MOVE_NAMES[m]}: sticker {j} rotates off the sticker grid")
            else:
                i = j
            if int(perm[i]) != j:
                raise AssertionError(f"move {cube.MOVE_NAMES[m]}: sticker {j} lands on {i}, perm[{i}] = {perm[i]}")


def build_tables(seed: int) -> dict:
    """All tables of the page, as plain JSON-serialisable Python objects."""
    seq_vectors, seq_uniform = np.random.SeedSequence(seed).spawn(2)

    rng = np.random.default_rng(seq_vectors)
    sequences = rng.integers(0, cube.N_MOVES, size=(N_TEST_VECTORS, TEST_VECTOR_LENGTH))
    vectors = [{"moves": [int(m) for m in seq], "state": cube.apply_sequence(cube.SOLVED, seq).tolist()}
               for seq in sequences]

    uniform = sampling.uniform_random_states(np.random.default_rng(seq_uniform), N_UNIFORM_STATES)

    # The group facts the page re-checks at load must hold for the Python model first.
    if not cube.is_solved(cube.apply_sequence(cube.SOLVED, cube.parse_moves(IDENTITY_SEQUENCE))):
        raise AssertionError("identity sequence does not return to solved")
    for text, order in SEQUENCE_ORDERS.items():
        if cube.perm_order(cube.sequence_perm(cube.parse_moves(text))) != order:
            raise AssertionError(f"order of {text!r} is not {order}")

    move_face, move_quarter_turns = _move_face_and_turns()
    face_normal = [cube.STICKER_NORMAL[c].tolist() for c in cube.CENTRES]
    for face in range(6):
        if face_normal[face] != cube.FACE_FRAMES[face][0].tolist():
            raise AssertionError(f"face {cube.FACE_NAMES[face]}: centre normal is not the face normal")
    check_turn_geometry(face_normal, move_face, move_quarter_turns, cube.MOVE_PERMS.tolist(),
                        cube.STICKER_POS, cube.STICKER_NORMAL)

    return {
        "format": "rubiks-nn web tables",
        "format_version": 1,
        "face_names": cube.FACE_NAMES,
        "move_names": list(cube.MOVE_NAMES),
        "move_perms": cube.MOVE_PERMS.tolist(),
        "inverse_move": [cube.inverse_move(m) for m in range(cube.N_MOVES)],
        "move_face": move_face,
        "move_quarter_turns": move_quarter_turns,
        "solved": cube.SOLVED.tolist(),
        "centres": cube.CENTRES.tolist(),
        "sticker_pos": cube.STICKER_POS.tolist(),
        "sticker_normal": cube.STICKER_NORMAL.tolist(),
        "face_normal": face_normal,
        "facelet_names": [f"{cube.FACE_NAMES[i // 9]}{i % 9 + 1}" for i in range(cube.N_FACELETS)],
        "net_size": [NET_ROWS, NET_COLS],
        "net_pos": _net_pos(),
        "colours": list(COLOURS),
        "conventions": {
            "state": "54 colours 0..5, Kociemba facelet order URFDLB, facelet = face*9 + row*3 + col",
            "apply": "new[i] = old[move_perms[m][i]]",
            "axes": "+x towards R, +y towards U, +z towards F (right-handed)",
            "turn": "move m rotates the stickers with sticker_pos . n == 1, n = face_normal[move_face[m]], by "
                    "-90 deg * move_quarter_turns[m] about n (right-hand rule; clockwise seen from outside for 1)",
        },
        "scramble_filter": SCRAMBLE_FILTER,
        "test_vectors": {
            "rule": f"{N_TEST_VECTORS} sequences of {TEST_VECTOR_LENGTH} moves, each move uniform over the 18 "
                    "(no filter), applied to solved with rubiks.cube.apply_sequence",
            "sequences": vectors,
            "identity": IDENTITY_SEQUENCE,
            "orders": dict(SEQUENCE_ORDERS),
        },
        "uniform_states": uniform.tolist(),
        "generated": {
            "generator": "rubiks.export_web",
            "export_version": EXPORT_VERSION,
            "seed": int(seed),
            "seed_scheme": "np.random.SeedSequence(seed).spawn(2) -> test_vectors, uniform_states; "
                           "np.random.default_rng(child) each",
            "uniform_rule": "rubiks.sampling.uniform_random_states (rubiks.cubie.random_cube per state)",
            "package_version": PACKAGE_VERSION,
            "numpy_version": np.__version__,
            "provenance": "no timestamp by design (same seed -> byte-identical file); the git commit of this "
                          "file and the seed identify it",
        },
    }


def _dump_value(value) -> str:
    """Compact JSON, except that a list of lists / objects gets one element per line (readable diffs)."""
    compact = {"separators": (",", ":"), "ensure_ascii": True}
    if isinstance(value, list) and value and all(isinstance(v, (list, dict)) for v in value):
        return "[\n    " + ",\n    ".join(json.dumps(v, **compact) for v in value) + "\n  ]"
    if isinstance(value, dict) and any(isinstance(v, list) and v and isinstance(v[0], (list, dict))
                                       for v in value.values()):
        items = [f"    {json.dumps(k)}: " + _dump_value(v).replace("\n", "\n  ") for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n  }"
    return json.dumps(value, **compact)


def render_js(tables: dict) -> str:
    body = "{\n" + ",\n".join(f"  {json.dumps(k)}: {_dump_value(v)}" for k, v in tables.items()) + "\n}"
    json.loads(body)  # the body must stay strict JSON (the tests parse it with json.loads)
    seed = tables["generated"]["seed"]
    header = [
        f"// cube-tables.js: GENERATED by `python -m rubiks.export_web --out web/cube-tables.js --seed {seed}`. "
        "Do not edit by hand.",
        "// Source of truth: rubiks/cube.py (move permutations, sticker geometry) and rubiks/sampling.py "
        "(uniform states). Format: web/README.md.",
    ]
    return "\n".join(header) + "\n" + JS_PREFIX + body + JS_SUFFIX + "\n"


def parse_js(text: str) -> dict:
    """Inverse of :func:`render_js` (comment lines are dropped). Used by the tests."""
    code = "\n".join(line for line in text.splitlines() if not line.startswith("//")).strip()
    if not code.startswith(JS_PREFIX) or not code.endswith(JS_SUFFIX):
        raise ValueError("not a cube-tables.js file")
    return json.loads(code[len(JS_PREFIX):-len(JS_SUFFIX)])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m rubiks.export_web", description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path, help="output .js file (overwritten; parent dirs created)")
    p.add_argument("--seed", type=int, default=20261012, help="root seed for SeedSequence (default 20261012)")
    args = p.parse_args(argv)
    if args.seed < 0:
        p.error("--seed must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tables = build_tables(args.seed)
    text = render_js(tables)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="ascii")
    print(f"rubiks.export_web {EXPORT_VERSION} -> {args.out} ({len(text.encode('ascii'))} bytes): seed {args.seed}, "
          f"{N_TEST_VECTORS} test vectors x {TEST_VECTOR_LENGTH} moves, {N_UNIFORM_STATES} uniform states, "
          f"numpy {np.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
