"""Parity cases for the in-browser beam search (``web/search.js``), checked here against ``rubiks/search.py``.

``tests/data/web-parity.json`` holds two kinds of cases. Every expected answer comes from
``rubiks.search.beam_search``:

* ``onnx``: 20 seeded scrambles (depths 5 to 20, beam widths 64 and 256, max depth 40), solved with
  ``models/policy-v2.onnx`` run by onnxruntime. The policy function computes the log-softmax exactly as
  ``web/search.js`` does (float64: the maximum, the sum of exponentials in move order, the log; rounded to float32),
  so the only difference left between this side and the browser is the ONNX Runtime build that computes the logits.
* ``logic``: a synthetic policy with exact values (multiples of 1/64, so every float32 sum is exact) and many exact
  score ties. These cases pin down the tie-breaking order (state key, then child index), the width-1 (greedy) and
  max-depth edge cases, and ``no_backtrack=False``; they need no onnxruntime.

When the fixture is missing, the ``parity`` fixture below writes it (onnxruntime needed); otherwise the tests check
that the Python search still reproduces every case. ``web/search-test.html`` runs the same cases through
``web/search.js`` in a browser and prints ``PARITY n/n PASS``.

Decision margins. Two ONNX Runtime builds (native here, WebAssembly in the browser) give logits that differ in the
last bits (about 1e-6), so a case whose answer hinges on two scores closer than that could come out differently for
a numerical reason rather than a search one. :func:`reference_search` is the algorithm of ``web/search.js`` written
in numpy (the same candidate order, the same four-part state key, the same sort keys); it must agree with
``beam_search`` on every case, and it also records the smallest score gap that any decision depended on (the beam
cut-off, the choice among duplicates of a kept state, the choice among solved children). The ONNX scrambles come
from one seeded stream; each case takes the next scramble whose margin is at least ``MARGIN_MIN``, and the fixture
records the margin and how many scrambles were drawn.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from rubiks import cube, sampling, search
from rubiks.data import encode_numpy

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "data" / "web-parity.json"
MODEL = REPO / "models" / "policy-v2.onnx"
SEARCH_JS = REPO / "web" / "search.js"
SOLVER_JS = REPO / "web" / "solver.js"
INDEX_HTML = REPO / "web" / "index.html"

FORMAT = "rubiks-nn web search parity"
SEED = 20261012
N_ONNX = 20
ONNX_WIDTHS = (64, 256)
MAX_DEPTH = 40
MARGIN_MIN = 1e-4
MAX_DRAWS = 200
LOGIC_POLICY = "misplaced-hash-v1"
# (scramble depth, width, max depth, no_backtrack); depth 0 is the solved cube
LOGIC_CASES = [
    (0, 64, 40, True),     # solved input: [] with 0 nodes
    (1, 1, 6, True),
    (3, 1, 8, True),       # width 1: a greedy walk
    (3, 2, 8, True),
    (4, 7, 8, True),
    (4, 32, 8, False),     # only the same-face rule
    (5, 64, 10, True),
    (6, 256, 8, True),
    (8, 16, 12, True),
    (2, 3, 0, True),       # max depth 0: nothing is expanded
    (4, 4096, 5, True),    # a beam wide enough to keep every state of depth <= 3
]

_FACE = np.arange(cube.N_MOVES) // 3
_SOLVED = np.asarray(cube.SOLVED)
_NON_CENTRE = np.array([i for i in range(cube.N_FACELETS) if i not in set(cube.CENTRES.tolist())])
_POW12 = 6.0 ** np.arange(12)  # exact in float64


# --- policies --------------------------------------------------------------------------------------------------

def log_softmax_like_web(logits: np.ndarray) -> np.ndarray:
    """web/search.js logSoftmax: float64 maximum, sum of exp in move order, log; the result rounded to float32."""
    x = np.asarray(logits).astype(np.float64)
    mx = x.max(axis=1)
    s = np.zeros(x.shape[0])
    for j in range(x.shape[1]):
        s += np.exp(x[:, j] - mx)
    return (x - (mx + np.log(s))[:, None]).astype(np.float32)


def logic_values(states: np.ndarray) -> np.ndarray:
    """The synthetic policy (float32 (N, 18)): for move m, child = the state after m (child[i] = state[perm_m[i]]);
    value = -(4 * misplaced(child) + (sum_i child[i] * (i + 1)) mod 5) / 64."""
    children = np.asarray(states)[:, cube.MOVE_PERMS]                  # (N, 18, 54)
    misplaced = (children != _SOLVED).sum(axis=2)
    h = (children.astype(np.int64) * np.arange(1, cube.N_FACELETS + 1)).sum(axis=2) % 5
    return (-(4 * misplaced + h) / 64.0).astype(np.float32)


def torch_policy(raw_fn, normalize: bool):
    """A rubiks.search policy_fn from a numpy function of uint8 (N, 54) states."""
    def policy_fn(states: torch.Tensor) -> torch.Tensor:
        out = raw_fn(states.cpu().numpy())
        return torch.from_numpy(log_softmax_like_web(out) if normalize else np.asarray(out, dtype=np.float32))

    policy_fn.device = torch.device("cpu")  # type: ignore[attr-defined]
    return policy_fn


def onnx_raw(session):
    def raw(states: np.ndarray) -> np.ndarray:
        return session.run(["logits"], {"onehot": encode_numpy(states)})[0]
    return raw


# --- the algorithm of web/search.js, in numpy ---------------------------------------------------------------------

def _key_parts(children: np.ndarray) -> np.ndarray:
    """(n, 4) float64: k0 high, k0 low, k1 high, k1 low (web/search.js keyParts; rubiks.search.state_keys split)."""
    x = children[:, _NON_CENTRE].astype(np.float64)
    return np.stack([x[:, 12:24] @ _POW12, x[:, 0:12] @ _POW12, x[:, 36:48] @ _POW12, x[:, 24:36] @ _POW12], axis=1)


def reference_search(raw_fn, normalize: bool, state, width: int, max_depth: int, no_backtrack: bool = True):
    """web/search.js beamSearch in numpy. Returns (moves, nodes, depth, margin, solved_children); margin is the
    smallest score gap a decision depended on (inf when none)."""
    s0 = np.asarray(state, dtype=np.uint8)
    if np.array_equal(s0, _SOLVED):
        return [], 0, 0, math.inf, 0
    beam = s0[None].copy()
    score = np.zeros(1, dtype=np.float32)
    last_face = np.full(1, -1)
    lock_axis = np.full(1, -1)
    hist_parent: list[np.ndarray] = []
    hist_move: list[np.ndarray] = []
    nodes = 0
    margin = math.inf
    for depth in range(1, max_depth + 1):
        b = beam.shape[0]
        nodes += b
        out = raw_fn(beam)
        logp = log_softmax_like_web(out) if normalize else np.asarray(out, dtype=np.float32)
        allowed = _FACE[None, :] != last_face[:, None]
        if no_backtrack:
            allowed &= (lock_axis[:, None] < 0) | ((_FACE[None, :] % 3) != lock_axis[:, None])
        parent, move = np.nonzero(allowed)                                   # parent-major order
        children = np.take_along_axis(beam[parent], cube.MOVE_PERMS[move], axis=1)
        cscore = score[parent] + logp[parent, move]                          # float32 + float32
        if np.isnan(cscore).any():
            raise ValueError("NaN log-probabilities")

        solved_idx = np.flatnonzero((children == _SOLVED).all(axis=1))
        if solved_idx.size:
            j = int(solved_idx[0])
            for k in solved_idx[1:]:
                if cscore[k] > cscore[j]:
                    j = int(k)
            if solved_idx.size > 1:
                top2 = np.sort(cscore[solved_idx].astype(np.float64))[::-1][:2]
                margin = min(margin, float(top2[0] - top2[1]))
            path = [int(move[j])]
            bi = int(parent[j])
            for k in range(len(hist_parent) - 1, -1, -1):
                path.append(int(hist_move[k][bi]))
                bi = int(hist_parent[k][bi])
            path.reverse()
            assert cube.is_solved(cube.apply_sequence(s0, path))
            return path, nodes, depth, margin, int(solved_idx.size)
        if depth == max_depth:
            break

        # dedupe: (key ascending, score descending, index ascending); the first child of each key survives
        keys = _key_parts(children)
        neg = -cscore.astype(np.float64)
        order = np.lexsort((np.arange(len(cscore)), neg, keys[:, 3], keys[:, 2], keys[:, 1], keys[:, 0]))
        ko = keys[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = (ko[1:] != ko[:-1]).any(axis=1)
        uniq = order[first]
        # the `width` best survivors: score descending, ties in key order
        pos = np.lexsort((np.arange(len(uniq)), neg[uniq]))
        kept = pos[:width]
        sel = uniq[kept]

        if len(uniq) > width:
            ranked = cscore[uniq][pos].astype(np.float64)
            margin = min(margin, float(ranked[width - 1] - ranked[width]))
        starts = np.flatnonzero(first)
        nxt = starts + 1
        has_dup = nxt < len(order)
        has_dup[has_dup] = ~first[nxt[has_dup]]
        dup_gap = np.full(len(uniq), math.inf)
        dup_gap[has_dup] = (cscore[order[starts[has_dup]]].astype(np.float64)
                            - cscore[order[nxt[has_dup]]].astype(np.float64))
        if kept.size:
            margin = min(margin, float(np.min(dup_gap[kept])))

        sel_parent = parent[sel]
        sel_move = move[sel]
        new_face = _FACE[sel_move]
        parent_face = last_face[sel_parent]
        lock_axis = np.where((parent_face >= 0) & (parent_face % 3 == new_face % 3), new_face % 3, -1)
        last_face = new_face
        beam = children[sel]
        score = cscore[sel]
        hist_parent.append(sel_parent)
        hist_move.append(sel_move)
    return None, nodes, max_depth, margin, 0


# --- fixture ------------------------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f9(x: float) -> float:
    return float(f"{float(x):.9g}")


def _margin(x: float):
    return None if math.isinf(x) else float(f"{x:.3g}")


def _onnx_session():
    ort = pytest.importorskip("onnxruntime")
    if not MODEL.is_file():
        pytest.skip(f"{MODEL} is missing")
    return ort, ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])


def _case(case_id, policy, scramble, state, width, max_depth, no_backtrack, res, ref, draws=None, root=None):
    out = {"id": case_id, "policy": policy, "scramble": [int(m) for m in scramble],
           "state": [int(c) for c in state], "width": width, "max_depth": max_depth, "no_backtrack": no_backtrack,
           "moves": res.moves, "nodes": res.nodes_expanded, "depth": res.depth_reached,
           "margin": _margin(ref[3]), "solved_children": ref[4]}
    if draws is not None:
        out["draws"] = draws
    if root is not None:
        out["root_logits"] = [_f9(v) for v in root]
    return out


def _agree(res, ref) -> bool:
    return (res.moves, res.nodes_expanded, res.depth_reached) == tuple(ref[:3])


def generate() -> dict:
    """Build the fixture: logic cases, then the ONNX cases (needs onnxruntime and models/policy-v2.onnx)."""
    ort, session = _onnx_session()
    raw = onnx_raw(session)
    onnx_policy = torch_policy(raw, normalize=True)
    logic_policy = torch_policy(logic_values, normalize=False)
    cases = []

    rng = np.random.default_rng(SEED + 1)
    for k, (d, width, max_depth, no_backtrack) in enumerate(LOGIC_CASES, start=1):
        if d == 0:
            scramble, state = [], _SOLVED.astype(np.uint8)
        else:
            states, moves, _ = sampling.random_scrambles(rng, 1, d, depths=d)
            scramble, state = moves[0].tolist(), states[0]
        res = search.beam_search(logic_policy, state, width=width, max_depth=max_depth, no_backtrack=no_backtrack)
        ref = reference_search(logic_values, False, state, width, max_depth, no_backtrack)
        if not _agree(res, ref):
            raise AssertionError(f"logic case {k}: reference {ref[:3]} != beam_search {res}")
        cases.append(_case(f"logic-{k:02d}", "logic", scramble, state, width, max_depth, no_backtrack, res, ref))

    rng = np.random.default_rng(SEED)
    for i in range(N_ONNX):
        d = 5 + (i * 15) // (N_ONNX - 1)
        width = ONNX_WIDTHS[i % 2]
        for draw in range(1, MAX_DRAWS + 1):
            states, moves, _ = sampling.random_scrambles(rng, 1, d, depths=d)
            res = search.beam_search(onnx_policy, states[0], width=width, max_depth=MAX_DEPTH)
            ref = reference_search(raw, True, states[0], width, MAX_DEPTH)
            if not _agree(res, ref):
                raise AssertionError(f"onnx case {i + 1} draw {draw}: reference {ref[:3]} != beam_search {res}")
            if ref[3] >= MARGIN_MIN:
                break
        else:
            raise AssertionError(f"onnx case {i + 1}: no scramble with margin >= {MARGIN_MIN} in {MAX_DRAWS} draws")
        root = raw(states[0][None])[0]
        cases.append(_case(f"onnx-{i + 1:02d}", "onnx", moves[0].tolist(), states[0], width, MAX_DEPTH, True, res,
                           ref, draws=draw, root=root))

    return {
        "format": FORMAT,
        "version": 1,
        "generator": "tests/test_web_search_parity.py: rubiks.search.beam_search",
        "model": "models/policy-v2.onnx",
        "model_sha256": _sha256(MODEL),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__.split("+")[0],
        "seed": SEED,
        "margin_min": MARGIN_MIN,
        "log_softmax": "float64: max, sum of exp in move order, log; rounded to float32 (web/search.js logSoftmax)",
        "logic_policy": {"name": LOGIC_POLICY,
                         "definition": "for move m: child = state after m (child[i] = state[perm_m[i]]); value = "
                                       "-(4 * #{i: child[i] != solved[i]} + (sum_i child[i] * (i + 1)) mod 5) / 64"},
        "cases": cases,
    }


def write_fixture(fx: dict, path: Path = FIXTURE) -> None:
    """One case per line, so a regenerated fixture diffs case by case."""
    head = {k: v for k, v in fx.items() if k != "cases"}
    lines = ["{"]
    lines += [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    lines.append('  "cases": [')
    lines += [f"    {json.dumps(c)}{',' if n < len(fx['cases']) - 1 else ''}" for n, c in enumerate(fx["cases"])]
    lines += ["  ]", "}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


@pytest.fixture(scope="module")
def parity() -> dict:
    if not FIXTURE.is_file():
        write_fixture(generate())
    return json.loads(FIXTURE.read_text(encoding="ascii"))


def _cases(fx: dict, policy: str) -> list[dict]:
    return [c for c in fx["cases"] if c["policy"] == policy]


# --- tests --------------------------------------------------------------------------------------------------------

def test_fixture_layout(parity):
    assert parity["format"] == FORMAT and parity["logic_policy"]["name"] == LOGIC_POLICY
    onnx_cases, logic_cases = _cases(parity, "onnx"), _cases(parity, "logic")
    assert len(onnx_cases) == N_ONNX and len(logic_cases) == len(LOGIC_CASES)
    assert len({c["id"] for c in parity["cases"]}) == len(parity["cases"])
    assert {len(c["scramble"]) for c in onnx_cases} == set(range(5, 21))
    assert {c["width"] for c in onnx_cases} == set(ONNX_WIDTHS)
    for c in parity["cases"]:
        state = np.array(c["state"], dtype=np.uint8)
        assert np.array_equal(state, cube.apply_sequence(cube.SOLVED, c["scramble"])), c["id"]
        if c["moves"] is not None:
            assert cube.is_solved(cube.apply_sequence(state, c["moves"])), c["id"]
            assert len(c["moves"]) == c["depth"] <= c["max_depth"], c["id"]
    for c in onnx_cases:
        assert c["margin"] is None or c["margin"] >= MARGIN_MIN, c["id"]
        assert len(c["root_logits"]) == cube.N_MOVES, c["id"]
    assert len(FIXTURE.read_bytes()) < 100_000  # commit-sized


def test_logic_cases_reproduce(parity):
    """beam_search and the numpy form of web/search.js both reproduce every logic case (exact arithmetic)."""
    policy = torch_policy(logic_values, normalize=False)
    bad = []
    for c in _cases(parity, "logic"):
        res = search.beam_search(policy, c["state"], width=c["width"], max_depth=c["max_depth"],
                                 no_backtrack=c["no_backtrack"])
        ref = reference_search(logic_values, False, c["state"], c["width"], c["max_depth"], c["no_backtrack"])
        want = (c["moves"], c["nodes"], c["depth"])
        if (res.moves, res.nodes_expanded, res.depth_reached) != want or tuple(ref[:3]) != want:
            bad.append((c["id"], want, (res.moves, res.nodes_expanded, res.depth_reached), ref[:3]))
    assert not bad, bad


def test_logic_cases_cover_the_edges(parity):
    logic_cases = _cases(parity, "logic")
    assert any(c["moves"] == [] and c["nodes"] == 0 for c in logic_cases)            # solved input
    assert any(c["moves"] is None and c["max_depth"] == 0 for c in logic_cases)      # nothing expanded
    assert any(c["moves"] is None and c["max_depth"] > 0 for c in logic_cases)       # search exhausted
    assert any(c["width"] == 1 for c in logic_cases)                                 # a greedy walk
    assert any(not c["no_backtrack"] for c in logic_cases)
    assert any(c["margin"] == 0.0 for c in logic_cases)                              # exact ties were decided


def test_onnx_cases_reproduce(parity):
    """beam_search with the ONNX model (onnxruntime) reproduces every onnx case."""
    _, session = _onnx_session()
    assert _sha256(MODEL) == parity["model_sha256"], "models/policy-v2.onnx changed: regenerate the fixture"
    raw = onnx_raw(session)
    policy = torch_policy(raw, normalize=True)
    bad = []
    for c in _cases(parity, "onnx"):
        root = raw(np.array([c["state"]], dtype=np.uint8))[0]
        assert np.allclose(root, c["root_logits"], atol=1e-4), c["id"]
        res = search.beam_search(policy, c["state"], width=c["width"], max_depth=c["max_depth"])
        want = (c["moves"], c["nodes"], c["depth"])
        if (res.moves, res.nodes_expanded, res.depth_reached) != want:
            bad.append((c["id"], want, (res.moves, res.nodes_expanded, res.depth_reached), c["margin"]))
    assert not bad, bad


def test_reference_matches_onnx_cases(parity):
    """The numpy form of web/search.js gives the fixture's answers with the ONNX policy too."""
    _, session = _onnx_session()
    raw = onnx_raw(session)
    bad = []
    for c in _cases(parity, "onnx"):
        ref = reference_search(raw, True, c["state"], c["width"], c["max_depth"])
        if tuple(ref[:3]) != (c["moves"], c["nodes"], c["depth"]):
            bad.append((c["id"], ref[:3]))
    assert not bad, bad


def test_search_js_is_pure_and_solver_pins_cdnjs():
    """search.js uses no DOM and no ONNX Runtime; solver.js loads ONNX Runtime Web from cdnjs at a pinned version;
    the page loads search.js before solver.js."""
    js = SEARCH_JS.read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", js, flags=re.S)
    for pattern in (r"\bdocument\b", r"\bfetch\(", r"\bort\.", r"InferenceSession", r"localStorage", r"XMLHttpRequest"):
        assert not re.search(pattern, code), pattern
    solver = SOLVER_JS.read_text(encoding="utf-8")
    assert re.search(r"const ORT_VERSION = '\d+\.\d+\.\d+';", solver)
    assert "const ORT_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/';" in solver
    assert re.findall(r"https://[^'\" )]+", re.sub(r"/\*.*?\*/", "", solver, flags=re.S)) == \
        ["https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/"]  # the only external host of solver.js
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert html.index('<script src="search.js">') < html.index('<script src="solver.js">')
