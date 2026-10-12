"""Parity cases for the in-browser beam search (``web/search.js``), checked here against ``rubiks/search.py``.

``tests/data/web-parity.json`` holds four families of cases. Every expected answer is what
``rubiks.search.beam_search`` returns when the policy is normalised the way the Python solver normalises it:
``torch.log_softmax`` over float32 logits, as in ``rubiks.search.make_policy_fn``. ``web/search.js`` has its own
log-softmax (float64, rounded to float32), so a difference between the two normalisations shows up as a failed case.

* ``onnx``: 20 seeded scrambles (depths 5 to 20, beam widths 64 and 256, max depth 40), solved with
  ``models/policy-v2.onnx`` run by onnxruntime.
* ``logic``: a synthetic policy with exact values (multiples of 1/64, so every float32 sum is exact) and many exact
  score ties, used as log-probabilities as they are. These cases pin down the tie-breaking order (state key, then
  child index), beam width 1, the max-depth edge cases and ``no_backtrack=False``.
* ``logits``: the same values used as logits, plus a common offset of 0, 1e6 or 1e30, on the logic scrambles. A
  log-softmax ignores a common offset in exact arithmetic; in float32, the offset 1e6 rounds the values to multiples of
  1/16, and 1e30 rounds them all to one value (a uniform policy, decided by the tie-breaking order alone).
* ``ties``: logits ``offset - scale * floor(misplaced / 8) + eps * h``: the best moves of a state are within
  ``4 * eps`` of each other and every other move is at least ``scale`` below them. With ``eps = 2**-12`` (offset 0,
  scale 256) they are near ties: each case is solved, and its closest nonzero decision gap lies between
  ``NEAR_FLOOR`` (above the float32 noise of either log-softmax) and ``NEAR_GAP`` (below the margin every ONNX case
  keeps). With the offset 1e30 (scale 2**80, eps 0) the best moves tie exactly and each gets -log(k) for k tied moves;
  a log-softmax computed as ``x - (max + log(sum))`` loses that term at this offset, and those cases are drawn so that
  such a normaliser answers each of them differently, so the fixture is known to catch it.

The synthetic families need no onnxruntime. When the fixture is missing, the ``parity`` fixture below writes it
(onnxruntime needed); otherwise the tests check that the Python search still reproduces every case.
``web/search-test.html`` runs the same cases through ``web/search.js`` in a browser and prints ``PARITY n/n PASS``.

Decision margins. Two ONNX Runtime builds (native here, WebAssembly in the browser) give logits that differ in the
last bits (about 1e-6), and torch's float32 log-softmax differs from the float64 one of ``web/search.js`` by a few
float32 ulps, so an ONNX case whose answer hinges on two scores closer than that could come out differently for a
numerical reason rather than a search one. :func:`reference_search` is the algorithm of ``web/search.js`` written in
numpy (the same candidate order, the same four-part state key, the same sort keys, the same log-softmax); it must
agree with ``beam_search`` on every case, and it also records the smallest score gap that any decision depended on
(the beam cut-off, the choice among duplicates of a kept state, the choice among solved children) and the smallest
such gap that is not zero. The ONNX scrambles come from one seeded stream; each case takes the next scramble whose
margin is at least ``MARGIN_MIN``, and the fixture records the margin and how many scrambles were drawn. The
synthetic families decide exact and near ties on purpose; their logits are exact float32 values, and every
log-probability they produce lies far from a float32 rounding boundary (``GUARD_ULPS``), so a browser's ``Math.exp``
and ``Math.log`` give the same float32 log-probabilities as numpy's.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import NamedTuple

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
VERSION = 2
SEED = 20261012
N_ONNX = 20
ONNX_WIDTHS = (64, 256)
MAX_DEPTH = 40
MARGIN_MIN = 1e-4
MAX_DRAWS = 200
# (scramble depth, width, max depth, no_backtrack); depth 0 is the solved cube
LOGIC_CASES = [
    (0, 64, 40, True),     # solved input: [] with 0 nodes
    (1, 1, 6, True),
    (3, 1, 8, True),       # beam width 1
    (3, 2, 8, True),
    (4, 7, 8, True),
    (4, 32, 8, False),     # only the same-face rule
    (5, 64, 10, True),
    (6, 256, 8, True),
    (8, 16, 12, True),
    (2, 3, 0, True),       # max depth 0: nothing is expanded
    (4, 4096, 5, True),    # a beam wide enough to keep every state of depth <= 3
]
LOGIT_OFFSETS = (0.0, 1e6, 1e30)
# (offset, scale, eps, whether a log-softmax computed as x - (max + log(sum)) must answer each case differently)
TIE_SETTINGS = ((0.0, 256.0, 2.0 ** -12, False), (1e30, 2.0 ** 80, 0.0, True))
N_TIES = 6                 # cases per tie setting
TIE_DEPTHS = (3, 4, 5, 6)
TIE_WIDTHS = (2, 3, 4, 6, 8, 16)
TIE_MAX_DEPTH = 10
NEAR_GAP = MARGIN_MIN      # near ties (eps > 0): the closest nonzero decision gap is below this ...
NEAR_FLOOR = 1e-5          # ... and not below this
MAX_TIE_DRAWS = 2000
GUARD_ULPS = 64            # synthetic log-probabilities: minimum distance from a float32 rounding boundary

_HASH_DEF = ("for move m: child = state after m (child[i] = state[perm_m[i]]); misplaced = #{i: child[i] != "
             "solved[i]}; h = (sum_i child[i] * (i + 1)) mod 5; computed in float64 in this order, rounded to float32")
POLICIES = {
    "logic": {"name": "misplaced-hash-v1", "normalize": False,
              "definition": _HASH_DEF + ": -(4 * misplaced + h) / 64, used as log-probabilities"},
    "logits": {"name": "misplaced-hash-v1-logits", "normalize": True,
               "definition": _HASH_DEF + ": offset + -(4 * misplaced + h) / 64, used as logits"},
    "ties": {"name": "misplaced-ties-v1", "normalize": True,
             "definition": _HASH_DEF + ": offset - scale * floor(misplaced / 8) + eps * h, used as logits"},
    "onnx": {"name": "models/policy-v2.onnx", "normalize": True,
             "definition": "the model's logits (input onehot, output logits), used as logits"},
}

_FACE = np.arange(cube.N_MOVES) // 3
_SOLVED = np.asarray(cube.SOLVED)
_NON_CENTRE = np.array([i for i in range(cube.N_FACELETS) if i not in set(cube.CENTRES.tolist())])
_POW12 = 6.0 ** np.arange(12)  # exact in float64


# --- normalisations --------------------------------------------------------------------------------------------

def log_softmax_torch(logits) -> torch.Tensor:
    """The Python solver's normalisation (rubiks.search.make_policy_fn): torch.log_softmax over float32 logits."""
    return torch.log_softmax(torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32)), dim=1)


def log_softmax_web(logits) -> np.ndarray:
    """web/search.js logSoftmax: in float64, (x - max) - log(sum of exp(x - max)), the exponentials summed in move
    order; the result rounded to float32."""
    z, log_sum = _web_parts(logits)
    return (z - log_sum).astype(np.float32)


def log_softmax_naive(logits) -> np.ndarray:
    """x - (max + log(sum of exp(x - max))) in float64, rounded to float32. Equal to the above in exact arithmetic,
    but max + log(sum) drops the log term when |max| is very large (the ties cases with the offset 1e30)."""
    x = np.asarray(logits, dtype=np.float32).astype(np.float64)
    mx = x.max(axis=1, keepdims=True)
    _, log_sum = _web_parts(logits)
    return (x - (mx + log_sum)).astype(np.float32)


def _web_parts(logits) -> tuple[np.ndarray, np.ndarray]:
    """(x - max, log(sum of exp(x - max)) as an (n, 1) column), float64, summed in move order as search.js does."""
    x = np.asarray(logits, dtype=np.float32).astype(np.float64)
    z = x - x.max(axis=1, keepdims=True)
    s = np.zeros(x.shape[0])
    for j in range(x.shape[1]):
        s += np.exp(z[:, j])
    return z, np.log(s)[:, None]


# --- policies --------------------------------------------------------------------------------------------------

def _misplaced_and_hash(states) -> tuple[np.ndarray, np.ndarray]:
    """(N, 18) int64 each, for child = the state after move m (child[i] = state[perm_m[i]])."""
    children = np.asarray(states)[:, cube.MOVE_PERMS]                  # (N, 18, 54)
    misplaced = (children != _SOLVED).sum(axis=2)
    h = (children.astype(np.int64) * np.arange(1, cube.N_FACELETS + 1)).sum(axis=2) % 5
    return misplaced, h


def logic_values(states) -> np.ndarray:
    """The ``logic`` policy (float32 (N, 18)): -(4 * misplaced + h) / 64, used as log-probabilities."""
    misplaced, h = _misplaced_and_hash(states)
    return (-(4 * misplaced + h) / 64.0).astype(np.float32)


def logits_policy(offset: float):
    """The ``logits`` policy: offset + -(4 * misplaced + h) / 64 in float64, rounded to float32."""
    def raw(states) -> np.ndarray:
        misplaced, h = _misplaced_and_hash(states)
        return (offset + -(4 * misplaced + h) / 64.0).astype(np.float32)
    return raw


def ties_policy(offset: float, scale: float, eps: float):
    """The ``ties`` policy: offset - scale * floor(misplaced / 8) + eps * h in float64, rounded to float32."""
    def raw(states) -> np.ndarray:
        misplaced, h = _misplaced_and_hash(states)
        return (offset - scale * (misplaced // 8) + eps * h).astype(np.float32)
    return raw


def onnx_raw(session):
    def raw(states) -> np.ndarray:
        return session.run(["logits"], {"onehot": encode_numpy(np.asarray(states))})[0]
    return raw


def case_policy(c: dict, onnx=None):
    """(raw function, normalize) of a fixture case."""
    p = c["policy"]
    if p == "logic":
        return logic_values, False
    if p == "logits":
        return logits_policy(c["offset"]), True
    if p == "ties":
        return ties_policy(c["offset"], c["scale"], c["eps"]), True
    if p == "onnx":
        return onnx, True
    raise ValueError(f"unknown policy {p!r}")


def torch_policy(raw_fn, normalize: bool):
    """A rubiks.search policy_fn over a numpy function of uint8 (N, 54) states: torch.log_softmax of its float32
    logits (normalize; as make_policy_fn does) or its float32 values as they are."""
    def policy_fn(states: torch.Tensor) -> torch.Tensor:
        out = np.ascontiguousarray(raw_fn(states.cpu().numpy()), dtype=np.float32)
        return log_softmax_torch(out) if normalize else torch.from_numpy(out)

    policy_fn.device = torch.device("cpu")  # type: ignore[attr-defined]
    return policy_fn


def python_search(raw_fn, normalize: bool, state, width: int, max_depth: int, no_backtrack: bool = True):
    """The expected answer: rubiks.search.beam_search with the Python solver's normalisation."""
    return search.beam_search(torch_policy(raw_fn, normalize), state, width=width, max_depth=max_depth,
                              no_backtrack=no_backtrack)


# --- the algorithm of web/search.js, in numpy ---------------------------------------------------------------------

class Ref(NamedTuple):
    moves: list[int] | None
    nodes: int
    depth: int
    margin: float           # the smallest score gap a decision depended on (inf when none)
    near: float             # the smallest such gap that is not zero (inf when none)
    solved_children: int


def _key_parts(children: np.ndarray) -> np.ndarray:
    """(n, 4) float64: k0 high, k0 low, k1 high, k1 low (web/search.js keyParts; rubiks.search.state_keys split)."""
    x = children[:, _NON_CENTRE].astype(np.float64)
    return np.stack([x[:, 12:24] @ _POW12, x[:, 0:12] @ _POW12, x[:, 36:48] @ _POW12, x[:, 24:36] @ _POW12], axis=1)


def reference_search(raw_fn, normalize: bool, state, width: int, max_depth: int, no_backtrack: bool = True,
                     log_softmax=log_softmax_web) -> Ref:
    """web/search.js beamSearch in numpy (with its log-softmax unless another is given)."""
    s0 = np.asarray(state, dtype=np.uint8)
    if np.array_equal(s0, _SOLVED):
        return Ref([], 0, 0, math.inf, math.inf, 0)
    beam = s0[None].copy()
    score = np.zeros(1, dtype=np.float32)
    last_face = np.full(1, -1)
    lock_axis = np.full(1, -1)
    hist_parent: list[np.ndarray] = []
    hist_move: list[np.ndarray] = []
    nodes = 0
    gaps = [math.inf]

    def note(values) -> None:
        gaps.extend(float(v) for v in np.atleast_1d(values) if np.isfinite(v))

    for depth in range(1, max_depth + 1):
        b = beam.shape[0]
        nodes += b
        out = raw_fn(beam)
        logp = log_softmax(out) if normalize else np.asarray(out, dtype=np.float32)
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
                note(top2[0] - top2[1])
            path = [int(move[j])]
            bi = int(parent[j])
            for k in range(len(hist_parent) - 1, -1, -1):
                path.append(int(hist_move[k][bi]))
                bi = int(hist_parent[k][bi])
            path.reverse()
            assert cube.is_solved(cube.apply_sequence(s0, path))
            return Ref(path, nodes, depth, min(gaps), min([g for g in gaps if g > 0]), int(solved_idx.size))
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
            note(ranked[width - 1] - ranked[width])
        starts = np.flatnonzero(first)
        nxt = starts + 1
        has_dup = nxt < len(order)
        has_dup[has_dup] = ~first[nxt[has_dup]]
        dup_gap = np.full(len(uniq), math.inf)
        dup_gap[has_dup] = (cscore[order[starts[has_dup]]].astype(np.float64)
                            - cscore[order[nxt[has_dup]]].astype(np.float64))
        note(dup_gap[kept])

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
    return Ref(None, nodes, max_depth, min(gaps), min([g for g in gaps if g > 0]), 0)


def boundary_distance_ulps(rows: list[np.ndarray]) -> float:
    """The smallest distance, in float64 ulps of the larger of |x - max|, |log(sum)| and the result, between a
    log-probability that web/search.js computes in float64 for these logit rows and a float32 rounding boundary."""
    z, log_sum = _web_parts(np.concatenate(rows))
    v = z - log_sum
    f = v.astype(np.float32)
    fd = f.astype(np.float64)
    up = np.nextafter(f, np.float32(np.inf)).astype(np.float64)
    down = np.nextafter(f, np.float32(-np.inf)).astype(np.float64)
    dist = np.minimum(np.abs(v - (fd + up) / 2), np.abs(v - (fd + down) / 2))
    size = np.maximum(np.maximum(np.abs(z), np.abs(np.broadcast_to(log_sum, z.shape))), np.abs(v))
    return float(np.min(dist / np.spacing(np.maximum(size, np.finfo(np.float64).tiny))))


# --- fixture ------------------------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f9(x: float) -> float:
    return float(f"{float(x):.9g}")


def _gap(x: float):
    return None if math.isinf(x) else float(f"{x:.3g}")


def _tag(x: float) -> str:
    return "0" if x == 0 else f"1e{round(math.log10(x))}"


def _onnx_session():
    ort = pytest.importorskip("onnxruntime")
    if not MODEL.is_file():
        pytest.skip(f"{MODEL} is missing")
    return ort, ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])


def _case(case_id, policy, scramble, state, width, max_depth, no_backtrack, res, ref: Ref, **extra) -> dict:
    out = {"id": case_id, "policy": policy, "scramble": [int(m) for m in scramble],
           "state": [int(c) for c in state], "width": width, "max_depth": max_depth, "no_backtrack": no_backtrack}
    out.update(extra)
    out.update({"moves": res.moves, "nodes": res.nodes_expanded, "depth": res.depth_reached,
                "margin": _gap(ref.margin), "near_gap": _gap(ref.near), "solved_children": ref.solved_children})
    return out


def _answer(res) -> tuple:
    return res.moves, res.nodes_expanded, res.depth_reached


def _check(what: str, res, ref: Ref) -> None:
    if _answer(res) != tuple(ref[:3]):
        raise AssertionError(f"{what}: web/search.js in numpy {tuple(ref[:3])} != beam_search {_answer(res)}")


def generate() -> dict:
    """Build the fixture (needs onnxruntime and models/policy-v2.onnx)."""
    ort, session = _onnx_session()
    raw = onnx_raw(session)
    cases = []

    # logic, and the same scrambles with the values as logits plus a common offset
    rng = np.random.default_rng(SEED + 1)
    logit_cases = []
    for k, (d, width, max_depth, no_backtrack) in enumerate(LOGIC_CASES, start=1):
        if d == 0:
            scramble, state = [], _SOLVED.astype(np.uint8)
        else:
            states, moves, _ = sampling.random_scrambles(rng, 1, d, depths=d)
            scramble, state = moves[0].tolist(), states[0]
        res = python_search(logic_values, False, state, width, max_depth, no_backtrack)
        ref = reference_search(logic_values, False, state, width, max_depth, no_backtrack)
        _check(f"logic case {k}", res, ref)
        cases.append(_case(f"logic-{k:02d}", "logic", scramble, state, width, max_depth, no_backtrack, res, ref))
        if d > 0 and max_depth > 0:  # the policy is called
            for offset in LOGIT_OFFSETS:
                policy = logits_policy(offset)
                res = python_search(policy, True, state, width, max_depth, no_backtrack)
                ref = reference_search(policy, True, state, width, max_depth, no_backtrack)
                _check(f"logits case {k} offset {offset:g}", res, ref)
                logit_cases.append(_case(f"logits-{k:02d}-{_tag(offset)}", "logits", scramble, state, width,
                                         max_depth, no_backtrack, res, ref, offset=offset))
    cases += logit_cases

    # ties: seeded draws until N_TIES solved cases with a near tie (eps > 0) or that a naive normaliser fails
    for t, (offset, scale, eps, naive_must_differ) in enumerate(TIE_SETTINGS):
        policy = ties_policy(offset, scale, eps)
        rng = np.random.default_rng(SEED + 2 + t)
        found = 0
        for draw in range(1, MAX_TIE_DRAWS + 1):
            d, width = int(rng.choice(TIE_DEPTHS)), int(rng.choice(TIE_WIDTHS))
            states, moves, _ = sampling.random_scrambles(rng, 1, d, depths=d)
            ref = reference_search(policy, True, states[0], width, TIE_MAX_DEPTH)
            if ref.moves is None or (eps > 0 and not NEAR_FLOOR <= ref.near < NEAR_GAP):
                continue
            naive = reference_search(policy, True, states[0], width, TIE_MAX_DEPTH, log_softmax=log_softmax_naive)
            naive_differs = tuple(naive[:3]) != tuple(ref[:3])
            if naive_must_differ and not naive_differs:
                continue
            res = python_search(policy, True, states[0], width, TIE_MAX_DEPTH)
            _check(f"ties offset {offset:g} draw {draw}", res, ref)
            found += 1
            cases.append(_case(f"ties-{_tag(offset)}-{found:02d}", "ties", moves[0].tolist(), states[0], width,
                               TIE_MAX_DEPTH, True, res, ref, offset=offset, scale=scale, eps=eps, draws=draw,
                               naive_differs=naive_differs))
            if found == N_TIES:
                break
        else:
            raise AssertionError(f"ties offset {offset:g}: only {found} of {N_TIES} cases in {MAX_TIE_DRAWS} draws")

    # onnx: the next scramble of the stream whose decisions all have a margin of at least MARGIN_MIN
    rng = np.random.default_rng(SEED)
    for i in range(N_ONNX):
        d = 5 + (i * 15) // (N_ONNX - 1)
        width = ONNX_WIDTHS[i % 2]
        for draw in range(1, MAX_DRAWS + 1):
            states, moves, _ = sampling.random_scrambles(rng, 1, d, depths=d)
            ref = reference_search(raw, True, states[0], width, MAX_DEPTH)
            if ref.margin >= MARGIN_MIN:
                break
        else:
            raise AssertionError(f"onnx case {i + 1}: no scramble with margin >= {MARGIN_MIN} in {MAX_DRAWS} draws")
        res = python_search(raw, True, states[0], width, MAX_DEPTH)
        _check(f"onnx case {i + 1}", res, ref)
        root = raw(states[0][None])[0]
        cases.append(_case(f"onnx-{i + 1:02d}", "onnx", moves[0].tolist(), states[0], width, MAX_DEPTH, True, res,
                           ref, draws=draw, root_logits=[_f9(v) for v in root]))

    return {
        "format": FORMAT,
        "version": VERSION,
        "generator": "tests/test_web_search_parity.py: rubiks.search.beam_search, policies normalised by "
                     "torch.log_softmax (rubiks.search.make_policy_fn)",
        "model": "models/policy-v2.onnx",
        "model_sha256": _sha256(MODEL),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__.split("+")[0],
        "seed": SEED,
        "margin_min": MARGIN_MIN,
        "near_gap_window": [NEAR_FLOOR, NEAR_GAP],
        "log_softmax": "expected answers: torch.log_softmax over float32 logits; web/search.js: float64 (x - max) - "
                       "log(sum of exp(x - max)), rounded to float32",
        "policies": POLICIES,
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


def _want(c: dict) -> tuple:
    return c["moves"], c["nodes"], c["depth"]


# --- tests --------------------------------------------------------------------------------------------------------

def test_fixture_layout(parity):
    assert parity["format"] == FORMAT and parity["version"] == VERSION and parity["policies"] == POLICIES
    assert {c["policy"] for c in parity["cases"]} == set(POLICIES)
    onnx_cases, logic_cases = _cases(parity, "onnx"), _cases(parity, "logic")
    logit_cases, tie_cases = _cases(parity, "logits"), _cases(parity, "ties")
    called = [k for k, (d, _, max_depth, _) in enumerate(LOGIC_CASES, start=1) if d > 0 and max_depth > 0]
    assert len(onnx_cases) == N_ONNX and len(logic_cases) == len(LOGIC_CASES)
    assert sorted(c["id"] for c in logit_cases) == sorted(f"logits-{k:02d}-{_tag(o)}" for k in called
                                                          for o in LOGIT_OFFSETS)
    assert len(tie_cases) == N_TIES * len(TIE_SETTINGS)
    assert sorted((c["offset"], c["scale"], c["eps"]) for c in tie_cases) == \
        sorted((o, s, e) for o, s, e, _ in TIE_SETTINGS for _ in range(N_TIES))
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
    bad = []
    for c in _cases(parity, "logic"):
        args = (c["state"], c["width"], c["max_depth"], c["no_backtrack"])
        res = python_search(logic_values, False, *args)
        ref = reference_search(logic_values, False, *args)
        if _answer(res) != _want(c) or tuple(ref[:3]) != _want(c):
            bad.append((c["id"], _want(c), _answer(res), ref[:3]))
    assert not bad, bad


def test_logic_cases_cover_the_edges(parity):
    logic_cases = _cases(parity, "logic")
    assert any(c["moves"] == [] and c["nodes"] == 0 for c in logic_cases)            # solved input
    assert any(c["moves"] is None and c["max_depth"] == 0 for c in logic_cases)      # nothing expanded
    assert any(c["moves"] is None and c["max_depth"] > 0 for c in logic_cases)       # search exhausted
    assert any(c["width"] == 1 for c in logic_cases)                                 # beam width 1
    assert any(not c["no_backtrack"] for c in logic_cases)
    assert any(c["margin"] == 0.0 for c in logic_cases)                              # exact ties were decided


def test_synthetic_logit_cases_reproduce(parity):
    """beam_search with torch.log_softmax (the Python solver's normalisation) and the numpy form of web/search.js
    (its own log-softmax) both reproduce every logits and ties case."""
    bad = []
    for c in _cases(parity, "logits") + _cases(parity, "ties"):
        raw, normalize = case_policy(c)
        args = (c["state"], c["width"], c["max_depth"], c["no_backtrack"])
        res = python_search(raw, normalize, *args)
        ref = reference_search(raw, normalize, *args)
        if _answer(res) != _want(c) or tuple(ref[:3]) != _want(c):
            bad.append((c["id"], _want(c), _answer(res), ref[:3]))
    assert not bad, bad


def test_offsets_cover_rounding_and_a_uniform_policy(parity):
    """The offset 1e30 rounds every logit of a row to one float32 value: the policy is uniform, and its answers differ
    from the offset-0 ones on some scrambles."""
    by_id = {c["id"]: c for c in _cases(parity, "logits")}
    called = sorted({c["id"][len("logits-"):len("logits-") + 2] for c in by_id.values()})
    assert any(_want(by_id[f"logits-{k}-1e30"]) != _want(by_id[f"logits-{k}-0"]) for k in called)
    row = logits_policy(1e30)(np.asarray(cube.SOLVED, dtype=np.uint8)[None])
    assert np.unique(row).size == 1


def test_ties_cases_are_near_ties_and_catch_a_naive_log_softmax(parity):
    """Every ties case is solved; with eps > 0 its closest nonzero decision gap lies in [NEAR_FLOOR, NEAR_GAP) (near
    ties); with the offset 1e30, a log-softmax computed as x - (max + log(sum)) answers every case differently, so the
    fixture catches it."""
    tie_cases = _cases(parity, "ties")
    assert all(c["moves"] is not None for c in tie_cases)
    near = [c for c in tie_cases if c["eps"] > 0]
    assert len(near) >= N_TIES
    for c in near:
        raw, _ = case_policy(c)
        ref = reference_search(raw, True, c["state"], c["width"], c["max_depth"])
        assert NEAR_FLOOR <= ref.near < NEAR_GAP and c["near_gap"] == _gap(ref.near), c["id"]
    caught = []
    for c in tie_cases:
        raw, _ = case_policy(c)
        naive = reference_search(raw, True, c["state"], c["width"], c["max_depth"], log_softmax=log_softmax_naive)
        if tuple(naive[:3]) != _want(c):
            caught.append(c["id"])
    assert caught == [c["id"] for c in tie_cases if c["naive_differs"]]
    must = {o for o, _, _, need in TIE_SETTINGS if need}
    assert [c["id"] for c in tie_cases if c["offset"] in must] == [c["id"] for c in tie_cases if c["naive_differs"]]
    assert len(caught) >= N_TIES


def test_synthetic_log_probabilities_are_far_from_float32_rounding(parity):
    """Every log-probability web/search.js computes for a logits or ties case lies at least GUARD_ULPS float64 ulps
    from a float32 rounding boundary, so a browser whose Math.exp and Math.log differ from numpy's in the last bits
    still gets the same float32 values (the cases decide exact and near ties on purpose)."""
    for c in _cases(parity, "logits") + _cases(parity, "ties"):
        raw, _ = case_policy(c)
        rows: list[np.ndarray] = []

        def recording(states, raw=raw, rows=rows):
            out = raw(states)
            rows.append(out)
            return out

        reference_search(recording, True, c["state"], c["width"], c["max_depth"], c["no_backtrack"])
        assert rows, c["id"]
        assert boundary_distance_ulps(rows) >= GUARD_ULPS, c["id"]


def test_web_log_softmax_is_close_to_torch(parity):
    """On the ONNX cases' root logits the two normalisations differ by a few float32 ulps at most, far below the
    MARGIN_MIN the ONNX cases keep; a large common offset leaves the float64 one unchanged."""
    x = np.array([c["root_logits"] for c in _cases(parity, "onnx")], dtype=np.float32)
    web, ref = log_softmax_web(x), log_softmax_torch(x).numpy()
    assert np.max(np.abs(web.astype(np.float64) - ref)) < MARGIN_MIN / 100
    tied = np.array([[0.0] * 3 + [-300.0] * 15, [-7.0] * 18], dtype=np.float32)
    for offset in (0.0, 1e6, 1e30):
        shifted = (tied.astype(np.float64) + offset).astype(np.float32)
        assert np.array_equal(log_softmax_web(shifted), log_softmax_torch(shifted).numpy()), offset
    big = np.array([[1e30] * 2 + [1e30 - 2.0 ** 80] * 16], dtype=np.float32)
    assert log_softmax_web(big)[0, 0] == np.float32(-math.log(2)) and log_softmax_naive(big)[0, 0] == 0.0


def test_onnx_cases_reproduce(parity):
    """beam_search with the ONNX model (onnxruntime) and torch.log_softmax reproduces every onnx case."""
    _, session = _onnx_session()
    assert _sha256(MODEL) == parity["model_sha256"], "models/policy-v2.onnx changed: regenerate the fixture"
    raw = onnx_raw(session)
    bad = []
    for c in _cases(parity, "onnx"):
        root = raw(np.array([c["state"]], dtype=np.uint8))[0]
        assert np.allclose(root, c["root_logits"], atol=1e-4), c["id"]
        res = python_search(raw, True, c["state"], c["width"], c["max_depth"])
        if _answer(res) != _want(c):
            bad.append((c["id"], _want(c), _answer(res), c["margin"]))
    assert not bad, bad


def test_reference_matches_onnx_cases(parity):
    """The numpy form of web/search.js (its own log-softmax) gives the fixture's answers with the ONNX policy too."""
    _, session = _onnx_session()
    raw = onnx_raw(session)
    bad = []
    for c in _cases(parity, "onnx"):
        ref = reference_search(raw, True, c["state"], c["width"], c["max_depth"])
        if tuple(ref[:3]) != _want(c):
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
