"""rubiks.serve: the /solve backends (fake network solvers, the real NNSolver on a tiny checkpoint, kociemba as the
fallback), GET /solver, the CLI flags, the HTTP wiring and malformed request targets. CPU only; the tiny checkpoint
trains in about a second."""

import contextlib
import functools
import json
import socket
import sys
import threading
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

import numpy as np
import pytest

from rubiks import cube, serve

SCRAMBLE = cube.parse_moves("R U F' L2 D B'")
STATE = cube.apply_sequence(cube.SOLVED, SCRAMBLE)
SOLUTION = cube.inverse_sequence(SCRAMBLE)


def body(state) -> bytes:
    return json.dumps({"state": [int(x) for x in state]}).encode()


class FakeSolver:
    """A network solver stand-in: ``solve`` returns ``answer(state)`` and counts its calls."""

    source = "policy-beam"
    device = "cpu"

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def solve(self, state):
        self.calls += 1
        return self.answer(np.asarray(state))


def fake_backend(answer, fallback="none") -> serve.Backend:
    return serve.Backend("policy", solver=FakeSolver(answer), checkpoint="fake.pt", beam=8, max_depth=5,
                         device="cpu", fallback=fallback)


def no_solution(state):
    return {"error": "no solution found within 5 moves at beam width 8", "source": "policy-beam", "nodes": 37,
            "ms": 1.0}


def overwrites_its_input(new_state, answer):
    """A fake network answer that writes ``new_state`` into the array it is given, then returns ``answer``; the arrays
    it saw (after writing) are kept in ``.seen``."""
    def run(state):
        state[:] = new_state  # a read-only array (the request state itself) raises here
        run.seen.append(state)
        return answer
    run.seen = []
    return run


class StubKociemba:
    """An always-importable stand-in for the kociemba package, whether or not the real one is installed: records the
    facelet strings it is asked to solve and answers with the known solution of STATE."""

    def __init__(self):
        self.calls = []

    def solve(self, facelets):
        self.calls.append(facelets)
        return cube.format_moves(SOLUTION)


@pytest.fixture
def stub_kociemba(monkeypatch):
    """Installs a :class:`StubKociemba` as what ``serve._import_kociemba`` (the fallback resolver) returns."""
    stub = StubKociemba()
    monkeypatch.setattr(serve, "_import_kociemba", lambda: stub)
    return stub


@contextlib.contextmanager
def serving(backend, directory):
    """An in-process ThreadingHTTPServer on port 0 for ``backend``; yields (opener, base URL)."""
    handler = functools.partial(serve.Handler, directory=str(directory), backend=backend)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never a proxy for 127.0.0.1
    try:
        yield opener, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


# -- handle_solve with fake backends -------------------------------------------------------------------------------

def test_fake_success_passes_nodes_and_notation_through():
    b = fake_backend(lambda s: {"moves": list(SOLUTION), "notation": cube.format_moves(SOLUTION),
                                "source": "policy-beam", "nodes": 1234, "ms": 5.0})
    status, out = serve.handle_solve(body(STATE), b)
    assert status == 200, out
    assert set(out) == {"moves", "solver", "ms", "verified", "nodes", "notation"}
    assert out["moves"] == SOLUTION and out["verified"] is True and out["solver"] == "policy-beam"
    assert out["nodes"] == 1234 and out["notation"] == cube.format_moves(SOLUTION)
    assert isinstance(out["ms"], float) and out["ms"] >= 0
    json.dumps(out)  # the payload is JSON-serialisable as sent
    assert b.solver.calls == 1


def test_fake_without_nodes_or_notation_omits_them():
    b = fake_backend(lambda s: {"moves": list(SOLUTION), "source": "policy-beam"})
    status, out = serve.handle_solve(body(STATE), b)
    assert status == 200 and out["verified"] is True and "nodes" not in out and "notation" not in out


def test_no_solution_without_fallback_is_500_with_the_message():
    b = fake_backend(no_solution, fallback="none")
    status, out = serve.handle_solve(body(STATE), b)
    assert status == 500
    assert "no solution found within 5 moves at beam width 8" in out["error"] and "moves" not in out


def test_no_solution_with_fallback_uses_kociemba():
    pytest.importorskip("kociemba")
    b = fake_backend(no_solution, fallback="kociemba")
    status, out = serve.handle_solve(body(STATE), b)
    assert status == 200, out
    assert out["solver"] == "kociemba (fallback after policy-beam)" and out["verified"] is True
    assert cube.is_solved(cube.apply_sequence(STATE, out["moves"]))
    assert "nodes" not in out and "notation" not in out  # kociemba reports neither
    assert b.solver.calls == 1


def test_no_solution_with_fallback_but_kociemba_missing_is_500(monkeypatch):
    monkeypatch.setitem(sys.modules, "kociemba", None)  # makes ``import kociemba`` raise ImportError
    status, out = serve.handle_solve(body(STATE), fake_backend(no_solution, fallback="kociemba"))
    assert status == 500
    assert "no solution found within" in out["error"] and "not importable" in out["error"]


def test_counting_fallback_stub_is_reached_on_no_solution(stub_kociemba):
    """Positive control for the never-fall-back test below: the stub IS the fallback that _backend_solve resolves."""
    b = fake_backend(no_solution, fallback="kociemba")
    status, out = serve.handle_solve(body(STATE), b)
    assert status == 200, out
    assert out["solver"] == "kociemba (fallback after policy-beam)" and out["verified"] is True
    assert out["moves"] == SOLUTION and stub_kociemba.calls == [cube.to_facelet_string(STATE)]


def _raises_device_lost(state):
    raise RuntimeError("device lost")


@pytest.mark.parametrize("answer, message", [
    (lambda s: {"error": "internal: solution failed the cubie-model check", "source": "policy-beam"},
     "solver error: policy-beam: internal: solution failed the cubie-model check"),
    (lambda s: {"error": "search: no solution found within 5 moves", "source": "policy-beam"},  # prefix, not substring
     "solver error: policy-beam: search: no solution found within 5 moves"),
    (_raises_device_lost, "solver error: RuntimeError: device lost"),
    (lambda s: {"moves": "R U", "source": "policy-beam"}, "malformed moves"),
    (lambda s: ["not", "a", "dict"], "returned list, not a dict"),
], ids=["error", "error-mentioning-no-solution", "exception", "malformed-moves", "not-a-dict"])
def test_other_backend_errors_are_500_and_never_fall_back(stub_kociemba, answer, message):
    """With the fallback enabled and always available (the counting stub), only "no solution found within ..." may
    reach it; every other failure is a 500 and the fallback is never called."""
    b = fake_backend(answer, fallback="kociemba")
    status, out = serve.handle_solve(body(STATE), b)
    assert status == 500 and message in out["error"] and "moves" not in out, out
    assert stub_kociemba.calls == []
    assert b.solver.calls == 1


def test_backend_exception_is_500():
    def boom(state):
        raise RuntimeError("device lost")
    status, out = serve.handle_solve(body(STATE), fake_backend(boom))
    assert status == 500 and out["error"] == "solver error: RuntimeError: device lost"


def test_non_solving_moves_are_returned_with_verified_false():
    b = fake_backend(lambda s: {"moves": [0, 3], "notation": "U R", "source": "policy-beam", "nodes": 2})
    status, out = serve.handle_solve(body(STATE), b)
    assert status == 200
    assert out["moves"] == [0, 3] and out["verified"] is False and out["notation"] == "U R" and out["nodes"] == 2


@pytest.mark.parametrize("new_state, moves", [
    (cube.SOLVED, []),                                                   # wipes the request, claims nothing to do
    (cube.apply_move(cube.SOLVED, 0), [cube.inverse_move(0)]),           # swaps in an easier cube and solves that
], ids=["wiped-to-solved", "swapped-for-depth-1"])
@pytest.mark.parametrize("fallback", ["none", "kociemba"])
def test_backend_that_mutates_its_input_cannot_fake_verified(new_state, moves, fallback):
    answer = overwrites_its_input(new_state, {"moves": list(moves), "source": "policy-beam"})
    b = fake_backend(answer, fallback=fallback)
    status, out = serve.handle_solve(body(STATE), b)
    assert status == 200, out
    assert out["moves"] == moves and out["verified"] is False  # verified against the request, not the mutated copy
    assert len(answer.seen) == 1 and np.array_equal(answer.seen[0], new_state)  # the fake did write its input
    assert cube.is_solved(cube.apply_sequence(answer.seen[0], moves))  # and its moves do solve what it made of it


@pytest.mark.parametrize("which", ["stub", "real"])
def test_fallback_after_a_mutating_backend_solves_the_original_state(which, request, monkeypatch):
    if which == "stub":
        stub = request.getfixturevalue("stub_kociemba")
    else:
        pytest.importorskip("kociemba")
        stub = None
    given = []
    real_kociemba_solve = serve._kociemba_solve

    def spy(koc, state):
        given.append(np.array(state, copy=True))
        return real_kociemba_solve(koc, state)

    monkeypatch.setattr(serve, "_kociemba_solve", spy)
    answer = overwrites_its_input(cube.SOLVED, no_solution(None))
    b = fake_backend(answer, fallback="kociemba")
    status, out = serve.handle_solve(body(STATE), b)
    assert status == 200, out
    assert out["solver"] == "kociemba (fallback after policy-beam)" and out["verified"] is True
    assert len(given) == 1 and np.array_equal(given[0], STATE)  # the ORIGINAL state, not the wiped copy
    assert cube.is_solved(answer.seen[0])  # the network's copy was wiped
    if stub is not None:
        assert stub.calls == [cube.to_facelet_string(STATE)]


@pytest.mark.parametrize("moves", [[18], [-1], ["R"], [True], "R U", None, [1.0]])
def test_malformed_moves_are_500(moves):
    status, out = serve.handle_solve(body(STATE), fake_backend(lambda s: {"moves": moves, "source": "policy-beam"}))
    assert status == 500 and "malformed moves" in out["error"]


def test_numpy_integer_moves_are_accepted():
    b = fake_backend(lambda s: {"moves": [np.int64(m) for m in SOLUTION], "source": "policy-beam"})
    status, out = serve.handle_solve(body(STATE), b)
    assert status == 200 and out["moves"] == SOLUTION and all(type(m) is int for m in out["moves"])


def test_validation_and_solved_state_never_reach_the_backend():
    b = fake_backend(lambda s: pytest.fail("the backend must not be called"))
    assert serve.handle_solve(b"not json", b)[0] == 400
    twisted = np.array(cube.SOLVED).copy()
    twisted[[8, 9, 20]] = twisted[[9, 20, 8]]  # a twisted corner: unreachable
    assert serve.handle_solve(body(twisted), b)[0] == 400
    status, out = serve.handle_solve(body(cube.SOLVED), b)
    assert status == 200 and out["moves"] == [] and out["verified"] is True
    assert b.solver.calls == 0


NESTING_TOO_DEEP = (400, {"error": "body is not JSON: nesting too deep"})
DEEP_BODIES = {
    "array-20k": b"[" * 20_000,
    "state-array-60k": b'{"state": ' + b"[" * 60_000,
    "object-16k": b'{"":' * 16_000,
}


@pytest.mark.parametrize("raw", list(DEEP_BODIES.values()), ids=list(DEEP_BODIES))
def test_deeply_nested_body_is_400_and_never_reaches_the_backend(raw):
    assert len(raw) < serve.MAX_BODY_BYTES  # inside the body-size limit: only the nesting is wrong
    b = fake_backend(lambda s: {"moves": [], "source": "policy-beam"})
    assert serve.handle_solve(raw, b) == NESTING_TOO_DEEP
    assert b.solver.calls == 0


def test_recursion_error_while_decoding_is_400_on_any_interpreter(monkeypatch):
    """The same path without depending on where this interpreter's decoder gives up."""
    def deep(_text):
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array from a unicode string")

    monkeypatch.setattr(serve, "json", types.SimpleNamespace(loads=deep, JSONDecodeError=json.JSONDecodeError))
    b = fake_backend(lambda s: {"moves": [], "source": "policy-beam"})
    assert serve.handle_solve(body(STATE), b) == NESTING_TOO_DEEP
    assert b.solver.calls == 0


def test_http_deeply_nested_body_gets_a_json_400(tmp_path):
    b = fake_backend(lambda s: {"moves": [], "source": "policy-beam"})
    with serving(b, tmp_path) as (opener, base):
        req = urllib.request.Request(base + "/solve", data=DEEP_BODIES["array-20k"],
                                     headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            opener.open(req, timeout=5)
        with exc.value as err:
            assert err.code == 400 and err.headers["Content-Type"] == "application/json"
            assert json.loads(err.read()) == NESTING_TOO_DEEP[1]
        with opener.open(base + "/solver", timeout=5) as r:  # the server still answers
            assert r.status == 200 and json.loads(r.read()) == b.info()
    assert b.solver.calls == 0


def test_none_backend_is_501():
    assert serve.handle_solve(body(STATE), serve.Backend("none")) == (501, {"error": "no solver backend"})


def test_default_backend_unchanged_without_kociemba(monkeypatch):
    monkeypatch.setitem(sys.modules, "kociemba", None)
    assert serve.handle_solve(body(STATE)) == (501, {"error": "no solver backend"})
    assert serve.handle_solve(body(STATE), serve.Backend("kociemba"))[0] == 501
    assert serve.handle_solver()[1]["backend"] == "none"


def test_default_backend_is_kociemba_when_importable():
    pytest.importorskip("kociemba")
    status, out = serve.handle_solve(body(STATE))
    assert status == 200 and out["solver"] == "kociemba" and out["verified"] is True
    assert set(out) == {"moves", "solver", "ms", "verified"}  # the kociemba answer is as before


# -- the real NNSolver on a tiny checkpoint ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """A tiny policy checkpoint trained in-process (the training CLI is not part of this repository): MLP
    324 -> 64 -> 32 -> 18 on 3,000 random scrambles of depth 1..3 with the training label (the inverse of the last
    scramble move), a few seeded AdamW epochs on CPU, saved with rubiks.model.save_checkpoint like a run's best.pt."""
    import torch

    from rubiks import data, model, sampling

    states, moves, depth = sampling.random_scrambles(np.random.default_rng(7), 3000, 3)
    last_move = moves[np.arange(len(depth)), depth.astype(np.int64) - 1]
    x = data.encode(torch.from_numpy(states))
    y = torch.from_numpy(data.undo_labels(last_move))
    torch.manual_seed(0)
    net = model.policy_mlp(hidden=(64, 32))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    sampler = data.BatchSampler(len(y), 256, seed=0)
    net.train()
    losses = []
    for _ in range(4):
        for idx in sampler:
            i = torch.from_numpy(idx)
            loss = loss_fn(net(x[i]), y[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    assert losses[-1] < losses[0]  # it trained
    net.eval()
    path = tmp_path_factory.mktemp("serve-run") / "best.pt"
    model.save_checkpoint(path, net, {"task": "policy", "n_train": 3000, "max_depth": 3}, epoch=4)
    return path


def test_real_policy_backend_solves_depth_one(tiny_checkpoint, tmp_path):
    args = serve.parse_args(["--web", str(tmp_path), "--backend", "policy", "--checkpoint", str(tiny_checkpoint),
                             "--beam", "16", "--max-depth", "4", "--device", "cpu", "--fallback", "none"])
    backend = serve.build_backend(args)
    assert backend.info() == {"backend": "policy", "checkpoint": str(tiny_checkpoint), "beam": 16, "max_depth": 4,
                              "device": "cpu", "fallback": "none"}
    assert "policy-beam" in backend.describe() and "beam 16" in backend.describe()
    for m in (0, 4, 17):
        state = cube.apply_move(cube.SOLVED, m)
        status, out = serve.handle_solve(body(state), backend)
        assert status == 200, out
        assert out["moves"] == [cube.inverse_move(m)] and out["verified"] is True
        assert out["solver"] == "policy-beam" and out["notation"] == cube.MOVE_NAMES[cube.inverse_move(m)]
        assert isinstance(out["nodes"], int) and out["nodes"] >= 1


def test_auto_with_checkpoint_is_policy(tiny_checkpoint, tmp_path, monkeypatch):
    args = serve.parse_args(["--web", str(tmp_path), "--checkpoint", str(tiny_checkpoint), "--device", "cpu"])
    assert serve.resolve_backend_name(args) == "policy"
    monkeypatch.setitem(sys.modules, "kociemba", None)
    backend = serve.build_backend(args)  # --fallback kociemba requested, package missing: the live fallback is none
    assert backend.name == "policy" and backend.fallback == "none" and backend.beam == 1024


def test_unloadable_checkpoint_fails_at_startup(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    args = serve.parse_args(["--web", str(tmp_path), "--backend", "policy", "--checkpoint", str(bad),
                             "--device", "cpu"])
    with pytest.raises(serve.BackendStartupError, match="cannot load"):
        serve.build_backend(args)


def test_value_backend_missing_module_fails_at_startup(tmp_path, monkeypatch, capsys):
    ckpt = tmp_path / "v.pt"
    ckpt.write_bytes(b"unused")
    monkeypatch.setitem(sys.modules, "rubiks.solver_value", None)  # the module is absent (it lands on main later)
    argv = ["--web", str(tmp_path), "--backend", "value", "--checkpoint", str(ckpt), "--port", "0"]
    with pytest.raises(serve.BackendStartupError, match="rubiks.solver_value"):
        serve.build_backend(serve.parse_args(argv))
    assert serve.main(argv) == 2  # exits before binding a port
    assert "--backend value needs rubiks.solver_value.ValueSolver" in capsys.readouterr().err


# -- parse_args -----------------------------------------------------------------------------------------------------

def test_parse_args_defaults(tmp_path):
    a = serve.parse_args(["--web", str(tmp_path)])
    assert (a.backend, a.checkpoint, a.beam, a.max_depth, a.device, a.fallback) == \
        ("auto", None, 1024, 40, "auto", "kociemba")
    assert (a.port, a.host) == (8765, "127.0.0.1")


@pytest.mark.parametrize("extra", [
    ["--backend", "policy"],                        # a network backend needs --checkpoint
    ["--backend", "value"],
    ["--beam", "0"],
    ["--max-depth", "0"],
    ["--backend", "bogus"],
    ["--fallback", "bogus"],
    ["--checkpoint", "/nonexistent/best.pt"],
])
def test_parse_args_errors(tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        serve.parse_args(["--web", str(tmp_path), *extra])
    assert exc.value.code == 2


def test_parse_args_checkpoint_with_a_non_network_backend_errors(tmp_path):
    ckpt = tmp_path / "x.pt"
    ckpt.write_bytes(b"x")
    for name in ("kociemba", "none"):
        with pytest.raises(SystemExit):
            serve.parse_args(["--web", str(tmp_path), "--backend", name, "--checkpoint", str(ckpt)])


def test_auto_without_checkpoint(tmp_path, monkeypatch):
    args = serve.parse_args(["--web", str(tmp_path)])
    monkeypatch.setitem(sys.modules, "kociemba", None)
    assert serve.resolve_backend_name(args) == "none"
    assert serve.build_backend(args).info() == {"backend": "none", "checkpoint": None, "beam": None,
                                                 "max_depth": None, "device": None, "fallback": "none"}
    with pytest.raises(serve.BackendStartupError, match="kociemba package cannot be imported"):
        serve.build_backend(serve.parse_args(["--web", str(tmp_path), "--backend", "kociemba"]))


# -- GET /solver ----------------------------------------------------------------------------------------------------

def test_handle_solver_pure():
    b = fake_backend(no_solution, fallback="kociemba")
    assert serve.handle_solver(b) == (200, {"backend": "policy", "checkpoint": "fake.pt", "beam": 8, "max_depth": 5,
                                            "device": "cpu", "fallback": "kociemba"})


def test_http_get_solver_and_post_solve(tmp_path):
    (tmp_path / "index.html").write_text("<p>hi</p>")
    b = fake_backend(lambda s: {"moves": list(SOLUTION), "notation": cube.format_moves(SOLUTION),
                                "source": "policy-beam", "nodes": 99})
    with serving(b, tmp_path) as (opener, base):
        with opener.open(base + "/solver", timeout=5) as r:
            assert r.status == 200 and r.headers["Access-Control-Allow-Origin"] == "*"
            assert r.headers["Content-Type"] == "application/json"
            assert json.loads(r.read()) == b.info()
        req = urllib.request.Request(base + "/solve", data=body(STATE), headers={"Content-Type": "application/json"},
                                     method="POST")
        with opener.open(req, timeout=5) as r:
            out = json.loads(r.read())
        assert out["moves"] == SOLUTION and out["verified"] is True and out["nodes"] == 99
        assert out["notation"] == cube.format_moves(SOLUTION) and out["solver"] == "policy-beam"
        with opener.open(base + "/", timeout=5) as r:  # static serving is unchanged
            assert r.read() == b"<p>hi</p>"
    assert b.solver.calls == 1


# -- malformed request targets: a 400, never a request that ends without an answer ---------------------------------

def raw_request(base: str, request: bytes) -> tuple[int, dict, bytes]:
    """Send ``request`` as raw bytes on a fresh connection and read the answer to EOF (the server speaks HTTP/1.0 and
    closes after one request). Returns (status, headers with lower-case names, body). A connection that closes without
    an HTTP response fails here; one that hangs times out after 5 s."""
    host, port = base.removeprefix("http://").rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.sendall(request)
        chunks = []
        while chunk := sock.recv(65536):
            chunks.append(chunk)
    raw = b"".join(chunks)
    assert raw, "no answer: the connection closed without an HTTP response"
    head, sep, payload = raw.partition(b"\r\n\r\n")
    assert sep, raw
    status_line, *lines = head.decode("latin-1").split("\r\n")
    headers = {}
    for line in lines:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return int(status_line.split()[1]), headers, payload


BAD_TARGETS = ["http://[", "http://[/solve", "http://[::1", "http://[abc]/solver"]


@pytest.mark.parametrize("target", BAD_TARGETS)
def test_split_target_is_none_where_urlsplit_raises(target):
    with pytest.raises(ValueError):
        urlsplit(target)  # positive control: this target really is one urlsplit rejects
    assert serve.split_target(target) is None
    assert serve.BAD_TARGET == (400, {"error": "bad request target"})  # what the handler answers for None


@pytest.mark.parametrize("target", ["/solver", "/solve", "/", "/a%20b/?q=1#f", "http://x/solver", "//x/y"])
def test_split_target_of_a_valid_target_is_urlsplit(target):
    assert serve.split_target(target) == urlsplit(target)


SOLVE_BODY = body(STATE)
BAD_REQUESTS = {
    "get": b"GET http://[ HTTP/1.1\r\nHost: x\r\n\r\n",
    "get-bracketed-host": b"GET http://[abc]/solver HTTP/1.1\r\nHost: x\r\n\r\n",
    "head": b"HEAD http://[ HTTP/1.1\r\nHost: x\r\n\r\n",
    "post-solve": (b"POST http://[/solve HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                   b"Content-Length: %d\r\n\r\n" % len(SOLVE_BODY) + SOLVE_BODY),
}


@pytest.mark.parametrize("name", list(BAD_REQUESTS))
def test_http_malformed_target_gets_a_json_400_and_the_server_keeps_answering(tmp_path, name):
    b = fake_backend(lambda s: {"moves": list(SOLUTION), "source": "policy-beam"})
    expected = json.dumps({"error": "bad request target"}).encode()
    with serving(b, tmp_path) as (opener, base):
        status, headers, payload = raw_request(base, BAD_REQUESTS[name])
        assert status == 400, (headers, payload)
        assert headers["content-type"] == "application/json" and headers["access-control-allow-origin"] == "*"
        assert headers["content-length"] == str(len(expected))
        assert payload == (b"" if name == "head" else expected)  # a HEAD answer carries the headers only
        with opener.open(base + "/solver", timeout=5) as r:  # the server still answers
            assert r.status == 200 and json.loads(r.read()) == b.info()
        req = urllib.request.Request(base + "/solve", data=SOLVE_BODY, headers={"Content-Type": "application/json"},
                                     method="POST")
        with opener.open(req, timeout=5) as r:  # and still solves
            out = json.loads(r.read())
        assert out["moves"] == SOLUTION and out["verified"] is True
    assert b.solver.calls == 1  # the malformed POST never reached the backend; the valid one did


def test_http_valid_targets_are_routed_as_before(tmp_path):
    (tmp_path / "index.html").write_text("<p>hi</p>")
    (tmp_path / "sub").mkdir()
    b = fake_backend(lambda s: {"moves": list(SOLUTION), "source": "policy-beam"})
    with serving(b, tmp_path) as (_opener, base):
        status, headers, payload = raw_request(base, b"GET http://x/solver HTTP/1.1\r\nHost: x\r\n\r\n")
        assert status == 200 and json.loads(payload) == b.info()  # an absolute-form target, routed by its path
        status, headers, payload = raw_request(base, b"HEAD /index.html HTTP/1.1\r\nHost: x\r\n\r\n")
        assert status == 200 and payload == b"" and headers["content-length"] == "9"
        status, headers, payload = raw_request(base, b"GET /index.html HTTP/1.1\r\nHost: x\r\n\r\n")
        assert status == 200 and payload == b"<p>hi</p>"
        status, headers, payload = raw_request(base, b"GET /sub?q=1 HTTP/1.1\r\nHost: x\r\n\r\n")
        assert status == 301 and headers["location"] == "/sub/?q=1"  # the redirect keeps the query, as before
        status, headers, payload = raw_request(base, b"GET /nope HTTP/1.1\r\nHost: x\r\n\r\n")
        assert status == 404
    assert b.solver.calls == 0


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_http_nul_byte_in_a_static_path_is_404_and_the_server_keeps_answering(tmp_path, method):
    """``%00`` decodes to a NUL byte, on which Path.resolve() raises ValueError: a 404, like a path outside the
    root."""
    (tmp_path / "index.html").write_text("<p>hi</p>")
    b = fake_backend(lambda s: {"moves": [], "source": "policy-beam"})
    with serving(b, tmp_path) as (opener, base):
        for target in (b"/a%00b", b"/index.html%00"):
            request = method.encode() + b" " + target + b" HTTP/1.1\r\nHost: x\r\n\r\n"
            status, headers, _ = raw_request(base, request)
            assert status == 404 and headers["access-control-allow-origin"] == "*"
        with opener.open(base + "/solver", timeout=5) as r:
            assert r.status == 200 and json.loads(r.read()) == b.info()
