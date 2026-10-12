"""Dev server for the web page: ``python -m rubiks.serve --port 8765 --web web``.

Standard library plus numpy; a network backend also needs torch, imported only when one is selected.

* ``GET /...``     serves the ``--web`` directory statically (``/`` -> ``index.html``). Every target is resolved with
  symlinks followed and must lie inside the resolved ``--web`` root (404 otherwise); no directory listings.
* ``GET /solver``  what answers ``/solve``: ``{"backend", "checkpoint", "beam", "max_depth", "device", "fallback"}``
  (null where it does not apply).
* ``POST /solve``  body ``{"state": [54 ints]}`` -> ``{"moves": [...], "solver": ..., "ms": ..., "verified": ...}``
  (move indices, the order of ``rubiks.cube.MOVE_NAMES``), plus ``"nodes"`` (int) and ``"notation"`` (str) when the
  backend reports them, or ``{"error": ...}`` with status 400 / 413 / 500 / 501.
* ``OPTIONS``      answers CORS preflights, so a page opened as ``file://`` can call ``/solve``.
* A GET / HEAD / POST whose request target ``urllib.parse.urlsplit`` cannot parse (e.g. ``GET http://[``) is answered
  400 ``{"error": "bad request target"}`` (:func:`split_target`); valid targets are routed as before.

The state is validated with ``rubiks.cubie.from_facelets(...).is_valid()``, then solved by the backend chosen with
``--backend`` and built once at start-up (:func:`build_backend`; a checkpoint is loaded then, never per request):

* ``policy``   the trained policy network with beam search (``rubiks.solver_nn.NNSolver``; ``--checkpoint``,
  ``--beam``, ``--max-depth``, ``--device``);
* ``value``    the cost-to-go network (``rubiks.solver_value.ValueSolver``, same ``solve()`` contract); start-up fails
  with a clear message while that module is not in the checkout;
* ``kociemba`` the ``kociemba`` package (an optional dependency, deliberately not in pyproject);
* ``none``     ``/solve`` answers 501 ``{"error": "no solver backend"}``;
* ``auto``     (default) ``policy`` when ``--checkpoint`` is given, else ``kociemba`` when importable, else ``none``.

Fallback rule: when a network backend answers ``{"error": "no solution found within ..."}``, ``--fallback kociemba``
(default) is set and the package imports, kociemba solves the state and the answer is named
``"kociemba (fallback after policy-beam)"``. Any other backend error is a 500 carrying the message. Whatever the
backend, the server applies the returned moves itself (``rubiks.cube``) to the request state as received (every
solver, the fallback included, works on its own copy) and reports the result as ``"verified"``; a non-solving answer
is returned with ``"verified": false``, never hidden. :func:`_backend_solve` is the plug-in point.
One log line per request. Binds 127.0.0.1 by default; ``--host 0.0.0.0`` exposes it to the LAN (phone).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import numpy as np

from . import cube, cubie

MAX_BODY_BYTES = 64 * 1024
_SOLVER_LOCK = threading.Lock()  # one backend call at a time (the kociemba C extension is not known thread-safe)

BACKEND_CHOICES = ("auto", "policy", "value", "kociemba", "none")
FALLBACK_CHOICES = ("kociemba", "none")
NO_SOLUTION_PREFIX = "no solution found within"  # the network solvers' "search exhausted" error
BAD_TARGET = (400, {"error": "bad request target"})  # the answer to a request target split_target cannot parse


class _NoBackend(Exception):
    pass


class _BackendError(Exception):
    """A backend answered ``{"error": ...}`` that is not covered by the fallback: reported as HTTP 500."""


class BackendStartupError(RuntimeError):
    """The backend chosen on the command line cannot be built; main() prints the message and exits 2."""


@dataclass(frozen=True)
class Backend:
    """What answers POST /solve. ``solver`` is the network solver (``.solve(state) -> dict``, see
    ``rubiks.solver_nn``) for ``policy`` / ``value`` and None for ``kociemba`` / ``none``. The other fields are what
    GET /solver reports; ``fallback`` is the fallback that is live (``"kociemba"`` or ``"none"``)."""

    name: str
    solver: Any = None
    checkpoint: str | None = None
    beam: int | None = None
    max_depth: int | None = None
    device: str | None = None
    fallback: str = "none"

    def info(self) -> dict:
        return {"backend": self.name, "checkpoint": self.checkpoint, "beam": self.beam,
                "max_depth": self.max_depth, "device": self.device, "fallback": self.fallback}

    def describe(self) -> str:
        """One line for the start-up banner."""
        if self.name == "none":
            return "none (POST /solve answers 501)"
        if self.solver is None:
            return self.name
        source = getattr(self.solver, "source", self.name)
        return (f"{self.name} ({source})  checkpoint {self.checkpoint}  beam {self.beam}  max depth {self.max_depth}  "
                f"device {self.device}  fallback {self.fallback}")


def _import_kociemba():
    """The optional ``kociemba`` module, or None when it cannot be imported."""
    try:
        import kociemba  # optional: installed in the dev venv only, never a declared dependency
    except ImportError:
        return None
    return kociemba


def _kociemba_solve(koc, state: np.ndarray) -> list[int]:
    with _SOLVER_LOCK:
        solution = koc.solve(cube.to_facelet_string(state))
    return cube.parse_moves(solution)


def default_backend() -> Backend:
    """The backend used when none is passed in (the behaviour before --backend): kociemba if importable, else none."""
    return Backend("kociemba") if _import_kociemba() is not None else Backend("none")


def _is_move_index(m) -> bool:
    return isinstance(m, (int, np.integer)) and not isinstance(m, bool) and 0 <= int(m) < cube.N_MOVES


def _backend_solve(state: np.ndarray, backend: Backend | None = None) -> tuple[list[int], str, dict]:
    """Solve a valid, unsolved state; returns (move indices, solver name, extra response fields). THE PLUG-IN POINT.

    ``extra`` holds ``"nodes"`` and ``"notation"`` when a network backend reports them. Raises _NoBackend (-> 501),
    _BackendError (-> 500) or whatever the backend raises (-> 500). The moves are NOT verified here; handle_solve does
    that and reports the result.

    Every call into a solver (the network backend, kociemba, the kociemba fallback) gets its own copy of ``state``,
    never ``state`` itself: a backend that writes into its input cannot change what the fallback solves, nor the
    state handle_solve verifies the moves against."""
    backend = default_backend() if backend is None else backend
    if backend.name == "kociemba":
        koc = _import_kociemba()
        if koc is None:
            raise _NoBackend
        return _kociemba_solve(koc, state.copy()), "kociemba", {}
    if backend.name == "none" or backend.solver is None:
        raise _NoBackend
    source = str(getattr(backend.solver, "source", backend.name))
    with _SOLVER_LOCK:
        result = backend.solver.solve(state.copy())  # a copy: the solver may write into the array it is given
    if not isinstance(result, dict):
        raise _BackendError(f"{source} returned {type(result).__name__}, not a dict")
    source = str(result.get("source") or source)
    if "error" in result:
        message = str(result["error"])
        if message.startswith(NO_SOLUTION_PREFIX) and backend.fallback == "kociemba":
            koc = _import_kociemba()
            if koc is None:
                raise _BackendError(f"{source}: {message} (fallback kociemba is not importable)")
            # the request state as received (a fresh copy), whatever the network did to the copy it was given
            return _kociemba_solve(koc, state.copy()), f"kociemba (fallback after {source})", {}
        raise _BackendError(f"{source}: {message}")
    moves = result.get("moves")
    if not isinstance(moves, (list, tuple)) or not all(_is_move_index(m) for m in moves):
        raise _BackendError(f"{source} returned malformed moves (need a list of move indices 0..{cube.N_MOVES - 1}): "
                            f"{str(moves)[:200]}")
    extra: dict = {}
    if result.get("nodes") is not None:
        extra["nodes"] = int(result["nodes"])
    if result.get("notation") is not None:
        extra["notation"] = str(result["notation"])
    return [int(m) for m in moves], source, extra


def _parse_state(body: bytes) -> np.ndarray:
    """Body bytes -> state (54,) uint8; raises ValueError with a message for the client."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"body is not JSON: {exc}") from None
    except RecursionError:  # deeply nested arrays/objects: far below MAX_BODY_BYTES, they exhaust the JSON decoder
        raise ValueError("body is not JSON: nesting too deep") from None
    if not isinstance(data, dict):
        raise ValueError('body must be a JSON object {"state": [54 ints]}')
    if "state" not in data:
        raise ValueError('missing "state" (a list of 54 ints in 0..5)')
    state = data["state"]
    if not isinstance(state, list) or len(state) != cube.N_FACELETS:
        raise ValueError(f'"state" must be a list of {cube.N_FACELETS} ints')
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in state):
        raise ValueError('"state" entries must be integers')
    if not all(0 <= x <= 5 for x in state):
        raise ValueError('"state" entries must be colours in 0..5')
    return np.array(state, dtype=np.uint8)


def handle_solve(body: bytes, backend: Backend | None = None) -> tuple[int, dict]:
    """The /solve endpoint as a pure function: request body -> (HTTP status, JSON payload).

    ``backend`` is what main() built; None means :func:`default_backend` (kociemba if importable, else 501)."""
    try:
        state = _parse_state(body)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    # The request state, read-only from here on: "verified" below is checked against exactly this array. It is never
    # handed to a backend (_backend_solve gives each solver its own copy), and the flag makes a regression loud.
    state.setflags(write=False)
    try:
        cc = cubie.from_facelets(state)
    except ValueError as exc:
        return 400, {"error": f"invalid state: {exc}"}
    if not cc.is_valid():
        return 400, {"error": "invalid state: not reachable (duplicated cubie, twisted corner, flipped edge "
                              "or parity)"}
    if cube.is_solved(state):
        return 200, {"moves": [], "solver": "none (already solved)", "ms": 0.0, "verified": True}
    t0 = time.perf_counter()
    try:
        moves, name, extra = _backend_solve(state, backend)
    except _NoBackend:
        return 501, {"error": "no solver backend"}
    except _BackendError as exc:
        return 500, {"error": f"solver error: {exc}"}
    except Exception as exc:  # noqa: BLE001 - any backend failure is reported to the client, not swallowed
        return 500, {"error": f"solver error: {type(exc).__name__}: {exc}"}
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    # the server's own check, whatever the backend, against the request state as received
    verified = cube.is_solved(cube.apply_sequence(state, moves))
    return 200, {"moves": [int(m) for m in moves], "solver": name, "ms": ms, "verified": verified, **extra}


def handle_solver(backend: Backend | None = None) -> tuple[int, dict]:
    """The GET /solver endpoint as a pure function: what answers POST /solve -> (HTTP status, JSON payload)."""
    return 200, (default_backend() if backend is None else backend).info()


def split_target(target: str) -> SplitResult | None:
    """The request target (``Handler.path``) split by ``urllib.parse.urlsplit``, or None when it cannot be parsed.

    urlsplit raises ValueError on some targets a client can send, e.g. ``http://[`` ("Invalid IPv6 URL") or
    ``http://[abc]/``; left uncaught that ends the request without an answer. The handler answers None with
    :data:`BAD_TARGET` (HTTP 400)."""
    try:
        return urlsplit(target)
    except ValueError:
        return None


class Handler(SimpleHTTPRequestHandler):
    """Static files from ``directory`` plus GET /solver and POST /solve, with CORS on every response.

    ``backend`` (keyword, bound with functools.partial in main()) is the :class:`Backend` built once at start-up."""

    server_version = "rubiks-serve/1.0"
    index_pages = ("index.html", "index.htm")  # own copy: send_head below must not depend on the stdlib version
    _log_extra = ""
    _target: SplitResult | None = None  # this request's target, split once by _parse_target

    def __init__(self, *args, backend: Backend | None = None, **kwargs) -> None:
        self.backend = backend  # before super().__init__, which handles the request
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")  # dev server: always serve the regenerated tables
        super().end_headers()

    # One line per request: log_request (called by send_response) writes it; send_error's extra line is dropped.
    def log_request(self, code="-", size="-") -> None:
        if isinstance(code, HTTPStatus):
            code = code.value
        self.log_message('"%s" %s%s', self.requestline, code, self._log_extra)
        self._log_extra = ""

    def log_error(self, format, *args) -> None:  # noqa: A002 - signature of BaseHTTPRequestHandler
        if format.startswith("code %d"):
            return
        super().log_error(format, *args)

    def log_message(self, format, *args) -> None:  # noqa: A002
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.address_string()} {format % args}\n")

    # -- static files: confined to the resolved web root ------------------------------------------------------
    def _inside_root(self, path) -> Path | None:
        """``path`` fully resolved (symlinks followed) if it lies inside the resolved web root, else None."""
        root = Path(self.directory).resolve()
        try:
            target = Path(path).resolve()
        except (OSError, RuntimeError, ValueError):  # e.g. a symlink loop; ValueError: a NUL byte (``%00``)
            return None
        return target if target.is_relative_to(root) else None

    def send_head(self):
        """GET / HEAD. Replaces SimpleHTTPRequestHandler.send_head, which follows symlinks out of the root:
        every target, including a directory's index file, is resolved and must lie inside the web root, and
        directories without an index page are not listed. Anything else is a 404."""
        # do_GET / do_HEAD split the target first (and answer 400 if they cannot); parse here only if called otherwise
        parts = self._target if self._target is not None else self._parse_target()
        if parts is None:
            return None
        url_path = parts.path
        requested = self.translate_path(self.path)
        target = self._inside_root(requested)
        if target is not None and target.is_dir():
            if not url_path.endswith("/"):  # redirect like the stdlib (and apache) do
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", urlunsplit((parts[0], parts[1], parts[2] + "/", parts[3], parts[4])))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            index = next((Path(requested, name) for name in self.index_pages
                          if os.path.lexists(Path(requested, name))), None)
            requested = index
            target = None if index is None else self._inside_root(index)
        elif url_path.endswith("/"):
            target = None  # a file asked for as a directory
        if target is None or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            f = open(target, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            fs = os.fstat(f.fileno())
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", self.guess_type(str(requested)))
            self.send_header("Content-Length", str(fs.st_size))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except BaseException:
            f.close()
            raise

    def list_directory(self, path):  # never reached via send_head above; refuse explicitly anyway
        self.send_error(HTTPStatus.NOT_FOUND, "File not found")
        return None

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":  # a HEAD answer carries the headers only
            self.wfile.write(data)

    def _parse_target(self) -> SplitResult | None:
        """Split this request's target once (:func:`split_target`) and keep it for send_head. None means it cannot be
        parsed and the 400 :data:`BAD_TARGET` has been sent; the caller then returns without answering again."""
        self._target = split_target(self.path)
        if self._target is None:
            status, payload = BAD_TARGET
            self._log_extra = f" {payload['error']}"
            self._send_json(status, dict(payload))
        return self._target

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server naming
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")  # Chrome: file:// or LAN page -> localhost
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parts = self._parse_target()
        if parts is None:
            return
        if parts.path == "/solver":
            status, payload = handle_solver(self.backend)
            self._send_json(status, payload)
            return
        super().do_GET()  # -> send_head, which reuses the split target

    def do_HEAD(self) -> None:  # noqa: N802
        if self._parse_target() is None:
            return
        super().do_HEAD()  # -> send_head, which reuses the split target

    def do_POST(self) -> None:  # noqa: N802
        parts = self._parse_target()
        if parts is None:
            return
        if parts.path != "/solve":
            self._send_json(404, {"error": f"no such endpoint: POST {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(411, {"error": "Content-Length required"})
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._send_json(413, {"error": f"body larger than {MAX_BODY_BYTES} bytes"})
            return
        t0 = time.perf_counter()
        status, payload = handle_solve(self.rfile.read(length), self.backend)
        ms = (time.perf_counter() - t0) * 1000.0
        if "moves" in payload:
            self._log_extra = (f" {len(payload['moves'])} moves {payload['solver']} {ms:.0f} ms"
                               + ("" if payload.get("verified") else " NOT VERIFIED (does not solve)"))
        else:
            self._log_extra = f" {payload.get('error', '')}"
        self._send_json(status, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m rubiks.serve", description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; 0.0.0.0 for the LAN)")
    p.add_argument("--web", type=Path, default=Path("web"), help="directory served statically (default web)")
    p.add_argument("--backend", choices=BACKEND_CHOICES, default="auto",
                   help="what answers POST /solve: policy (the trained policy network with beam search; needs "
                        "--checkpoint), value (the cost-to-go network; needs rubiks.solver_value and --checkpoint), "
                        "kociemba, none (501). Default auto: policy when --checkpoint is given, else kociemba when "
                        "importable, else none")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="network checkpoint (best.pt of a training run), loaded once at start-up")
    p.add_argument("--beam", type=int, default=1024, help="beam width of the network search (default 1024)")
    p.add_argument("--max-depth", type=int, default=40,
                   help="longest solution the network search tries, in moves (default 40)")
    p.add_argument("--device", default="auto",
                   help="network device: auto (cuda, else mps, else cpu) | cpu | mps | cuda[:i] (default auto)")
    p.add_argument("--fallback", choices=FALLBACK_CHOICES, default="kociemba",
                   help="when the network finds no solution within its limits: kociemba (default; used only if the "
                        "package imports) or none (the answer is a 500 with the network's message)")
    args = p.parse_args(argv)
    if not args.web.is_dir():
        p.error(f"--web {args.web} is not a directory")
    if args.backend in NETWORK_FACTORIES and args.checkpoint is None:
        p.error(f"--backend {args.backend} needs --checkpoint")
    if args.checkpoint is not None and args.backend in ("kociemba", "none"):
        p.error(f"--checkpoint is only used by a network backend (policy, value, auto), not --backend {args.backend}")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        p.error(f"--checkpoint {args.checkpoint} is not a file")
    if args.beam < 1:
        p.error("--beam must be >= 1")
    if args.max_depth < 1:
        p.error("--max-depth must be >= 1")
    return args


def _make_policy(args: argparse.Namespace):
    from .solver_nn import NNSolver  # imports torch: only when a network backend is selected

    return NNSolver(args.checkpoint, device=args.device, width=args.beam, max_depth=args.max_depth)


def _make_value(args: argparse.Namespace):
    try:
        from .solver_value import ValueSolver  # lands on main later; same solve() contract, source "value-astar"
    except ImportError as exc:
        raise BackendStartupError(
            f"--backend value needs rubiks.solver_value.ValueSolver, which cannot be imported ({exc}); the value "
            "method is not in this checkout yet, use --backend policy") from None
    # [assumed] the same constructor keywords as NNSolver; a mismatch fails here, at start-up, with the message
    return ValueSolver(args.checkpoint, device=args.device, width=args.beam, max_depth=args.max_depth)


NETWORK_FACTORIES = {"policy": _make_policy, "value": _make_value}  # name -> factory(args) -> solver with .solve()


def resolve_backend_name(args: argparse.Namespace) -> str:
    """``auto`` -> policy when --checkpoint is given, else kociemba when importable, else none."""
    if args.backend != "auto":
        return args.backend
    if args.checkpoint is not None:
        return "policy"
    return "kociemba" if _import_kociemba() is not None else "none"


def build_backend(args: argparse.Namespace) -> Backend:
    """Build the backend chosen on the command line, once (a network checkpoint is loaded here, never per request).
    Raises BackendStartupError with a message for whoever starts the server."""
    name = resolve_backend_name(args)
    if name == "none":
        return Backend("none")
    if name == "kociemba":
        if _import_kociemba() is None:
            raise BackendStartupError("--backend kociemba: the kociemba package cannot be imported (it is an "
                                      "optional dev dependency); use --backend policy or none")
        return Backend("kociemba")
    try:
        solver = NETWORK_FACTORIES[name](args)
    except BackendStartupError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported at start-up with the cause, never swallowed
        message = f"--backend {name}: cannot load {args.checkpoint}: {type(exc).__name__}: {exc}"
        raise BackendStartupError(message) from exc
    fallback = args.fallback
    if fallback == "kociemba" and _import_kociemba() is None:
        fallback = "none"  # what is live; main() says why in the banner
    device = getattr(solver, "device", None)
    return Backend(name, solver=solver, checkpoint=str(args.checkpoint), beam=args.beam, max_depth=args.max_depth,
                   device=str(device) if device is not None else args.device, fallback=fallback)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        backend = build_backend(args)
    except BackendStartupError as exc:
        print(f"rubiks.serve: {exc}", file=sys.stderr, flush=True)
        return 2
    handler = functools.partial(Handler, directory=str(args.web.resolve()), backend=backend)
    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        port = httpd.server_address[1]
        print(f"rubiks.serve: http://{args.host}:{port}/  static {args.web.resolve()}  "
              f"POST /solve backend: {backend.describe()}", file=sys.stderr, flush=True)
        if backend.solver is not None and args.fallback == "kociemba" and backend.fallback != "kociemba":
            print("rubiks.serve: --fallback kociemba requested but the kociemba package cannot be imported: no "
                  "fallback (a search that finds no solution answers 500)", file=sys.stderr, flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
