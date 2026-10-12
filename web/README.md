# rubiks-nn web page

An interactive 3D Rubik's cube: turn it, scramble it, load a uniformly random state, and press **Solve** to ask a
pluggable solver for a move sequence, which the page verifies and then animates. Works on a phone (touch, 360 px
wide) and on a desktop, opened as a plain file (`file://`) or from a web server. No build step.

| file | what it is |
|---|---|
| `index.html` | the page and its CSS (light / dark theme as CSS custom properties) |
| `app.js` | 3D view (three.js r128 from cdnjs), 2D net, move queue, controls, solver wiring |
| `solver.js` | cube logic driven only by the tables, and the solver interface with its backends |
| `cube-tables.js` | **generated** by `python -m rubiks.export_web`; never edit it by hand |

The only external script is `https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js`. Without it
(offline) the page still works: the 3D view is replaced by a note, and the 2D net and all controls keep working.

## Regenerating `cube-tables.js`

```
python -m rubiks.export_web --out web/cube-tables.js --seed 20261012
```

The page knows nothing about the cube except what this file tells it. Everything comes from the Python model that
generates the training data (`rubiks/cube.py`, `rubiks/sampling.py`):

| key | content |
|---|---|
| `move_names`, `move_perms`, `inverse_move` | the 18 half-turn-metric moves; applying move `m` is `new[i] = old[move_perms[m][i]]` |
| `move_face`, `move_quarter_turns` | face of each move and its signed clockwise quarter turns (1, -1, 2), read from the names and checked against the permutations |
| `solved`, `centres`, `face_names` | solved state, centre facelets, `"URFDLB"` |
| `sticker_pos`, `sticker_normal`, `face_normal` | the 3D geometry of `rubiks.cube` (+x towards R, +y towards U, +z towards F): cubie position and outward normal of each facelet; the 3D view is built from these |
| `facelet_names`, `net_size`, `net_pos` | `U1` ... `B9`, and the cell of each facelet in the 2D net of `rubiks.cube.print_net` |
| `colours` | default colour per face: U white, R red, F green, D yellow, L orange, B blue |
| `scramble_filter` | the dataset generator's canonical filter, which the page's scrambler applies |
| `test_vectors` | 50 seeded random 30-move sequences with their final states, the identity `R L U2 R L' B2 U2 R2 F2 L2 D2 L2 F2`, and the orders of `R U` (105), `R U R' U'` (6), `R2 U2` (6) |
| `uniform_states` | 1,000 uniformly random reachable states (`rubiks.sampling.uniform_random_states`) for **Random state** |
| `generated` | seed, seed scheme, package and numpy versions |

The file is a classic script (`window.CUBE_TABLES = {...};`), not JSON fetched at run time, because `fetch()` of a
local file fails under `file://`. The body is strict JSON. The file has no timestamp: two runs with the same seed,
code and numpy version give byte-identical files, so `git diff web/cube-tables.js` after a regeneration shows only
real changes. Its provenance is the git commit of the file plus the seed and versions in `generated`.
`tests/test_export_web.py` checks the byte-identical rerun, the shapes, every test vector against `rubiks.cube`, the
validity of every uniform state with `rubiks.cubie`, and the 2D net against `print_net`.

At load the page re-checks the tables and shows **"tables verified 50/50 vectors, identity OK, orders OK"**; any
mismatch shows a red banner. It also checks that the 3D turn animation (rotating the layer about `face_normal` by
`-90 deg x move_quarter_turns`) lands every sticker exactly where `move_perms` puts it, for all 18 moves.

## Opening the page

- **As a file:** open `web/index.html` in a browser. Everything except the 3D view (three.js comes from cdnjs, so it
  needs the network or a cached copy) and the remote solver works offline. The remote solver works too if the dev
  server is running, because the server sends CORS headers.
- **From the dev server** (Python standard library only):

  ```
  python -m rubiks.serve --port 8765 --web web
  ```

  then open <http://localhost:8765/>. `POST /solve` is served on the same port. To use a phone on the same network,
  bind all interfaces with `--host 0.0.0.0` and open `http://<this computer's IP>:8765/` on the phone; the page
  then uses the same origin for `/solve`. Static files are confined to the `--web` directory: every requested path,
  including a directory's `index.html`, is resolved with symlinks followed and must lie inside the resolved web
  root, or the answer is 404; directories are never listed.
- **With the trained network answering Solve** (needs torch, which is imported only for a network backend):

  ```
  python -m rubiks.serve --backend policy --checkpoint runs/policy-v2/best.pt --beam 1024
  ```

  The checkpoint is loaded once at start-up, and the start-up line names the backend, checkpoint, beam width and
  device. `GET /solver` reports the same (below). Flags that choose the `/solve` backend:

  | flag | default | meaning |
  |---|---|---|
  | `--backend {auto,policy,value,kociemba,none}` | `auto` | `policy`: the policy network with beam search (`rubiks.solver_nn.NNSolver`). `value`: the cost-to-go network (`rubiks.solver_value`, not in the repo yet; start-up fails with a clear message until it lands). `kociemba`: the `kociemba` package (start-up fails if it cannot be imported). `none`: `/solve` answers 501. `auto`: `policy` when `--checkpoint` is given, else `kociemba` when importable, else `none` |
  | `--checkpoint PATH` | none | the network checkpoint (`best.pt` of a training run); required by `policy` and `value`, rejected with `kociemba` and `none` |
  | `--beam N` | 1024 | beam width of the network search, >= 1 |
  | `--max-depth N` | 40 | longest solution the search tries, in moves, >= 1 |
  | `--device NAME` | `auto` | `auto` (cuda, else mps, else cpu), `cpu`, `mps`, `cuda[:i]` |
  | `--fallback {kociemba,none}` | `kociemba` | what happens when the network finds no solution within `--max-depth` at `--beam` (see the fallback rule below) |

  Speed: policy v2 at beam 1024 solved 100% of 1,000 uniformly random states in about 90 ms each on MPS
  (`docs/RESULTS-POLICY-V2.md`); on CPU expect several seconds per solve at width 1024. The page's Remote solver
  gives up after 20 s, so pick a beam that answers well inside that on your device.

## Using it

- **Turn:** the 18 move buttons (one row of three per face), or the keyboard: `u r f d l b` turn a face
  clockwise, Shift + letter counter-clockwise, `2` then a letter turns it 180 degrees. Moves queue up and animate in
  order (speed: slow 400 ms, normal 180 ms, fast 60 ms per turn).
- **Scramble:** a random walk from solved with the generator's canonical filter, at the given depth (default 20). A
  seed makes it reproducible (mulberry32 PRNG); with an empty seed field a random seed is drawn and shown. JavaScript
  cannot reproduce numpy's generator, so the same seed gives a different sequence here than in
  `rubiks.generate`; the *filter* is the same.
- **Random state:** one of the 1,000 exported uniformly random states (picked with the seed, or at random).
- **Undo** reverts the last move; **Reset** returns to solved.
- **Type a move sequence** such as `R U R' U'` and press Apply (or Enter).
- **State:** the current state as the 54-character Kociemba facelet string (`UUUUUUUUURRRRRRRRRFFFFFFFFF...` when
  solved). **Copy** puts it on the clipboard; editing it and pressing **Load** sets the cube to that state. The page
  checks length, letters, centres and 9 stickers per colour; full reachability (twists, flips, parity) is checked by
  the server's `rubiks.cubie` on Solve.
- **2D net:** U on top, then L F R B, D below, each face in its own view, always in sync with the 3D view. Tick
  "show facelet indices" to see the Kociemba index (0..53) on every sticker; hover a sticker for its name.
- **Solve:** asks the selected backend. While it works or while a solution plays, the controls are locked and
  **Stop** cancels. The status line shows the last result (source, number of moves, time) and whether it solves.

## The solver interface (`solver.js`)

```js
class Solver {
  async solve(state /* Uint8Array(54) */, context) -> { moves: number[], source: string, ms: number }
}
// context = { history: number[] | null, logic, signal: AbortSignal | null }
```

- `state`: 54 colours 0..5 in Kociemba facelet order (a copy; the backend may keep it).
- `context.history`: the moves applied since the cube was last solved, or `null` when unknown (after
  Random state or Load). `context.logic` is the table-driven cube logic (`applyMove`, `applySequence`, `isSolved`,
  `inverseSequence`, `parseMoves`, ...). `context.signal` aborts when the user presses Stop.
- `moves`: indices into `CUBE_TABLES.move_names`. A backend that cannot solve **rejects** with an `Error`
  (`RubiksSolver.SolverError` with a `code`); the page shows the message, never silently.
- **The page never trusts a backend.** It applies the returned moves to the state with the tables before
  animating. A sequence that does not solve the cube is reported as *"solver returned a non-solving sequence
  (n moves)"*, and an **Animate anyway** button plays it. Values that are not move indices are rejected.

Backends (menu order, `RubiksSolver.createSolvers`):

1. **Reverse history** (`ReverseHistorySolver`): the inverse of the page's own moves since the last solved state.
   Solves anything scrambled or turned on the page; for a Random state or a loaded state it says it cannot.
2. **Remote solver** (`RemoteSolver(url)`): `POST {"state": [54 ints]}` to the URL (default
   `http://localhost:8765/solve` under `file://`, same-origin `/solve` when served over http). Expects
   `{"moves": [...]}` or `{"error": "..."}`; times out after 20 s. HTTP errors, network errors, timeouts and
   malformed answers are all shown in the page.
3. **Neural network** (`OnnxSolver`): a documented stub, not implemented. Selecting it shows "model not available
   yet".

### `POST /solve` (`rubiks/serve.py`)

Request `{"state": [54 ints in 0..5]}`. A body that is not JSON is a 400 (one nested too deeply for the decoder
answers `{"error": "body is not JSON: nesting too deep"}`). The state is validated with
`rubiks.cubie.from_facelets(...).is_valid()` (400 with the reason if not). A solved state returns
`{"moves": []}`. Otherwise the backend chosen with `--backend`
(see *Opening the page*) solves it:

- **`policy`** (the network): `{"moves": [...], "solver": "policy-beam", "ms": ..., "verified": true,
  "nodes": 5894, "notation": "F' R' U2 ..."}`. `nodes` (int) is the number of states the network scored during the
  search and `notation` (string) the solution in `rubiks.cube` notation; both are added only when the backend
  reports them.
- **`kociemba`**: its notation is converted with `rubiks.cube.parse_moves`; the answer is
  `{"moves": [...], "solver": "kociemba", "ms": ..., "verified": true}` as before (no `nodes`, no `notation`). The
  package is an optional dev dependency, not in `pyproject.toml`.
- **`none`** (or `kociemba` not importable when no backend was chosen): 501 `{"error": "no solver backend"}`.

`ms` is the server's wall time for the backend call. `verified` is the server's own check: it applies the returned
moves with `rubiks.cube.apply_sequence` to the request state as received and tests `is_solved`, whatever the backend.
Every solver, the kociemba fallback included, is handed its own copy of the state, so a backend that writes into its
input can change neither what is verified nor what the fallback solves. A sequence that does not solve
the state is still returned, with `"verified": false` (the page then reports it and offers **Animate anyway**); it
is never hidden. Malformed moves from a backend (not a list of indices 0..17) are a 500.

**Fallback rule.** When a network backend answers `{"error": "no solution found within ..."}` (the search ran out
of depth at its beam width), `--fallback kociemba` is set (the default) and the `kociemba` package imports, kociemba
solves the state and the answer is named `"solver": "kociemba (fallback after policy-beam)"`. With
`--fallback none`, or when the package cannot be imported, the answer is 500 with the network's message. Any other
backend error (for example an internal verification failure) is a 500 carrying the message and never falls back.

### `GET /solver`

What answers `/solve`, as JSON, for the page or a test:

```
{"backend": "policy", "checkpoint": "runs/policy-v2/best.pt", "beam": 1024, "max_depth": 40, "device": "mps",
 "fallback": "kociemba"}
```

`backend` is the resolved name (`policy`, `value`, `kociemba` or `none`, never `auto`); `checkpoint`, `beam`,
`max_depth` and `device` (the device actually in use) are `null` for `kociemba` and `none`. `fallback` is the
fallback that is live: `"none"` for `kociemba` and `none`, and also when `--fallback kociemba` was asked for but the
package cannot be imported (the start-up output says so).

Every response carries `Access-Control-Allow-Origin: *` and OPTIONS preflights are answered, so a `file://` page
can call it. The server logs one line per request; a `/solve` answer that does not solve the state is marked
`NOT VERIFIED` in that line.

## How the network will plug in

Two routes, both without changing the page's contract:

1. **Server side (done):** `python -m rubiks.serve --backend policy --checkpoint ...` answers `/solve` with the
   policy network and beam search behind `_backend_solve` in `rubiks/serve.py` (see *Opening the page* and
   `POST /solve`). The page keeps using the Remote solver backend and verifies the answer as before. The value
   network will be added behind the same `--backend` switch (`--backend value`).
2. **In the browser:** implement `OnnxSolver` in `solver.js` (the stub documents the plan):
   - export the trained network with `torch.onnx.export` to e.g. `web/model/rubiks-nn.onnx`, with a small JSON
     sidecar pinning the input encoding (the design fixes a 54-sticker one-hot, 324 features) and the output order
     (`CUBE_TABLES.move_names`);
   - load ONNX Runtime Web from cdnjs on first use (`https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/<pinned
     version>/ort.min.js`, injected as a script tag), then `ort.InferenceSession.create(modelUrl)`;
   - in `solve()`, encode the state exactly as in training, run the search with `context.logic.applyMove`, stop
     at `context.logic.isSolved` or a move limit, and return `{moves, source: 'onnx: <model>', ms}`;
   - make `available` return `true`.

   ONNX Runtime Web fetches its `.wasm` files and the model with `fetch()`, which fails under `file://`, so this
   backend needs the page served over http (`python -m rubiks.serve`), or the model embedded in a `.js` file.

A different backend can also be added by pushing another `Solver` subclass onto the list in `createSolvers`.

For the browser console: `window.rubiksApp` exposes `state`, `logic`, `solvers`, the load-time `report`, and
`enqueueMoves([...])`.
