# The web page

The page shows an interactive 3D Rubik's cube that you can turn, scramble and solve. It works on a phone (touch, 360
px wide) and on a desktop. It is plain HTML and JavaScript with no build step. The user guide is in the main
[README](../README.md#using-the-app).

| file | what it is |
|---|---|
| `index.html` | the page and its CSS (light and dark themes) |
| `app.js` | the 3D view (three.js r128 from cdnjs), the 2D net, the move queue, the controls and the solver wiring |
| `solver.js` | cube logic driven only by the tables, and the solver interface with its backends |
| `cube-tables.js` | **generated** by `python -m rubiks.export_web`; never edit it by hand |

## How to open it

- **From the app (recommended).** Start the app (see the main README) and open `http://localhost:<port>/`. The
  Remote solver then calls `/solve` on the same address.
- **As a file.** Open `web/index.html` directly. The Remote solver then calls `http://localhost:8765/solve`, which
  works while the app runs on that port, because the server sends CORS headers.

The only external script is three.js from cdnjs. Offline, the page replaces the 3D view with a note, and the 2D net
and every control keep working.

## `cube-tables.js`

The page knows nothing about the cube except what this file tells it:

- the 18 move permutations;
- the 3D sticker geometry;
- the 2D net layout;
- the scramble filter;
- 1,000 uniformly random states (for **Random state**);
- 50 seeded test vectors.

Regenerate the file with:

```
python -m rubiks.export_web --out web/cube-tables.js --seed 20261012
```

The output is deterministic: the same seed, code and numpy version give a byte-identical file.

When the page loads, it checks itself and shows "tables verified 50/50 vectors, identity OK, orders OK". It checks
the test vectors and the 3D turn animation against the permutations; a mismatch shows a red banner.

## Solver backends (`solver.js`)

Each backend implements `async solve(state, context) -> {moves, source, ms}`. The `state` is 54 colours in Kociemba
facelet order, and `moves` are indices into `CUBE_TABLES.move_names`. The page never trusts a backend: it applies
the returned moves itself before it animates them.

1. **Reverse history** undoes the moves made on the page.
2. **Remote solver** sends `POST {"state": [54 ints]}` to `/solve` and gives up after 20 s.
3. **Neural network (ONNX, in the browser)** is a stub that is not implemented yet. It is meant to run
   `models/policy-v2.onnx` in the browser with ONNX Runtime Web.

To add another backend, push a `Solver` subclass onto the list in `createSolvers`. The browser console exposes
`window.rubiksApp` (`state`, `logic`, `solvers`, the load-time `report`, `enqueueMoves([...])`).
