# What was done

## The problem

The goal is to solve the 3x3x3 Rubik's Cube with a neural network that learns from random scrambles alone, with
no hand-written solving method. The cube has 43,252,003,274,489,856,000 reachable states. We count moves in the
half-turn metric, which has 18 moves: each of the six faces turns a quarter clockwise (`R`), a quarter
counter-clockwise (`R'`) or a half turn (`R2`). In this metric every state can be solved in at most 20 moves.

## The cube representation

A state is 54 colour ids (0 to 5), one per sticker ("facelet"), in Kociemba's order: faces U R F D L B, and
`facelet = face*9 + row*3 + col`. Each move is a permutation of the 54 positions.

Two independent models of the cube were written and checked against each other:

- **Facelet model (`rubiks/cube.py`).** The six quarter turns are not copied from a table. Each sticker gets a 3D
  position and an outward normal, the turned layer is rotated by 90 degrees, and the permutation is read off from
  where each sticker lands.
- **Cubie model (`rubiks/cubie.py`).** This is Kociemba's representation: corner and edge permutation and
  orientation, with his reference move tables. It also decides whether a state is reachable. A reachable state has
  corner twists that sum to 0 mod 3, edge flips that sum to 0 mod 2, and equal corner and edge permutation parity.

The tests check that the two models agree on every move and on random move sequences. They also check move orders
(`R U` has order 105) and a known identity sequence. During development, the `kociemba` package (a two-phase
solver) served as an independent oracle.

## The dataset

The generator (`rubiks/sampling.py`) makes training data as follows:

- **Scrambles.** Each training example is a random walk from the solved cube, with a depth drawn uniformly from 1
  to 20.
- **Canonical filter.** The walks use the same filter that the search uses: a face never turns twice in a row, and
  after two opposite faces (for example `U D`), neither face of that axis may follow.
- **Labels.** The label of each state is the inverse of the last scramble move, that is, the move that steps back
  towards the solved cube.
- **Size.** Version 2 used 10 million training rows and 100,000 validation rows.
- **Test set.** Separate test states are drawn uniformly from all reachable cubes through the cubie model: random
  permutations and orientations, corrected for parity and the orientation sums.

## The policy network and its training

- **Input.** The input is a one-hot encoding of the 54 stickers times 6 colours, which gives 324 features.
- **Architecture.** The network is an MLP, 324 → 1024 → 512 → 18, with LayerNorm and ReLU after each hidden
  layer: 869,906 parameters in total. Its output is a distribution over the 18 moves (`rubiks/model.py`).
- **Training setup.** Training used cross-entropy loss and AdamW (learning rate 1e-3, weight decay 1e-4), with
  batch size 2048, one epoch of linear warm-up followed by a cosine decay, and gradient clipping at 1.0.
- **Run.** The network trained for 12 epochs on the 10 million scrambles, which took 203 s on an Apple-silicon GPU
  (PyTorch MPS). The best epoch by validation accuracy, epoch 10, is the checkpoint shipped in `models/`.
- **Accuracy.** Next-move top-1 accuracy on validation is 46.2% (top-3: 62.9%). By scramble depth, it falls from
  100% at depth 1 to about 90% at depths 2 to 5 and 8% at depth 20. A deep scramble is rarely the shortest way
  back, and many moves are equally good there. So the network gives a useful ranking of moves rather than a single
  reliable move, and a search is needed on top of it.

## The search and its verification

The beam search is in `rubiks/search.py`. It works layer by layer:

1. It keeps the states in a batch on the device.
2. It expands every state by its allowed moves, using the same canonical filter as the dataset.
3. It scores each child by the network's cumulative log-probability along its path.
4. It removes duplicate states.
5. It keeps the best *W* states, where *W* is the beam width.

The search stops at the first layer that contains the solved cube, and it checks for the solved cube among all
children before any are pruned. The greedy baseline instead follows the top move at every step.

Every solution is verified three times:

1. **The search.** The search replays its path with the facelet model, and `NNSolver` replays it once more with
   the cubie model.
2. **The server.** The server applies the returned moves to the request state and reports `"verified"`.
3. **The page.** The page applies the moves with its own tables before it animates anything. A non-solving answer
   is shown as such, never hidden.

## Results

The test set is 1,000 uniformly random cube states:

| search | solved | mean solution length |
|---|---|---|
| greedy | 0% | - |
| beam 256 | 87% | 29.8 moves |
| beam 1024 | 100% | 26.4 moves |
| beam 4096 | 100% | 23.8 moves |

- **Version comparison.** Version 1, trained on 1M rows, solved 96% with 29.5 moves at beam 4096. Version 2, with
  ten times more data, solves 100% with 23.8 moves.
- **Speed.** On a CPU (Apple M4 Max) a solve takes about 130 ms at beam 1024 and 73 ms at beam 256. Beam 4096
  takes about 100 ms on an Apple GPU.

## The web app

The page is plain HTML and JavaScript with no build step. It shows the cube in 3D, using three.js r128 from cdnjs,
next to a 2D net, and it has controls to turn, scramble and load states.

The page knows nothing about the cube except a generated table file, `web/cube-tables.js`, produced by
`python -m rubiks.export_web`. That file holds:

- the move permutations and the sticker geometry;
- the scramble filter;
- 1,000 uniformly random states;
- 50 test vectors.

When the page loads, it re-checks those vectors and the 3D turn animation against the permutations.

The server (`rubiks/serve.py`) serves the page and two endpoints, `GET /solver` and `POST /solve`. It loads the
checkpoint once at start-up.

The page can also solve on its own, with no server. Its *Neural network (ONNX, in the browser)* backend runs
`models/policy-v2.onnx` with ONNX Runtime Web (WebAssembly, loaded from cdnjs) and `web/search.js`, a JavaScript
copy of the beam search in `rubiks/search.py`. The copy keeps the same move filter, cumulative log-probability scores
(float32, after a log-softmax), duplicate removal and tie-breaking order, and it checks every child for the solved
state before the beam is cut. It calls the network once per depth over the whole beam, and the beam width is a page
setting from 64 to 4096 (default 1024). A fixture of 70 cases (`tests/data/web-parity.json`) holds the answers of
the Python search, with the policy normalised by `torch.log_softmax` as the Python solver does it. `pytest` checks
it, and `web/search-test.html` checks `web/search.js` against it in a browser, with the JavaScript log-softmax
(computed in float64 and rounded to float32). Twenty cases use the ONNX model. Eleven use an exact synthetic policy
that forces score ties and so pins down the tie-breaking. Twenty-seven use the same synthetic values as logits, with a
common offset of 0, 1e6 or 1e30. The last twelve use logits built to give near ties, or exact ties that a log-softmax
written as x - (max + log(sum)) gets wrong at the offset 1e30. The native and the WebAssembly builds of ONNX Runtime
can differ in the last bits of the logits, so the ONNX scrambles were chosen so that no decision of the search
depended on a score gap smaller than 1e-4. GitHub Pages publishes the page together with the model.

## The second method and hybrid

Version 1.1 adds a 14,663,001-parameter residual cost-to-go network trained by a supervised warm start and
approximate value iteration, paired with batched weighted A* (`f = 0.6*g + h`, 200,000 expanded nodes). On the
same 200 random cubes, A* solved 177 at 19.21 moves on average; policy beam 65,536 solved all 200 at 20.90 moves
(20.79 on the common 177, where A* was shorter on 132, equal on 32 and longer on 13; 21.74 on the remaining
23). Running A* first and policy on exhaustion therefore solved all 200 at about 19.5 moves. The optional
server backend `hybrid` implements that ordering with a default fallback beam of 1,024, reports the successful
method and both search timings, and independently checks the final path. Policy remains the default. On ten
seed-0 uniform cubes, the native macOS CPU comparison (Apple silicon, macOS 26.5, PyTorch 2.14.1, 12 threads;
models loaded and warmed up; beam 1,024, maximum depth 40, A* batch 1,000) was:

| method | mean moves | mean seconds/cube | beam fallback share | solved |
|---|---|---|---|---|
| policy, beam 1,024 | 26.9 | 0.1738 | — | 10/10 |
| hybrid, cap 200,000 | 21.6 | 30.1874 | 3/10 (30%) | 10/10 |
| hybrid, cap 50,000 | 26.0 | 8.7421 | 9/10 (90%) | 10/10 |

These are native figures, not in-container. Keep policy on CPU-only hosts: 200,000 nodes gives shorter
solutions but took about 21 seconds per cube in an earlier native run and 30.1874 seconds in this repeat;
50,000 mostly falls back to policy with a several-second delay. A separate Linux CPU-container measurement
on the same Mac was 25–28 seconds per cube at 200,000 nodes. The page's 20-second timeout applies to local
servers too and does not cancel server work; subsequent Solve requests queue behind the lock. Hybrid is mainly
recommended natively with `--device mps` or `--device cuda`; CPU clients need a longer timeout. The README
also records the memory observations.

The fp16-weight ONNX export is included for later browser A*, with float32 I/O and computation and a maximum
absolute error of 0.0045747757 against PyTorch fp32 on 2,000 random cubes.

One documented limit of the in-browser parity: the page normalises the logits in float64 and rounds to float32, while PyTorch normalises in float32 with its own reduction order. On every real scramble tested the two agree exactly; only an artificial exact tie between two candidate paths, within one float32 rounding step, could make the page pick a different, equally valid solution.
