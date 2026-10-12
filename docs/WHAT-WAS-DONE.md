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

## What is next

The next step is a cost-to-go network that estimates how many moves a state is from solved, paired with A* search,
to shorten the solutions. It is in training.
