# Trained weights

| file | what it is |
|---|---|
| `policy-v2.pt` | the trained policy network (PyTorch checkpoint, 3.5 MB); this is what the server loads |
| `policy-v2.onnx` | the same network exported to ONNX (opset 17); the page's in-browser solver loads it |
| `policy-v2.onnx.json` | the ONNX file's contract: input and output names, shapes, encoding, move order, and an export check |
| `policy-v2.train-config.json` | the training settings and results: data, optimiser, schedule, best epoch, validation accuracy |
| `SHA256SUMS` | SHA-256 checksums of the two model files |

The network is an MLP, 324 → 1024 → 512 → 18, with LayerNorm and ReLU: 869,906 parameters. It was trained for 12
epochs on 10M random scrambles. The shipped checkpoint is from epoch 10, the best by validation accuracy, with
next-move top-1 accuracy of 46.2%.

The paths in the two `.json` files (`runs/...`, the data directory, and `policy.onnx`) are where the files lived
when they were trained and exported. They are a record of the run, not paths to use.

## Verify the files

```
cd models
shasum -a 256 -c SHA256SUMS        # macOS
sha256sum -c SHA256SUMS            # Linux
```

On Windows PowerShell, `Get-FileHash models\policy-v2.pt` prints a hash to compare with the line in `SHA256SUMS`.

## Load the checkpoint in Python

Run this from the repository root, after `pip install -e .`:

```python
from rubiks.model import load_checkpoint
model, ckpt = load_checkpoint("models/policy-v2.pt", "cpu")     # an MLP in eval mode, and the raw checkpoint
print(model.config())                                          # {'in_features': 324, 'hidden': [1024, 512], ...}
print(ckpt["epoch"], ckpt["metrics"]["val_top1"])              # 10 0.46238
```

To solve a cube with the checkpoint:

```python
from rubiks import cube
from rubiks.solver_nn import NNSolver
solver = NNSolver("models/policy-v2.pt", device="cpu", width=1024)
print(solver.solve(cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U F' L2 D B'"))))
```

The checkpoint is a plain dict, `{"model_state", "config", "epoch", "metrics"}`, written with `torch.save`.
`load_checkpoint` reads it with `weights_only=True` and rebuilds the model from `config["model"]`.

## ONNX contract (`policy-v2.onnx`)

| | name | dtype | shape |
|---|---|---|---|
| input | `onehot` | float32 | `[batch, 324]` |
| output | `logits` | float32 | `[batch, 18]` |

- **Encoding.** One-hot over 54 positions and 6 colours, row-major: `feature = position*6 + colour`. This is
  `rubiks.data.encode`.
- **Facelet order.** Kociemba's U R F D L B order: `facelet = face*9 + row*3 + col`, and a solved cube has colour
  `facelet // 9` at each facelet.
- **Output.** The output holds unnormalised logits over the moves, in this order:
  `U U' U2 R R' R2 F F' F2 D D' D2 L L' L2 B B' B2`. A softmax over them gives the next-move policy.
- **Export check.** On 256 random states, at batch sizes 256, 7 and 1, the ONNX Runtime outputs matched PyTorch to
  within 6.2e-6, and the arg-max move agreed for every state (details in `policy-v2.onnx.json`).
