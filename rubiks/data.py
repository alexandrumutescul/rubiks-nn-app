"""Training data for the policy network: loading, the undo label, the one-hot encoding and batching.

* ``load_split(dir, name)`` reads ``<dir>/<name>.npz`` (format: ``the dataset format note of the research repository``) and returns its
  arrays; ``load_uniform(dir)`` returns the uniformly random test states.
* ``undo_labels(last_move)`` is the policy label: ``inverse_move(last_move)``, the move that steps back along the
  scramble (X <-> X', X2 is its own inverse), via an 18-entry lookup table.
* ``encode(states)`` maps a uint8 (N, 54) torch tensor to the float32 (N, 324) one-hot input, ON THE TENSOR'S
  DEVICE, so the training set stays resident as uint8 and is expanded one batch at a time. Feature index =
  ``position * 6 + colour`` (row-major over (54, 6)). ``encode_numpy`` is the same encoding in numpy (used by
  the ONNX check and as the reference for the web page).
* ``BatchSampler(n, batch_size, seed)``: a seeded shuffled index iterator; each ``iter()`` is one epoch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import cube

N_COLOURS = 6
N_FEATURES = cube.N_FACELETS * N_COLOURS  # 324
ENCODING = "one-hot 54x6 row-major, feature = position*6 + colour"

# UNDO_TABLE[m] = cube.inverse_move(m)
UNDO_TABLE = np.array([cube.inverse_move(m) for m in range(cube.N_MOVES)], dtype=np.int64)
UNDO_TABLE.setflags(write=False)

SPLIT_KEYS = ("states", "depth", "moves", "last_move")


def load_split(data_dir: str | Path, name: str) -> dict[str, np.ndarray]:
    """Arrays of ``<data_dir>/<name>.npz``: states uint8 (N, 54), depth uint8 (N,), moves int8 (N, D),
    last_move int8 (N,). Shapes and dtypes are checked."""
    path = Path(data_dir) / f"{name}.npz"
    with np.load(path) as z:
        missing = [k for k in SPLIT_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        out = {k: z[k] for k in SPLIT_KEYS}
    n = out["states"].shape[0]
    if out["states"].dtype != np.uint8 or out["states"].shape != (n, cube.N_FACELETS):
        raise ValueError(f"{path}: states must be uint8 (N, 54), got {out['states'].dtype} {out['states'].shape}")
    for k in ("depth", "last_move"):
        if out[k].shape != (n,):
            raise ValueError(f"{path}: {k} must have shape ({n},), got {out[k].shape}")
    if out["moves"].ndim != 2 or out["moves"].shape[0] != n:
        raise ValueError(f"{path}: moves must have shape ({n}, max_depth), got {out['moves'].shape}")
    return out


def load_uniform(data_dir: str | Path, name: str = "test_uniform") -> np.ndarray:
    """States uint8 (N, 54) of ``<data_dir>/test_uniform.npz``."""
    path = Path(data_dir) / f"{name}.npz"
    with np.load(path) as z:
        states = z["states"]
    if states.dtype != np.uint8 or states.ndim != 2 or states.shape[1] != cube.N_FACELETS:
        raise ValueError(f"{path}: states must be uint8 (N, 54), got {states.dtype} {states.shape}")
    return states


def undo_labels(last_move: np.ndarray) -> np.ndarray:
    """Policy labels int64 (N,): ``UNDO_TABLE[last_move]`` = ``cube.inverse_move(last_move[i])`` per row."""
    lm = np.asarray(last_move).astype(np.int64)
    if lm.size and (lm.min() < 0 or lm.max() >= cube.N_MOVES):
        raise ValueError("last_move out of range [0, 18)")
    return UNDO_TABLE[lm]


def encode(states: torch.Tensor) -> torch.Tensor:
    """One-hot encoding on the tensor's device: uint8 (N, 54) -> float32 (N, 324), feature = position*6 + colour.

    Colours outside 0..5 give an all-zero block for that position (callers validate states; see solver_nn).
    """
    if not isinstance(states, torch.Tensor):
        raise TypeError("encode expects a torch.Tensor; use encode_numpy for numpy arrays")
    if states.dtype != torch.uint8 or states.ndim != 2 or states.shape[1] != cube.N_FACELETS:
        raise ValueError(f"states must be uint8 (N, 54), got {states.dtype} {tuple(states.shape)}")
    colours = torch.arange(N_COLOURS, dtype=torch.uint8, device=states.device)
    return (states.unsqueeze(-1) == colours).reshape(states.shape[0], N_FEATURES).to(torch.float32)


def encode_numpy(states: np.ndarray) -> np.ndarray:
    """The same encoding in numpy: uint8 (N, 54) -> float32 (N, 324)."""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != cube.N_FACELETS:
        raise ValueError(f"states must have shape (N, 54), got {states.shape}")
    return (states[:, :, None] == np.arange(N_COLOURS)).reshape(states.shape[0], N_FEATURES).astype(np.float32)


class BatchSampler:
    """Seeded shuffled batches of row indices. Each ``iter(sampler)`` is one epoch: a fresh permutation from the
    sampler's own ``np.random.default_rng(seed)``, cut into consecutive slices of ``batch_size`` (the last one
    shorter unless ``drop_last``). Same seed -> same sequence of epochs."""

    def __init__(self, n: int, batch_size: int, seed: int, drop_last: bool = False):
        if n < 0 or batch_size < 1:
            raise ValueError("need n >= 0 and batch_size >= 1")
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return -(-self.n // self.batch_size)

    def __iter__(self):
        perm = self.rng.permutation(self.n)
        for start in range(0, self.n, self.batch_size):
            idx = perm[start:start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                return
            yield idx


class DeviceSplit:
    """A split resident on a device: states uint8 (N, 54), undo labels int64 (N,), depth int64 (N,)."""

    def __init__(self, split: dict[str, np.ndarray], device: torch.device | str, limit: int = 0):
        n = split["states"].shape[0] if not limit else min(int(limit), split["states"].shape[0])
        self.n = n
        self.states = torch.from_numpy(np.ascontiguousarray(split["states"][:n])).to(device)
        self.labels = torch.from_numpy(undo_labels(split["last_move"][:n])).to(device)
        self.depth_np = split["depth"][:n].astype(np.int64)
        self.depth = torch.from_numpy(self.depth_np).to(device)
