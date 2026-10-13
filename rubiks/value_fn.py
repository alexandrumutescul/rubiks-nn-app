"""Cost-to-go inference and cached move tables; no training dependencies."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from . import cube
from .model import VALUE_OUT

ValueFn = Callable[[torch.Tensor], torch.Tensor]


def make_value_fn(model: torch.nn.Module, device: torch.device | str | None = None, pad_to: int = 256) -> ValueFn:
    """Wrap a value model as a heuristic: uint8 (N, 54) -> float32 (N,) cost-to-go, clamped at 0, on the device.

    Inference mode, eval mode. Batches are padded up to a multiple of ``pad_to`` rows (copies of row 0) and the
    padding is cut off again, so a search that calls the network with many different batch sizes only meets a few
    distinct shapes (each new shape costs a kernel compilation on MPS). ``pad_to=1`` disables the padding. The
    function carries ``.device``, which :mod:`rubiks.astar` uses by default, and ``.pad_to``."""
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    if model.out_features != VALUE_OUT:
        raise ValueError(f"not a value model ({model.out_features} outputs, expected {VALUE_OUT})")
    if pad_to < 1:
        raise ValueError("pad_to must be >= 1")
    model.eval()

    def value_fn(states: torch.Tensor) -> torch.Tensor:
        n = int(states.shape[0])
        x = states.to(dev)
        with torch.inference_mode():
            if n == 0:
                return torch.zeros(0, dtype=torch.float32, device=dev)
            m = -(-n // pad_to) * pad_to
            if m != n:
                x = torch.cat([x, x[:1].expand(m - n, x.shape[1])], dim=0)
            return model.logits_from_uint8(x).reshape(-1)[:n].float().clamp_min(0.0)

    value_fn.device = dev  # type: ignore[attr-defined]
    value_fn.pad_to = pad_to  # type: ignore[attr-defined]
    return value_fn


class _DeviceTables:
    def __init__(self, device: torch.device):
        self.perms = torch.from_numpy(cube.MOVE_PERMS.astype(np.int64)).to(device)   # (18, 54)
        self.solved = torch.from_numpy(np.array(cube.SOLVED)).to(device)             # (54,) uint8


_TABLES: dict[str, _DeviceTables] = {}


def _tables(device: torch.device) -> _DeviceTables:
    key = str(device)
    if key not in _TABLES:
        _TABLES[key] = _DeviceTables(device)
    return _TABLES[key]


def expand_children(states: torch.Tensor) -> torch.Tensor:
    """All 18 children of each state, on the states' device: uint8 (B, 54) -> (B, 18, 54), child[b, a] = a(s_b)
    (``torch.gather`` with ``cube.MOVE_PERMS``; no move filter)."""
    b = states.shape[0]
    perms = _tables(states.device).perms
    return torch.gather(states.unsqueeze(1).expand(b, cube.N_MOVES, cube.N_FACELETS), 2,
                        perms.unsqueeze(0).expand(b, cube.N_MOVES, cube.N_FACELETS))


