"""Policy MLP and residual cost-to-go network over the 324-feature one-hot encoding.

``MLP(out_features=18)`` is the policy (logits over the 18 moves); ``MLP(out_features=1)`` is a plain value
variant. ``ResMLP`` is the larger residual network used by the shipped cost-to-go checkpoint.
Each hidden block is ``Linear -> norm -> activation -> dropout``; the head is a single ``Linear``.

Default policy: 324 -> 1024 -> 512 -> 18 with LayerNorm, 869,906 parameters
(324*1024 + 1024 + 1024*512 + 512 + 512*18 + 18 = 866,834 weights and biases, plus 2*(1024 + 512) = 3,072
LayerNorm parameters).

Checkpoints are plain dicts ``{"model_state": state_dict, "config": {..., "model": model.config()}}`` saved with
``torch.save``; :func:`load_checkpoint` rebuilds the model from ``config["model"]`` with no training code.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from .data import N_FEATURES, encode

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
NORMS = ("layer", "batch", None)
POLICY_OUT = 18
VALUE_OUT = 1


class MLP(nn.Module):
    def __init__(self, in_features: int = N_FEATURES, hidden: Sequence[int] = (1024, 512),
                 out_features: int = POLICY_OUT, norm: str | None = "layer", dropout: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        if norm == "none":
            norm = None
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got {activation!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_features = int(in_features)
        self.hidden = tuple(int(h) for h in hidden)
        self.out_features = int(out_features)
        self.norm = norm
        self.dropout = float(dropout)
        self.activation = activation

        layers: list[nn.Module] = []
        prev = self.in_features
        for h in self.hidden:
            layers.append(nn.Linear(prev, h))
            if norm == "layer":
                layers.append(nn.LayerNorm(h))
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            layers.append(ACTIVATIONS[activation]())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """float32 (N, 324) one-hot -> (N, out_features): logits for the policy, the estimate for the value."""
        return self.head(self.body(x))

    def logits_from_uint8(self, states: torch.Tensor) -> torch.Tensor:
        """``forward(encode(states))`` for uint8 (N, 54) states on the model's device."""
        return self.forward(encode(states))

    def config(self) -> dict:
        """JSON-serialisable constructor arguments (rebuild with ``MLP(**config)``)."""
        return {"in_features": self.in_features, "hidden": list(self.hidden), "out_features": self.out_features,
                "norm": self.norm, "dropout": self.dropout, "activation": self.activation}


def _norm_layer(norm: str | None, width: int) -> nn.Module:
    if norm == "layer":
        return nn.LayerNorm(width)
    if norm == "batch":
        return nn.BatchNorm1d(width)
    return nn.Identity()


class _ResBlock(nn.Module):
    """``act(x + norm2(fc2(act(norm1(fc1(x))))))``, width ``w`` throughout."""

    def __init__(self, width: int, norm: str | None, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.norm1 = _norm_layer(norm, width)
        self.act1 = ACTIVATIONS[activation]()
        self.fc2 = nn.Linear(width, width)
        self.norm2 = _norm_layer(norm, width)
        self.act2 = ACTIVATIONS[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.fc1(x)))
        return self.act2(self.norm2(self.fc2(h)) + x)


class ResMLP(nn.Module):
    """DeepCubeA-style residual network over the 324-feature one-hot input.

    stem: ``Linear(in, stem[0]) -> norm -> act -> Linear(stem[0], w) -> norm -> act`` with ``w = stem[1]``; then
    ``blocks`` residual blocks of width ``w`` (:class:`_ResBlock`); head ``Linear(w, out_features)``.

    Parameters (``nb(n)`` = 2n for batch/layer norm, 0 for none; batch-norm running statistics are buffers, not
    parameters)::

        in*s0 + s0 + nb(s0) + s0*w + w + nb(w) + blocks * (2*(w*w + w) + 2*nb(w)) + w*out + out

    Defaults (stem 5000,1000, 4 blocks, batch norm, 1 output): 14,663,001."""

    def __init__(self, in_features: int = N_FEATURES, stem: Sequence[int] = (5000, 1000), blocks: int = 4,
                 out_features: int = VALUE_OUT, norm: str | None = "batch", activation: str = "relu"):
        super().__init__()
        if norm == "none":
            norm = None
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got {activation!r}")
        stem = tuple(int(h) for h in stem)
        if len(stem) != 2 or any(h < 1 for h in stem):
            raise ValueError(f"stem must be two widths >= 1, got {stem}")
        if int(blocks) < 0:
            raise ValueError("blocks must be >= 0")
        self.in_features = int(in_features)
        self.stem_widths = stem
        self.n_blocks = int(blocks)
        self.out_features = int(out_features)
        self.norm = norm
        self.activation = activation

        s0, w = stem
        self.stem = nn.Sequential(nn.Linear(self.in_features, s0), _norm_layer(norm, s0), ACTIVATIONS[activation](),
                                  nn.Linear(s0, w), _norm_layer(norm, w), ACTIVATIONS[activation]())
        self.blocks = nn.ModuleList(_ResBlock(w, norm, activation) for _ in range(self.n_blocks))
        self.head = nn.Linear(w, self.out_features)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The input to the head: float32 (N, 324) -> (N, w)."""
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """float32 (N, 324) one-hot -> (N, out_features)."""
        return self.head(self.features(x))

    def logits_from_uint8(self, states: torch.Tensor) -> torch.Tensor:
        """``forward(encode(states))`` for uint8 (N, 54) states on the model's device."""
        return self.forward(encode(states))

    def config(self) -> dict:
        """JSON-serialisable constructor arguments plus ``"arch": "resnet"`` (rebuild with :func:`build_model`)."""
        return {"arch": "resnet", "in_features": self.in_features, "stem": list(self.stem_widths),
                "blocks": self.n_blocks, "out_features": self.out_features, "norm": self.norm,
                "activation": self.activation}


ARCHS = ("mlp", "resnet")


def build_model(model_config: dict) -> nn.Module:
    """Rebuild a model from ``config["model"]``: dispatch on ``"arch"`` (absent = ``"mlp"``, the format of every
    checkpoint written before ResMLP existed); the remaining keys are the constructor arguments."""
    cfg = dict(model_config)
    arch = cfg.pop("arch", "mlp")
    if arch == "mlp":
        return MLP(**cfg)
    if arch == "resnet":
        return ResMLP(**cfg)
    raise ValueError(f"unknown model arch {arch!r} (known: {ARCHS})")


def policy_mlp(**kwargs) -> MLP:
    return MLP(out_features=POLICY_OUT, **kwargs)


def value_mlp(**kwargs) -> MLP:
    return MLP(out_features=VALUE_OUT, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def save_checkpoint(path: str | Path, model: nn.Module, config: dict, **extra) -> None:
    """Write ``{"model_state": <CPU state_dict>, "config": config, **extra}``; ``config["model"]`` is set to
    ``model.config()`` so the file is self-describing. ``config`` and ``extra`` must be JSON-serialisable."""
    cfg = dict(config)
    cfg["model"] = model.config()
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    # JSON round trip: only plain types reach the pickle, so torch.load(weights_only=True) accepts the file.
    plain = json.loads(json.dumps({"config": cfg, **extra}))
    torch.save({"model_state": state, **plain}, Path(path))


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild on ``device`` in eval mode; return ``(model, checkpoint_dict)``.

    ``config.model.arch`` selects the architecture; an absent tag means the original MLP.
    """
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if "model_state" not in ckpt or "config" not in ckpt or "model" not in ckpt["config"]:
        raise ValueError(f"{path}: not a rubiks-nn checkpoint (need model_state and config.model)")
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` -> cuda if available, else mps if available, else cpu; any other name is used as given."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
