import numpy as np
import pytest
import torch

from rubiks import data, model, sampling


@pytest.fixture(scope="module")
def states():
    s, _, _ = sampling.random_scrambles(np.random.default_rng(11), 64, 20)
    return torch.from_numpy(s)


def test_output_shapes(states):
    m = model.MLP().eval()
    x = data.encode(states)
    assert tuple(m(x).shape) == (64, 18)
    assert tuple(m(x[:1]).shape) == (1, 18)
    assert m(x).dtype == torch.float32


def test_default_parameter_count():
    n = model.count_parameters(model.MLP())
    assert 800_000 <= n <= 1_200_000
    expected = 324 * 1024 + 1024 + 1024 * 512 + 512 + 512 * 18 + 18 + 2 * (1024 + 512)
    assert n == expected == 869_906


def test_value_variant_has_one_output(states):
    v = model.value_mlp().eval()
    assert v.out_features == 1
    assert tuple(v(data.encode(states)).shape) == (64, 1)
    assert model.count_parameters(v) == model.count_parameters(model.MLP()) - 512 * 17 - 17


@pytest.mark.parametrize("norm", ["layer", "batch", None, "none"])
def test_norm_variants(states, norm):
    torch.manual_seed(0)
    m = model.MLP(hidden=(32, 16), norm=norm, dropout=0.1)
    m.train()
    out = m(data.encode(states))
    out.sum().backward()
    m.eval()
    assert tuple(m(data.encode(states)).shape) == (64, 18)


def test_logits_from_uint8_matches_forward_of_encode(states):
    torch.manual_seed(0)
    m = model.MLP(hidden=(64, 32)).eval()
    with torch.no_grad():
        assert torch.equal(m.logits_from_uint8(states), m(data.encode(states)))


def test_bad_arguments():
    with pytest.raises(ValueError):
        model.MLP(norm="group")
    with pytest.raises(ValueError):
        model.MLP(activation="tanh")
    with pytest.raises(ValueError):
        model.MLP(dropout=1.0)


def test_checkpoint_round_trip(tmp_path, states):
    torch.manual_seed(0)
    m = model.MLP(hidden=(48, 24), norm="batch").eval()
    model.save_checkpoint(tmp_path / "c.pt", m, {"parameters": model.count_parameters(m)}, epoch=3)
    raw = torch.load(tmp_path / "c.pt", map_location="cpu", weights_only=True)
    assert set(raw) >= {"model_state", "config", "epoch"}
    assert raw["config"]["model"] == m.config()
    m2, ckpt = model.load_checkpoint(tmp_path / "c.pt", "cpu")
    assert ckpt["epoch"] == 3 and not m2.training
    with torch.no_grad():
        assert torch.equal(m2.logits_from_uint8(states), m.logits_from_uint8(states))


def test_resolve_device():
    assert model.resolve_device("cpu") == torch.device("cpu")
    assert model.resolve_device("auto").type in ("cpu", "cuda", "mps")
