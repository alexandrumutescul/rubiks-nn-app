import numpy as np
import pytest
import torch

from rubiks import cube, model, value_fn
from rubiks.solver_value import ValueSolver


def test_value_clamps_pads_and_preserves_rows():
    torch.manual_seed(0)
    net = model.ResMLP(stem=(16, 8), blocks=1).eval()
    states = torch.from_numpy(np.stack([cube.apply_move(cube.SOLVED, m) for m in range(18)]))
    fn = value_fn.make_value_fn(net, "cpu", pad_to=8)
    with torch.inference_mode():
        torch.testing.assert_close(fn(states), net.logits_from_uint8(states).reshape(-1).clamp_min(0))
        net.head.weight.zero_()
        net.head.bias.fill_(-2)
    assert torch.equal(fn(states), torch.zeros(18))
    assert fn(states[:0]).shape == (0,)
    assert fn.device == torch.device("cpu") and fn.pad_to == 8
    assert not fn(states).requires_grad and not net.training


def test_expand_matches_facelet_model():
    states = np.stack([cube.SOLVED, cube.apply_move(cube.SOLVED, 4)])
    out = value_fn.expand_children(torch.from_numpy(states)).numpy()
    for i, state in enumerate(states):
        for m in range(18):
            np.testing.assert_array_equal(out[i, m], cube.apply_move(state, m))
    assert value_fn.expand_children(torch.from_numpy(states[:0])).shape == (0, 18, 54)


def test_invalid_value_model_and_padding():
    with pytest.raises(ValueError, match="not a value"):
        value_fn.make_value_fn(model.MLP(hidden=(4,)))
    with pytest.raises(ValueError, match="pad_to"):
        value_fn.make_value_fn(model.ResMLP(stem=(8, 4), blocks=0), pad_to=0)


def test_value_solver_validation_and_solution(tmp_path):
    path = tmp_path / "value.pt"
    model.save_checkpoint(path, model.ResMLP(stem=(8, 4), blocks=0), {})
    solver = ValueSolver(path, device="cpu", max_nodes=1)
    for state in ([0] * 53, [6] * 54, [0.5] * 54, [True] * 54, np.zeros(54)):
        assert "error" in solver.solve(state)
    result = solver.solve(cube.apply_move(cube.SOLVED, 3))
    assert result["source"] == "value-astar" and result["moves"] == [4]
    assert solver.solve(cube.SOLVED)["moves"] == []
    state = cube.apply_sequence(cube.SOLVED, cube.parse_moves("R U"))
    assert solver.solve(state)["error"].startswith("no solution found within")
