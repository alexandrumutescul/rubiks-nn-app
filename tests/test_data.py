import numpy as np
import pytest
import torch

from rubiks import cube, data, sampling

N = 5000
MAXD = 20


@pytest.fixture(scope="module")
def split():
    states, moves, depth = sampling.random_scrambles(np.random.default_rng(7), N, MAXD)
    last_move = moves[np.arange(N), depth.astype(np.int64) - 1].astype(np.int8)
    return {"states": states, "depth": depth, "moves": moves, "last_move": last_move}


def test_undo_table():
    assert data.UNDO_TABLE.tolist() == [cube.inverse_move(m) for m in range(18)]
    names = [cube.MOVE_NAMES[m] for m in data.UNDO_TABLE]
    assert names[:3] == ["U'", "U", "U2"] and names[-3:] == ["B'", "B", "B2"]


def test_undo_label_is_inverse_of_last_move(split):
    undo = data.undo_labels(split["last_move"])
    assert undo.dtype == np.int64 and undo.shape == (N,)
    assert all(int(u) == cube.inverse_move(int(m)) for u, m in zip(undo, split["last_move"]))


def test_undo_label_steps_back_along_the_scramble(split):
    undo = data.undo_labels(split["last_move"])
    for i in range(200):
        d = int(split["depth"][i])
        prev = cube.apply_sequence(cube.SOLVED, split["moves"][i, :d - 1].tolist())
        assert np.array_equal(cube.apply_move(split["states"][i], int(undo[i])), prev)


def test_undo_label_rejects_padding():
    with pytest.raises(ValueError):
        data.undo_labels(np.array([3, -1], dtype=np.int8))


def test_one_hot_encoding(split):
    states = split["states"][:500]
    x = data.encode(torch.from_numpy(states))
    assert x.dtype == torch.float32 and tuple(x.shape) == (500, 324)
    xn = x.numpy()
    assert np.all(xn.sum(axis=1) == 54)
    assert set(np.unique(xn).tolist()) == {0.0, 1.0}
    rows, cols = np.nonzero(xn)
    expected_cols = (np.arange(54)[None, :] * 6 + states.astype(np.int64)).ravel()
    assert np.array_equal(rows, np.repeat(np.arange(500), 54))
    assert np.array_equal(cols, expected_cols)
    assert np.array_equal(data.encode_numpy(states), xn)


def test_encode_solved_and_dtype_checks():
    x = data.encode(torch.from_numpy(np.array(cube.SOLVED)[None]))[0].numpy()
    assert np.flatnonzero(x).tolist() == [p * 6 + p // 9 for p in range(54)]
    with pytest.raises(ValueError):
        data.encode(torch.zeros(2, 54, dtype=torch.int64))
    with pytest.raises(ValueError):
        data.encode(torch.zeros(2, 53, dtype=torch.uint8))


@pytest.mark.parametrize("n,batch", [(1000, 64), (1024, 256), (7, 10), (1, 1)])
def test_batches_cover_every_index_once_per_epoch(n, batch):
    sampler = data.BatchSampler(n, batch, seed=3)
    epochs = [list(sampler) for _ in range(3)]
    for ep in epochs:
        assert len(ep) == len(sampler) == -(-n // batch)
        assert all(len(b) == batch for b in ep[:-1])
        idx = np.concatenate(ep)
        assert np.array_equal(np.sort(idx), np.arange(n))
    if n > batch:
        assert not np.array_equal(np.concatenate(epochs[0]), np.concatenate(epochs[1]))  # reshuffled
    again = data.BatchSampler(n, batch, seed=3)
    assert all(np.array_equal(np.concatenate(a), np.concatenate(list(again))) for a in epochs)


def test_batches_drop_last():
    sampler = data.BatchSampler(1025, 256, seed=0, drop_last=True)
    ep = list(sampler)
    assert len(ep) == len(sampler) == 4 and all(len(b) == 256 for b in ep)
    assert len(np.unique(np.concatenate(ep))) == 1024


def test_load_split_round_trip(tmp_path, split):
    np.savez(tmp_path / "val.npz", **split)
    np.savez(tmp_path / "test_uniform.npz", states=split["states"][:10])
    got = data.load_split(tmp_path, "val")
    for k in data.SPLIT_KEYS:
        assert np.array_equal(got[k], split[k]) and got[k].dtype == split[k].dtype
    assert np.array_equal(data.load_uniform(tmp_path), split["states"][:10])
    np.savez(tmp_path / "bad.npz", states=split["states"])
    with pytest.raises(ValueError):
        data.load_split(tmp_path, "bad")


def test_device_split(split):
    ds = data.DeviceSplit(split, "cpu", limit=100)
    assert ds.n == 100 and ds.states.dtype == torch.uint8 and tuple(ds.states.shape) == (100, 54)
    assert np.array_equal(ds.labels.numpy(), data.undo_labels(split["last_move"][:100]))
    assert np.array_equal(ds.depth_np, split["depth"][:100].astype(np.int64))
