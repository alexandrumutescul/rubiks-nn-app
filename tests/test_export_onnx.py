import hashlib
import json

import numpy as np
import pytest
import torch

from rubiks import cube, model
from rubiks.data import encode
from rubiks.export_onnx import export_checkpoint

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("arch", ["mlp", "resnet"])
def test_value_export_contract_and_dynamic_batch(tmp_path, arch, fp16):
    torch.manual_seed(1)
    net = (model.MLP(hidden=(16, 8), out_features=1) if arch == "mlp" else
           model.ResMLP(stem=(16, 8), blocks=1)).eval()
    checkpoint, out = tmp_path / "value.pt", tmp_path / "cost-to-go-v1.onnx"
    model.save_checkpoint(checkpoint, net, {})
    original = checkpoint.read_bytes()
    contract = export_checkpoint(checkpoint, out, fp16=fp16, probes=12)
    assert checkpoint.read_bytes() == original
    assert json.loads((tmp_path / "cost-to-go-v1.onnx.json").read_text()) == contract
    assert contract["input"] == {"name": "onehot", "dtype": "float32", "shape": ["batch", 324]}
    assert contract["output"]["name"] == "value" and contract["output"]["shape"] == ["batch", 1]
    assert contract["output"]["dtype"] == "float32"
    assert contract["check"]["max_abs_diff"] <= (1e-2 if fp16 else 1e-4)
    assert contract["check"]["rtol"] == 0 and contract["check"]["n"] == 12
    assert contract["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert contract["checkpoint_sha256"] == hashlib.sha256(original).hexdigest()
    graph = onnx.load(str(out))
    types = {tensor.data_type for tensor in graph.graph.initializer}
    assert onnx.TensorProto.FLOAT16 in types if fp16 else onnx.TensorProto.FLOAT in types
    if fp16:
        assert onnx.TensorProto.FLOAT not in types
    runtime = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for batch in (1, 3, 19):
        x = encode(torch.from_numpy(np.tile(cube.SOLVED, (batch, 1))))
        with torch.inference_mode():
            expected = net(x).numpy()
        actual = runtime.run(["value"], {"onehot": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-2 if fp16 else 1e-4, rtol=0)


def test_policy_export_still_supported(tmp_path):
    checkpoint, out = tmp_path / "policy.pt", tmp_path / "policy.onnx"
    model.save_checkpoint(checkpoint, model.MLP(hidden=(8,)), {})
    contract = export_checkpoint(checkpoint, out, probes=2)
    assert contract["output"]["name"] == "logits"
    assert contract["output"]["shape"] == ["batch", 18]
    assert contract["move_names"] == list(cube.MOVE_NAMES)


def test_export_refuses_to_overwrite_checkpoint(tmp_path):
    checkpoint = tmp_path / "value.pt"
    model.save_checkpoint(checkpoint, model.ResMLP(stem=(8, 4), blocks=0), {})
    original = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="differ from checkpoint"):
        export_checkpoint(checkpoint, checkpoint, fp16=True, probes=2)
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize("existing_output", [False, True])
def test_parity_failure_preserves_output_and_removes_temporary_files(tmp_path, monkeypatch, existing_output):
    checkpoint, out = tmp_path / "value.pt", tmp_path / "value.onnx"
    contract_path = tmp_path / "value.onnx.json"
    model.save_checkpoint(checkpoint, model.ResMLP(stem=(8, 4), blocks=0), {})
    if existing_output:
        out.write_bytes(b"previous model")
        contract_path.write_text('{"previous": true}\n')
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    inference_session = ort.InferenceSession

    class WrongOutputSession:
        def __init__(self, path, **kwargs):
            assert path != str(out)
            self.runtime = inference_session(path, **kwargs)

        def run(self, names, inputs):
            return [self.runtime.run(names, inputs)[0] + 1.0]

    monkeypatch.setattr(ort, "InferenceSession", WrongOutputSession)
    with pytest.raises(AssertionError, match="ONNX parity failed"):
        export_checkpoint(checkpoint, out, probes=2)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files
