"""Export a policy or cost-to-go checkpoint to ONNX and check it with ONNX Runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import cube, cubie
from .data import encode
from .model import count_parameters, load_checkpoint


def _fp16_weights(graph) -> None:
    """Store weights in float16, casting to float32 for portable CPU arithmetic and I/O.

    Keeping accumulation in float32 avoids rounding the scalar value to half precision.
    The checkpoint itself is never converted or written.
    """
    import onnx
    from onnx import helper, numpy_helper

    casts = []
    for tensor in graph.graph.initializer:
        if tensor.data_type != onnx.TensorProto.FLOAT:
            continue
        name = tensor.name
        half_name = name + "__stored_fp16"
        values = numpy_helper.to_array(tensor).astype(np.float16)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: weights cannot be represented as finite float16")
        tensor.CopyFrom(numpy_helper.from_array(values, half_name))
        casts.append(helper.make_node("Cast", [half_name], [name], to=onnx.TensorProto.FLOAT))
    original = list(graph.graph.node)
    del graph.graph.node[:]
    graph.graph.node.extend([*casts, *original])


def export_checkpoint(checkpoint: str | Path, out: str | Path, *, fp16: bool = False,
                      probes: int = 2000, seed: int = 0) -> dict:
    """Replace the ONNX and ``<out>.json`` only after absolute-error parity passes.

    Export dependencies are optional: ``onnx`` and ``onnxruntime`` are only needed here.
    """
    import onnx
    import onnxruntime as ort

    if probes < 1 or seed < 0:
        raise ValueError("need probes >= 1 and seed >= 0")
    out = Path(out)
    if out.resolve() == Path(checkpoint).resolve():
        raise ValueError("output must differ from checkpoint")
    ort.disable_telemetry_events()
    model, _ = load_checkpoint(checkpoint, "cpu")
    if model.in_features != 324 or model.out_features not in (1, cube.N_MOVES):
        raise ValueError("expected 324 input features and 1 value or 18 policy outputs")
    is_value = model.out_features == 1
    output_name = "value" if is_value else "logits"
    with tempfile.NamedTemporaryFile(prefix=f".{out.name}.", suffix=".onnx",
                                     dir=out.parent, delete=False) as temporary:
        temporary_out = Path(temporary.name)
    temporary_contract = temporary_out.with_suffix(temporary_out.suffix + ".json")
    try:
        torch.onnx.export(model, torch.zeros(2, 324), str(temporary_out), input_names=["onehot"],
                          output_names=[output_name], opset_version=17, dynamo=False,
                          dynamic_axes={"onehot": {0: "batch"}, output_name: {0: "batch"}})
        graph = onnx.load(str(temporary_out))
        if fp16:
            _fp16_weights(graph)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(temporary_out))

        options = ort.SessionOptions()
        options.intra_op_num_threads = torch.get_num_threads()
        runtime = ort.InferenceSession(str(temporary_out), sess_options=options, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(seed)
        states = np.stack([cubie.random_cube(rng).to_facelets() for _ in range(probes)])
        inputs = encode(torch.from_numpy(states))
        tolerance = 1e-2 if fp16 else 1e-4
        max_diff = 0.0
        batches = [inputs[i:i + 256] for i in range(0, probes, 256)]
        batches += [inputs[:min(7, probes)], inputs[:1]]
        with torch.inference_mode():
            for x in batches:
                expected = model(x).numpy()
                actual = runtime.run([output_name], {"onehot": x.numpy()})[0]
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise AssertionError("ONNX output has the wrong shape or non-finite values")
                difference = float(np.max(np.abs(actual - expected)))
                max_diff = max(max_diff, difference)
                if not np.isfinite(expected).all() or difference > tolerance:
                    raise AssertionError(f"ONNX parity failed: max absolute difference {difference} > {tolerance}")
        blob = temporary_out.read_bytes()
        contract = {
            "format": f"rubiks-nn {'value' if is_value else 'policy'} ONNX v1",
            "file": out.name, "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
            "input": {"name": "onehot", "dtype": "float32", "shape": ["batch", 324]},
            "output": {"name": output_name, "dtype": "float32", "shape": ["batch", model.out_features],
                       "semantics": "raw cost-to-go in half-turn moves; clamp at 0 for A*" if is_value else
                                    "unnormalised logits over move_names; softmax gives the next-move policy"},
            "encoding": "one-hot 54x6 row-major, feature = position*6 + colour",
            "facelet_order": "Kociemba URFDLB, facelet = face*9 + row*3 + col, solved colour = facelet // 9",
            "weights_dtype": "float16" if fp16 else "float32", "compute_dtype": "float32",
            "opset": 17, "exporter": "torch.onnx (TorchScript)", "model": model.config(),
            "parameters": count_parameters(model), "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__),
            "check": {"n": probes, "seed": seed, "distribution": "rubiks.cubie.random_cube: uniformly random cubes",
                      "batch_sizes": sorted({len(x) for x in batches}, reverse=True),
                      "max_abs_diff": max_diff, "tol": tolerance, "rtol": 0,
                      "onnxruntime_version": ort.__version__, "provider": "CPUExecutionProvider",
                      "reference_dtype": "torch float32"},
        }
        if not is_value:
            contract["move_names"] = list(cube.MOVE_NAMES)
        temporary_contract.write_text(json.dumps(contract, indent=2) + "\n")
        temporary_out.replace(out)
        temporary_contract.replace(out.with_suffix(out.suffix + ".json"))
        return contract
    finally:
        temporary_out.unlink(missing_ok=True)
        temporary_contract.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m rubiks.export_onnx", description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fp16", action="store_true", help="float16 weight storage, float32 I/O and computation; atol 1e-2")
    p.add_argument("--probes", type=int, default=2000, help="uniform random cubes for parity (default 2000)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.checkpoint.is_file():
        p.error("--checkpoint must be a file")
    if args.out.resolve() == args.checkpoint.resolve():
        p.error("--out must differ from --checkpoint")
    if args.probes < 1 or args.seed < 0:
        p.error("--probes must be >= 1 and --seed >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = export_checkpoint(args.checkpoint, args.out, fp16=args.fp16, probes=args.probes, seed=args.seed)
    check = contract["check"]
    print(f"{args.out}: {contract['bytes']} bytes, {contract['weights_dtype']} weights; "
          f"{check['n']} random cubes, max absolute difference {check['max_abs_diff']:.9g} "
          f"<= {check['tol']}, torch fp32 vs ONNX Runtime CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
