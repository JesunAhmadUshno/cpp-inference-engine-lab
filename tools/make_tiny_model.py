#!/usr/bin/env python3
"""Generate the tiny random-weight GPT-2 fixture used by the tests.

Config: vocab 97, width 32, 4 heads, 2 layers, context 16. Weights
are drawn with NumPy seed 20261113 (frozen in PREREGISTRATION.md).
The checkpoint is written in the same safetensors layout and tensor
naming as the real GPT-2 file, including the stored [in][out]
orientation of the Conv1D weights and the (ignored) attn.bias mask
buffers. Expected logits come from tools/reference_logits.py, an
independent NumPy forward implementation, and are written next to
the model as expected_logits.json.

Outputs (tests/fixtures/): model.safetensors, ids.txt,
expected_logits.json. These fixtures are committed; CI does not
need to regenerate them.
"""

import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reference_logits import forward_logits  # noqa: E402

VOCAB, EMBD, HEADS, LAYERS, CTX = 97, 32, 4, 2, 16
IDS = [3, 14, 15, 92, 65, 35, 89, 79]
OUT_DIR = os.path.join(os.path.dirname(HERE), "tests", "fixtures")


def build_weights():
    rng = np.random.default_rng(20261113)

    def mat(i, o):
        return rng.normal(0.0, 0.08, size=(i, o)).astype(np.float32)

    def vec(n, scale=0.05):
        return rng.normal(0.0, scale, size=(n,)).astype(np.float32)

    def gain(n):
        return (1.0 + rng.normal(0.0, 0.05, size=(n,))).astype(np.float32)

    w = {}
    w["wte.weight"] = rng.normal(0.0, 0.08, size=(VOCAB, EMBD)).astype(np.float32)
    w["wpe.weight"] = rng.normal(0.0, 0.08, size=(CTX, EMBD)).astype(np.float32)
    mask = np.tril(np.ones((CTX, CTX), dtype=np.float32))
    for l in range(LAYERS):
        p = f"h.{l}."
        w[p + "ln_1.weight"] = gain(EMBD)
        w[p + "ln_1.bias"] = vec(EMBD)
        w[p + "attn.c_attn.weight"] = mat(EMBD, 3 * EMBD)
        w[p + "attn.c_attn.bias"] = vec(3 * EMBD)
        w[p + "attn.c_proj.weight"] = mat(EMBD, EMBD)
        w[p + "attn.c_proj.bias"] = vec(EMBD)
        w[p + "attn.bias"] = mask.reshape(1, 1, CTX, CTX).copy()
        w[p + "ln_2.weight"] = gain(EMBD)
        w[p + "ln_2.bias"] = vec(EMBD)
        w[p + "mlp.c_fc.weight"] = mat(EMBD, 4 * EMBD)
        w[p + "mlp.c_fc.bias"] = vec(4 * EMBD)
        w[p + "mlp.c_proj.weight"] = mat(4 * EMBD, EMBD)
        w[p + "mlp.c_proj.bias"] = vec(EMBD)
    w["ln_f.weight"] = gain(EMBD)
    w["ln_f.bias"] = vec(EMBD)
    return w


def write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        begin = len(payload)
        payload += arr.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [begin, len(payload)],
        }
    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(payload)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    weights = build_weights()
    model_path = os.path.join(OUT_DIR, "model.safetensors")
    write_safetensors(model_path, weights)
    with open(os.path.join(OUT_DIR, "ids.txt"), "w") as f:
        f.write(" ".join(str(i) for i in IDS) + "\n")
    logits = forward_logits(weights, IDS, EMBD, HEADS)
    expected = {
        "ids": IDS,
        "config": {"vocab": VOCAB, "embd": EMBD, "heads": HEADS,
                   "layers": LAYERS, "ctx": CTX},
        "logits_last": [float(x) for x in logits[-1]],
        "argmax_last": int(np.argmax(logits[-1])),
        "note": "float64 NumPy reference (tools/reference_logits.py) "
                "on the committed tiny fixture; engine fp32 must match "
                "within 2e-4, engine int8 within 0.1 (frozen gates).",
    }
    with open(os.path.join(OUT_DIR, "expected_logits.json"), "w") as f:
        json.dump(expected, f, indent=1)
    print(f"wrote {model_path}")
    print(f"ids {IDS}, reference argmax at last position: "
          f"{expected['argmax_last']}")


if __name__ == "__main__":
    main()
