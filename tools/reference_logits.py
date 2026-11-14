#!/usr/bin/env python3
"""Independent NumPy reference forward pass for GPT-2.

Used only for the tiny end-to-end test fixture: it implements the
same math as the C++ engine (pre-LayerNorm blocks, tanh-approximation
GELU, causal attention, tied embeddings) in float64, in the stored
checkpoint orientation, as a cross-check of the engine's fp32 path.
It shares no code with the engine.

Can also be run directly: python3 tools/reference_logits.py
MODEL.safetensors IDS... prints the last-position logits.
"""

import json
import struct
import sys

import numpy as np


def read_safetensors(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        payload = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        arr = np.frombuffer(payload[b:e], dtype=np.float32)
        out[name] = arr.reshape(meta["shape"]).astype(np.float64)
    return out


def layernorm(x, gain, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gain + bias


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def forward_logits(w, ids, embd, heads):
    dh = embd // heads
    t = len(ids)
    x = w["wte.weight"][ids] + w["wpe.weight"][:t]
    layers = 0
    while f"h.{layers}.ln_1.weight" in w:
        layers += 1
    for l in range(layers):
        p = f"h.{l}."
        h = layernorm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        # Stored Conv1D orientation [in][out]: y = h @ W + b.
        qkv = h @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
        q, k, v = qkv[:, :embd], qkv[:, embd:2 * embd], qkv[:, 2 * embd:]
        attn = np.zeros_like(x)
        for hd in range(heads):
            sl = slice(hd * dh, (hd + 1) * dh)
            scores = q[:, sl] @ k[:, sl].T / np.sqrt(dh)
            mask = np.triu(np.ones((t, t), dtype=bool), k=1)
            scores = np.where(mask, -1e30, scores)
            scores = scores - scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs = probs / probs.sum(axis=-1, keepdims=True)
            attn[:, sl] = probs @ v[:, sl]
        x = x + attn @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        h = layernorm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        f = gelu_tanh(h @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"])
        x = x + f @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
    h = layernorm(x, w["ln_f.weight"], w["ln_f.bias"])
    return h @ w["wte.weight"].T


if __name__ == "__main__":
    weights = read_safetensors(sys.argv[1])
    ids = [int(x) for x in sys.argv[2:]]
    embd = weights["wte.weight"].shape[1]
    logits = forward_logits(weights, ids, embd, heads=4)
    print([round(float(x), 6) for x in logits[-1]])
