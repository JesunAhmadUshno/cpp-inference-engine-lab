# Spec: cpp-inference-engine-lab

## Research question

For a small open transformer model run CPU-only, what does int8
weight quantization cost in output quality per unit of throughput
gained, against the fp32 baseline of the same from-scratch C++ engine?

## What is built

A minimal inference engine for GPT-2 small, written from scratch in
C++20 in this repo. Standard library only. Every tensor operation,
the attention mechanism, layernorm, GELU, greedy sampling, the KV
cache, the safetensors weight loader, and the quantization path are
this repo's own code. No inference library is linked, vendored, or
wrapped: no llama.cpp, no GGML, no ONNX Runtime, no PyTorch.

Components:

- `safetensors`: parser for the safetensors container (8-byte little
  endian header length, JSON header, raw tensor payload), including a
  small purpose-written JSON parser for the header structure.
- `tensor`: row-major fp32 tensors, matmul and matvec with the output
  rows split across a fixed thread pool, plus an int8 path that
  dequantizes weights on the fly (see PREREGISTRATION.md for the exact
  frozen scheme).
- `gpt2`: the transformer forward pass (token and position
  embeddings, 12 blocks of pre-LayerNorm attention and MLP, final
  LayerNorm, tied lm head), in two modes: full-sequence forward (for
  perplexity) and incremental decoding with a KV cache (for
  generation).
- `bench`: the benchmark harness implementing the frozen measurement
  protocol: throughput, peak memory, perplexity, and the prompt-suite
  degeneration scores, writing raw results as JSON.
- `tests`: unit tests for the tensor ops against hand-computed
  values, the safetensors parser on a synthetic file, quantization
  round-trip error bounds, KV-cache equivalence, and a tiny
  end-to-end model whose logits are compared against an independent
  NumPy reference implementation of the same math.

## Scope limits (honest)

- The engine consumes and produces token ids. Token ids for the
  measurement sets were produced once by `tools/prepare_data.py` and
  are committed as data with provenance. The only decode path is a
  minimal id-to-bytes lookup for displaying generations.
- One model (GPT-2 small), one machine, one quantization scheme,
  greedy decoding only. Nothing here measures batch serving, GPU
  execution, or production workloads.
- Claims ceiling: study scope only, never production experience.

## Acceptance criteria

- CMake Release build with g++ (C++20) is clean with -Wall -Wextra.
- All unit tests pass locally and in CI; CI also runs a smoke
  generation on a small random-weight model of the same architecture,
  labelled as such.
- PREREGISTRATION.md is committed before any benchmark runs; the
  results commit contains RESULTS.md, the harness, and the raw
  benchmark outputs, with CI green on it.
