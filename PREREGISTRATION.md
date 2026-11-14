# Pre-registration: cpp-inference-engine-lab

Frozen before the engine runs any benchmark. The git commit that
introduces this file is the receipt: it precedes all benchmark runs
in this repo. Any change to this file after results exist must be
recorded as an amendment with a date and a reason, never a silent
edit.

## Research question

For a small open transformer model run CPU-only, what does int8
weight quantization cost in output quality per unit of throughput
gained, against the fp32 baseline of the same from-scratch C++ engine?

## Model (exact)

- GPT-2 small: 12 layers, 12 heads, embedding width 768, context
  1024, vocabulary 50,257, about 124M parameters.
- Source: Hugging Face repo `openai-community/gpt2`, revision
  607a30d783dfa663caf39e06633721c8d4cfcd7e, file
  `model.safetensors`, 548,105,171 bytes, SHA-256
  248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707.
- Licence: MIT (licence tag on the Hugging Face model card for
  openai-community/gpt2, checked 2026-11-13). Weights released by
  OpenAI.
- Architecture details as implemented: pre-LayerNorm blocks,
  LayerNorm epsilon 1e-5, GELU in the tanh approximation form
  (gelu_new, the Hugging Face GPT-2 default activation), learned
  position embeddings, causal attention with scale 1/sqrt(64) per
  head, lm head tied to the token embedding matrix.
- Loader route: the engine parses the safetensors file directly
  (own parser: 8-byte little-endian header length, JSON header via
  a purpose-written minimal JSON parser, raw payload). No converted
  checkpoint is used. The checkpoint's `h.*.attn.bias` mask buffers
  are ignored; causality is applied by the engine itself. There is
  no separate lm_head tensor in the file; the tied embedding is
  used, as in the reference architecture.

## Tokenization scope (frozen, honest)

The engine consumes and produces token ids only. Token ids for the
perplexity set and the prompt suite were produced once by
`tools/prepare_data.py` (written for this repo; GPT-2 byte-level
BPE with the canonical pre-tokenization regex, run with Python
3.12.3 and the `regex` package 2026.11.9) from `vocab.json` and
`merges.txt` of the same model repo and revision, and are committed
under `data/` with provenance in `data/PROVENANCE.md`. Displaying a
generation uses a minimal id-to-bytes decode via `data/vocab.tsv`.
No from-scratch BPE is claimed for the engine itself; the data-prep
encoder is a preparation tool, stated as such.

## Arms

- Arm A (fp32 baseline): all weights fp32, exactly as loaded.
- Arm B (int8): weight-only symmetric int8 quantization, frozen
  scheme:
  - Per-output-channel scales: one fp32 scale per output channel
    (per row of the weight matrix in the orientation the matmul
    consumes it, that is per column of the stored Conv1D matrices),
    scale = max|w| / 127 over that channel; values rounded to
    nearest integer and clamped to [-127, 127].
  - Quantized tensors: every `h.*.attn.c_attn.weight`,
    `h.*.attn.c_proj.weight`, `h.*.mlp.c_fc.weight`,
    `h.*.mlp.c_proj.weight` (48 matrices, about 85M weights).
  - Excluded, kept fp32 in both arms: token embeddings `wte.weight`
    (and the tied lm head), position embeddings `wpe.weight`, all
    biases, all LayerNorm parameters.
  - Execution: dequantize-on-the-fly. The matmul and matvec kernels
    read int8 weights and multiply each by its channel scale inside
    the inner loop; activations stay fp32 and accumulation is fp32.
    There is no integer matmul and no activation quantization.
    The arm is named W8A32 (weights int8, activations fp32,
    dequantize-on-the-fly) wherever a short name is needed.
- Both arms run the same engine code, the same build, and the same
  machine, each arm as its own process invocation.

## Machine and build (frozen)

- Machine: AMD EPYC 9D25 126-Core Processor, 4 vCPUs visible to
  this VM, about 7.7 GiB RAM, Ubuntu 24.04.
- Compiler: g++ 13.3.0, CMake Release, flags -O3, C++20, no
  -march=native (portability; identical flags for every arm, so
  ratios are fair).
- Threads: exactly 4 worker threads in the engine's thread pool
  (one per vCPU), splitting matmul output rows. Same thread count
  for both arms and for every measure.

## Measures and protocols (frozen)

### Throughput (generation)

- Prompt: the first 64 token ids of `data/ppl_ids.txt`.
- Greedy decoding (argmax), KV cache on, generate exactly 128 new
  tokens. Temperature and sampling are not used anywhere in this
  study.
- One full untimed warm-up generation, then 7 timed repetitions.
  A repetition's time is the wall time of the whole generation
  call, prefill included; tokens per second = 128 / time.
- Report per arm: the 7 repetition values, mean, standard
  deviation, median, min, max.

### Peak memory

- Read from the same throughput process: `/proc/self/status` VmHWM
  (peak resident set size) after the 7 timed repetitions complete,
  reported in MiB.
- Also reported: exact weight-storage bytes held by the engine per
  arm (sum over tensors of stored bytes, int8 payload plus fp32
  scales for Arm B), as accounted by the loader.
- H3 is judged on the VmHWM ratio, not on the storage accounting.

### Perplexity

- Set: `data/ppl_ids.txt`, the first 8,192 GPT-2 BPE token ids of
  the frozen wikitext-2 test text form defined in
  `data/PROVENANCE.md` (Salesforce/wikitext, wikitext-2-raw-v1,
  test split; full text 284,614 tokens, this study uses the first
  8,192).
- Protocol: non-overlapping blocks of 1,024 tokens (8 blocks). In
  each block, every position after the first is predicted from its
  in-block context: 1,023 predicted tokens per block, 8,184
  predicted tokens in total. Negative log likelihood is summed in
  double precision; perplexity = exp(total NLL / 8,184).
- Full-sequence forward pass (no KV cache), one pass per arm.

### Degeneration (prompt suite)

- Prompts: the 12 prompts of `data/prompts.txt`, verbatim, ids in
  `data/prompt_ids.txt`. Greedy decoding, generate exactly 64 new
  tokens per prompt, KV cache on.
- Frozen mechanical scores, per prompt:
  - rep20: among the last 20 generated tokens, the fraction whose
    token id also occurs at any of the first 44 generated
    positions of the same generation.
  - distinct20: unique token ids among the last 20 generated
    tokens, divided by 20.
  - sane: 1 if the generation produced all 64 tokens and every
    logit vector stayed finite (no NaN, no Inf), else 0.
- Suite score per arm: mean rep20, mean distinct20, and the count
  of sane generations out of 12. Reported per prompt and per arm.
  No hypothesis is attached to these scores; they are descriptive,
  reported as measured, and no quality claim beyond them is made.

## Hypotheses and verdict rules (frozen)

- H1 (speed): int8 gives at least 1.5x tokens per second over fp32.
  CONFIRMED if mean int8 tokens/sec divided by mean fp32
  tokens/sec is at least 1.50 under the throughput protocol.
  REFUTED if the ratio is below 1.50. INCONCLUSIVE only if a timed
  run cannot execute at all, with the failure recorded.
- H2 (quality cost): int8 perplexity increase stays under 5
  percent relative to fp32. CONFIRMED if (PPL int8 - PPL fp32) /
  PPL fp32 is strictly below 0.05. REFUTED if it is at least 0.05.
  INCONCLUSIVE only if a perplexity pass cannot execute, with the
  failure recorded.
- H3 (memory): int8 peak memory is at most 60 percent of fp32.
  CONFIRMED if VmHWM int8 divided by VmHWM fp32 is at most 0.60.
  REFUTED if it is above 0.60. INCONCLUSIVE only if a memory read
  cannot be taken, with the failure recorded.

## Correctness gates before benchmarking (frozen)

Benchmark numbers are only valid if, on the same build:

- Unit tests pass: tensor ops against hand-computed values,
  layernorm, GELU, softmax, a hand-computed tiny attention case,
  the safetensors parser on a synthetic file, quantization
  round-trip max error at most scale/2 per channel, and KV-cache
  incremental decoding matching full-forward logits on a tiny model
  within 1e-5 absolute.
- Tiny end-to-end test: a random-weight tiny GPT-2 (vocab 97,
  width 32, 4 heads, 2 layers, context 16; generated by
  tools/make_tiny_model.py with NumPy seed 20261113) whose
  final-position logits from this engine match an independent
  NumPy forward implementation of the same math
  (tools/reference_logits.py) within 2e-4 absolute on the fp32
  path, and within 0.1 absolute on the int8 path (quantization
  error is expected there; the bound is the gate).

## Reporting

- Raw outputs of every benchmark run are saved under `results/` as
  JSON. RESULTS.md renders the numbers from those files; verdicts
  publish either way, including REFUTED and INCONCLUSIVE.
- No operating point outside this protocol may be reported as a
  result; exploratory observations, if any, are labelled
  exploratory and carry no verdict weight.
- Claims ceiling: study scope only (one small model, one machine,
  one quantization scheme, greedy decoding), never production
  experience.
