# cpp-inference-engine-lab

A from-scratch C++20 transformer inference engine, built to measure
one thing honestly: what int8 weight quantization costs and buys on
CPU.

## Research question

For a small open transformer model run CPU-only, what does int8
quantization cost in output quality per unit of throughput gained,
against the fp32 baseline of the same from-scratch C++ engine?

## The engine

Everything under `include/engine/` is this repo's own code, C++20,
standard library only:

- A safetensors loader that parses the format directly (own minimal
  JSON parser for the header, streaming tensor reads, no second copy
  of the checkpoint in memory).
- Tensor kernels: matvec and matmat in fp32 and in int8 with
  dequantize-on-the-fly, a fixed 4-thread pool, LayerNorm, GELU
  (tanh form), softmax, argmax.
- A GPT-2 forward pass: pre-LayerNorm blocks, causal multi-head
  attention, tied embedding lm head, KV-cache incremental decoding.
- The quantization path: weight-only symmetric int8 with
  per-output-channel scales (W8A32; see PREREGISTRATION.md).

No inference library is linked, vendored or wrapped: no llama.cpp,
no GGML, no ONNX Runtime, no PyTorch. Tokenization scope is stated
honestly in PREREGISTRATION.md: the engine consumes token ids, the
frozen id files were produced once by a reference GPT-2 BPE
tokenizer (`tools/prepare_data.py`), and a minimal id-to-text decode
path exists for displaying generations.

## Method

GPT-2 small (124M, MIT licence) from `openai-community/gpt2`,
checkpoint pinned by revision and SHA-256 in PREREGISTRATION.md,
which was committed before any benchmark ran. Three hypotheses were
frozen with exact decision rules: H1 int8 at least 1.5x tokens per
second; H2 int8 perplexity increase under 5 percent on a frozen
8,192-token wikitext-2 test slice; H3 int8 peak memory at most 60
percent of fp32. Plus a 12-prompt degeneration suite scored by a
frozen mechanical rule. Full protocol: PREREGISTRATION.md. Data
provenance: data/PROVENANCE.md.

## Headline results (full numbers in RESULTS.md)

- H1 REFUTED. int8 was slower, not faster: mean 3.395 tok/s against
  fp32 5.713 (ratio 0.594) on the completed run, and 4.910 tok/s in
  the interrupted first attempt's consistent repetitions. The
  dequantize-on-the-fly inner loop costs more than the bandwidth it
  saves under this build. Reported as measured.
- H2 CONFIRMED with room to spare. Perplexity 33.8158 (int8) against
  33.8165 (fp32): a relative change of -0.002 percent on 8,184
  predicted tokens. The quality cost of W8A32 here is effectively
  zero.
- H3 CONFIRMED. Peak memory (VmHWM) 310.9 MiB against 567.0 MiB,
  ratio 0.548. Weight storage halves: 243,287,040 against
  497,759,232 bytes.
- Degeneration suite: both arms loop heavily under greedy decoding
  (mean rep20 0.954 fp32, 0.979 int8) and stay sane on all 12
  prompts; the int8 profile tracks the fp32 profile closely.

Correctness is gated, not asserted: 14 automated tests, including a
tiny end-to-end fixture whose fp32 logits match an independent
float64 NumPy reference within 4.2e-07, and a real-model cross-check
where the same reference reproduces the engine's block-0 perplexity
to 5 significant figures.

## Why this matters

Capable AI that runs on cheap, offline hardware is what makes the
technology usable in low-resource settings: rural clinics, schools
without reliable connectivity, field work, and places where a GPU
server is not an option. Knowing exactly what quantization buys
(half the memory, no measurable quality loss here) and what it does
not automatically buy (speed, unless the kernel is built for it) is
the kind of measurement that decides whether a small model is
deployable on the hardware people actually have.

## Local run protocol

The full model is not downloaded in CI (548 MB); CI builds the
engine, runs the 14 tests, and smoke-tests generation on a tiny
random-weight GPT-2 of the same architecture (tests/fixtures).
To reproduce the study locally:

    # fetch the pinned checkpoint (see PREREGISTRATION.md for the
    # revision and SHA-256) into assets/model.safetensors
    curl -L --retry 5 -o assets/model.safetensors \
      "https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j4
    ./build/engine_tests tests/fixtures
    ./build/engine_bench --model assets/model.safetensors --arm fp32 --data data --out results/run_fp32.json
    ./build/engine_bench --model assets/model.safetensors --arm int8 --data data --out results/run_int8.json
    python3 tools/make_results.py

Reference machine for the published numbers: AMD EPYC 9D25
126-Core Processor, 4 vCPUs, about 7.7 GiB RAM, Ubuntu 24.04,
g++ 13.3.0, -O3, no -march=native, 4 engine threads. Expect roughly
20 to 35 minutes per arm.

## Usage: the engine_generate tool

`engine_generate` is the command-line front end: it loads a
checkpoint in either arm and generates greedily from a prompt.
`tools/generate.py` wraps it for text prompts.

Prerequisites: a C++20 compiler and CMake 3.20+ (Python 3 only for
the text wrapper). No other dependencies.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
# optional: install engine_generate and engine_bench plus the headers
cmake --install build --prefix ~/.local
./build/engine_tests tests/fixtures
```

Try it with no download, on the tiny random-weight fixture model
(random weights, so the ids are meaningless; this checks the
pipeline end to end):

```sh
./build/engine_generate --model tests/fixtures/model.safetensors \
    --heads 4 --prompt-ids "3 14 15" --tokens 8
# prompt_ids: 3 14 15
# generated_ids: 38 38 38 38 38 38 38 38
```

For real generation, fetch the pinned GPT-2 checkpoint and the two
tokenizer files (same repo and revision as the study; SHA-256 hashes
in data/PROVENANCE.md):

```sh
REV=607a30d783dfa663caf39e06633721c8d4cfcd7e
curl -L --retry 5 -o assets/model.safetensors \
  "https://huggingface.co/openai-community/gpt2/resolve/$REV/model.safetensors"
curl -L --retry 5 -o assets/vocab.json \
  "https://huggingface.co/openai-community/gpt2/resolve/$REV/vocab.json"
curl -L --retry 5 -o assets/merges.txt \
  "https://huggingface.co/openai-community/gpt2/resolve/$REV/merges.txt"
```

Then generate from a text prompt:

```sh
python3 tools/generate.py --prompt "The history of the Roman Empire is" \
    --tokens 40
```

Output (measured on this machine; speed varies by host):

```
prompt_ids: 464 2106 286 262 7993 8065 318
generated_ids: 257 13899 530 13 383 7993 8065 373 9393 416 262 22482 11 290 262 7993 8065 373 9393 416 262 22482 13 383 22482 547 262 717 284 4474 257 1181 286 1175 11 290 262 22482 547 262
text: The history of the Roman Empire is a fascinating one. The Roman Empire was founded by the Romans, and the Roman Empire was founded by the Romans. The Romans were the first to establish a state of war, and the Romans were the
```

Add `--arm int8` for the quantized path; its continuation tracks the
fp32 one closely before drifting, which is the H2 result in miniature.
The wrapper encodes the prompt with `tools/gpt2_tokenizer.py`, a
standard-library-only GPT-2 tokenizer written for this repo and
validated against the study's frozen artifacts: it reproduces all 12
prompt id lines in data/prompt_ids.txt and the first 8,192 ids of
data/ppl_ids.txt exactly (`python3 tools/generate.py --self-test`
runs the prompt check). The engine itself still consumes token ids
only; `engine_generate --prompt-ids "..."` (or `--prompt-file`) is
the direct route.

## Claims ceiling

Study scope only: one model, one machine, one quantization scheme,
greedy decoding, a fixed perplexity slice and prompt suite. Refuted
results are reported as refuted. No production serving, batching,
GPU, or other-model claims are made, and no production experience
is claimed.

## Layout

    include/engine/   the engine (safetensors, tensors, GPT-2, quantization)
    src/bench.cpp     the benchmark harness (frozen protocol)
    src/generate.cpp  the engine_generate command-line tool (see Usage)
    tests/            unit and end-to-end tests, tiny fixtures
    tools/            data preparation, tiny-model and reference tools, results renderer,
                      the stdlib GPT-2 tokenizer and the generate.py text front end
    data/             frozen token data with provenance
    results/          raw harness outputs and logs

## Licence

Code: Apache-2.0 (LICENSE). Model weights: GPT-2, MIT licence
(Hugging Face model card tag for openai-community/gpt2). Data:
wikitext-2 (CC BY-SA 4.0 and GFDL, per the dataset card).
