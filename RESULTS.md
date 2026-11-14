# Results: cpp-inference-engine-lab

All numbers are from the raw harness outputs `results/run_fp32.json` and `results/run_int8.json`, rendered by `tools/make_results.py`. Protocol: PREREGISTRATION.md (committed before any benchmark run). Verdicts follow the frozen rules exactly.

## Study setup

- Model: GPT-2 small (124M), `openai-community/gpt2` revision 607a30d783dfa663caf39e06633721c8d4cfcd7e, file `model.safetensors` SHA-256 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707. Licence: MIT (Hugging Face model card licence tag).
- Engine: this repo's from-scratch C++20 engine (own safetensors parser, tensor ops, attention, KV cache, quantization path; standard library only).
- Machine: AMD EPYC 9D25 126-Core Processor, 4 vCPUs, about 7.7 GiB RAM, Ubuntu 24.04. Build: g++ 13.3.0, CMake Release, -O3, C++20, no -march=native. 4 engine threads, both arms.
- Arms: fp32 baseline, and W8A32 int8 (weight-only symmetric int8, per-output-channel scales, applied to the 48 block linear matrices; embeddings, biases and LayerNorm parameters stay fp32; dequantize-on-the-fly in the kernels, fp32 activations and accumulation).
- Decoding: greedy everywhere. Perplexity set: first 8,192 GPT-2 BPE tokens of the frozen wikitext-2 test text form (data/PROVENANCE.md), blocks of 1,024, 8,184 predicted tokens.

## Verdicts

- H1 (int8 at least 1.5x tokens/sec): **REFUTED**. Mean throughput ratio is 0.594 (3.395 vs 5.713 tok/s).
- H2 (int8 perplexity increase under 5 percent): **CONFIRMED**. Relative change is -0.002 percent (33.8158 vs 33.8165).
- H3 (int8 peak memory at most 60 percent of fp32): **CONFIRMED**. VmHWM ratio is 0.548 (310.9 vs 567.0 MiB).

## Throughput (generation)

Prompt: first 64 ids of the perplexity set. 128 greedy tokens per repetition, KV cache on, one warm-up, 7 timed repetitions; time includes prefill.

| arm | rep tok/s | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 6.44, 6.55, 4.52, 4.54, 4.99, 6.56, 6.39 | 5.713 | 0.977 | 6.391 | 4.524 | 6.564 |
| int8 | 3.73, 3.80, 3.06, 2.35, 3.45, 3.54, 3.84 | 3.395 | 0.533 | 3.537 | 2.348 | 3.843 |

Mean ratio int8 / fp32: 0.594.

Interpretation (not a measured claim): in this engine the int8 path dequantizes every weight on every use, inside the inner loop. Under the frozen build (baseline x86-64 instruction set, no AVX or VNNI), the per-element int8-to-fp32 conversion costs more than the memory traffic it saves at this model size and batch size 1, so the smaller footprint does not turn into speed here. A true integer dot-product kernel or a wider instruction set could change this; neither was in the frozen design.

## Perplexity

| arm | total NLL | predicted tokens | perplexity |
| --- | --- | --- | --- |
| fp32 | 28815.44 | 8184 | 33.8165 |
| int8 | 28815.27 | 8184 | 33.8158 |

Relative change: -0.002 percent (33.8158 vs 33.8165).

## Peak memory

VmHWM (peak RSS) of the benchmark process, read right after the timed throughput repetitions. Weight storage is the loader's exact byte accounting (the checkpoint's attention mask buffers are not loaded; the file itself is 548,105,171 bytes).

| arm | VmHWM (MiB) | weight storage (bytes) |
| --- | --- | --- |
| fp32 | 566.99 | 497759232 |
| int8 | 310.89 | 243287040 |

VmHWM ratio: 0.548. Weight storage ratio: 0.489.

## Degeneration suite (descriptive, no hypothesis)

12 frozen prompts, 64 greedy tokens each. rep20: fraction of the last 20 generated tokens that repeat a token from the first 44 generated positions. distinct20: unique ids among the last 20, divided by 20. sane: all 64 tokens produced with finite logits.

| prompt | fp32 rep20 | fp32 distinct20 | fp32 sane | int8 rep20 | int8 distinct20 | int8 sane |
| --- | --- | --- | --- | --- | --- | --- |
| The history of the Roman Empire is | 0.95 | 0.75 | True | 1.00 | 0.60 | True |
| In a small village by the sea, there lived | 1.00 | 0.45 | True | 1.00 | 0.45 | True |
| The quick brown fox jumps over the | 1.00 | 0.45 | True | 1.00 | 0.45 | True |
| Scientists have discovered a new species of | 0.75 | 0.75 | True | 1.00 | 0.70 | True |
| To make a good cup of coffee, you should | 1.00 | 0.75 | True | 1.00 | 0.75 | True |
| The stock market rose sharply after | 1.00 | 0.90 | True | 1.00 | 0.90 | True |
| Once upon a time, in a kingdom far away, | 1.00 | 0.25 | True | 1.00 | 0.25 | True |
| The recipe calls for two cups of flour and | 1.00 | 0.70 | True | 1.00 | 0.70 | True |
| When the train arrived at the station, | 0.85 | 0.85 | True | 0.85 | 0.85 | True |
| Artificial intelligence will change the way we | 1.00 | 0.70 | True | 1.00 | 0.70 | True |
| The old lighthouse keeper watched as the | 1.00 | 0.20 | True | 1.00 | 0.20 | True |
| If you want to learn a new language, | 0.90 | 0.60 | True | 0.90 | 0.60 | True |

Suite means: fp32 rep20 0.9542, distinct20 0.6125, sane 12 of 12. int8 rep20 0.9792, distinct20 0.5958, sane 12 of 12.

Both arms loop heavily under greedy decoding (rep20 near 1.0): that is GPT-2 small's known greedy behaviour, and the int8 arm tracks the fp32 arm closely on these mechanical scores. No broader quality claim is made.

## Correctness evidence

- 14 automated tests pass (hand-computed tensor ops, safetensors round trip, quantization error bound, a hand-computed transformer block, causality, KV-cache equivalence with max diff 0.0, and the tiny end-to-end fixture): fp32 logits match the independent float64 NumPy reference within 4.2e-07 (frozen gate 2e-4); int8 within 1.21e-02 (frozen gate 0.1).
- Real-model cross-check: the same NumPy reference, run on the real GPT-2 weights over perplexity block 0, gives perplexity 16.50174; the C++ engine gives 16.5017 for that block. Agreement to 5 significant figures on the production code path.
- CI and CodeQL are green on the commit that adds the engine, and CI runs the tests plus smoke generations on the tiny fixture for both arms.

## Run record

The fp32 arm completed in one run. The int8 arm's first attempt was interrupted by an environment restart after its throughput repetitions (that attempt is preserved as results/run_int8_attempt1_interrupted.log; its seven repetitions were consistent at 4.248 to 5.246 tok/s, mean 4.910, and its VmHWM was 311.1 MiB). It produced no results file, so the arm was re-run in full under the same frozen protocol. The re-run's early throughput repetitions overlapped post-restart load on this shared VM, which is why its repetitions vary more and its mean (3.395 tok/s) sits below the interrupted attempt's clean mean. The H1 verdict is the same under either mean (ratio 0.594 on the completed run, 0.859 on the interrupted attempt's repetitions; both below 1.0, far under the 1.50 line), and the timing-independent measures (perplexity, suite, VmHWM) are unaffected. Every headline number above is from the completed runs only.

## Claims ceiling

Study scope only: one small model (GPT-2 small), one machine, one quantization scheme (W8A32, dequantize on the fly), greedy decoding, a fixed 8,192-token perplexity slice and a 12-prompt suite. Throughput repetitions on this shared VM show visible variance (see the per-repetition values); the means above are the frozen decision quantities. Nothing here measures production serving, batching, GPUs, or other models, and no production experience is claimed.
