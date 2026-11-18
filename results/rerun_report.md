# Benchmark re-execution report: cpp-inference-engine-lab

Date: 2026-11-17 to 2026-11-18 (America/Toronto). Purpose: close the
gap left at paper review, where the test suite was re-run and the
checkpoint hash verified but the timed benchmark itself was not
re-executed. This report re-runs the full frozen protocol of
PREREGISTRATION.md (both arms, each as its own process) and compares
every headline number against the committed record
(`results/run_fp32.json`, `results/run_int8.json`). Raw outputs of
this re-execution: `results/rerun_fp32.json`, `results/rerun_int8.json`
(logs: `results/rerun_fp32.log`, `results/rerun_int8.log`). No
committed file was edited; the runs were made in a scratch copy of
the repo and only these new files were added.

## Environment note (read before the throughput numbers)

This is NOT the machine of record. The host CPU is the same model
(AMD EPYC 9D25, 4 vCPUs visible, about 7.9 GiB RAM) and the build is
the frozen one (g++ 13.3.0, CMake Release, -O3, C++20, no
-march=native, 4 engine threads), but the VM is shared and
CPU-throttled (about 1.5 effective cores). Absolute tokens per
second are environment-bound and were expected to differ. The
decision quantities are the ratios and the verdicts under the
frozen pass lines; those are what this report judges.

## Gates before benchmarking

- Fresh build from the repo sources, exactly as README.md documents
  (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`,
  `cmake --build build -j4`): clean build of engine_tests,
  engine_generate, engine_bench.
- Test suite: `./build/engine_tests tests/fixtures` reports
  ALL TESTS PASSED (14 tests, 0 failures). Tiny end-to-end fixture:
  fp32 logits vs the NumPy reference max abs diff 4.17e-07 (frozen
  gate 2e-4; committed record 4.2e-07); int8 max abs diff 1.21e-02
  (frozen gate 0.1; committed record 1.21e-02); KV-cache vs full
  forward max abs diff 0.0.
- Checkpoint verified before benchmarking: `assets/model.safetensors`
  is 548,105,171 bytes with SHA-256
  248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707,
  exactly the size and hash frozen in PREREGISTRATION.md.

## Fresh numbers vs the committed record

Throughput: prompt = first 64 ids of data/ppl_ids.txt, 128 greedy
tokens, one warm-up, 7 timed repetitions (same protocol both runs).

| arm | measure | committed | rerun |
| --- | --- | --- | --- |
| fp32 | rep tok/s | 6.44, 6.55, 4.52, 4.54, 4.99, 6.56, 6.39 | 6.80, 6.71, 6.80, 6.92, 6.57, 6.56, 6.81 |
| fp32 | mean tok/s | 5.713 | 6.739 |
| fp32 | std / median | 0.977 / 6.391 | 0.133 / 6.801 |
| fp32 | VmHWM (MiB) | 566.99 | 567.04 |
| fp32 | weight storage (bytes) | 497,759,232 | 497,759,232 |
| fp32 | perplexity (NLL) | 33.816475 (28,815.4393) | 33.816475 (28,815.4393) |
| int8 | rep tok/s | 3.73, 3.80, 3.06, 2.35, 3.45, 3.54, 3.84 | 3.27, 2.34, 2.32, 3.03, 3.04, 3.26, 2.97 |
| int8 | mean tok/s | 3.395 | 2.893 |
| int8 | std / median | 0.533 / 3.537 | 0.399 / 3.034 |
| int8 | VmHWM (MiB) | 310.89 | 310.97 |
| int8 | weight storage (bytes) | 243,287,040 | 243,287,040 |
| int8 | perplexity (NLL) | 33.815783 (28,815.2719) | 33.815783 (28,815.2719) |

Degeneration suite (descriptive): identical in every particular.
All 12 prompts in both arms produced exactly the same generated
token ids as the committed run, so every rep20, distinct20 and sane
score matches per prompt; suite means fp32 rep20 0.9542, distinct20
0.6125, sane 12 of 12; int8 rep20 0.9792, distinct20 0.5958, sane
12 of 12, in both runs.

## Verdict comparison (frozen rules of PREREGISTRATION.md)

- H1 (speed, CONFIRMED only if mean int8 / mean fp32 tok/s is at
  least 1.50): committed ratio 0.594, rerun ratio 0.429
  (2.893 / 6.739). The ratio's magnitude does not match: on this VM
  the fp32 arm ran faster than the record's load-affected run while
  the int8 arm ran slower in absolute terms, so the gap widened.
  The verdict is unchanged and not close: REFUTED in both runs, with
  the rerun ratio below 1.50, below 1.10 and below 1.0. int8 was
  slower than fp32 on both machines.
- H2 (quality cost, CONFIRMED if the relative perplexity change is
  strictly below 5 percent): committed -0.002 percent, rerun
  -0.002 percent (-0.002046 percent computed from the JSON values in
  both runs). Exact match: both arms' perplexities, and their total
  NLLs over the 8,184 predicted tokens, reproduced to every printed
  digit. Verdict CONFIRMED in both runs.
- H3 (memory, CONFIRMED if VmHWM int8 / fp32 is at most 0.60):
  committed ratio 0.548, rerun ratio 0.548 (0.5484 from the JSON
  values; committed 0.5483). Match at the record's precision; the
  per-arm VmHWM values agree within 0.1 MiB. Verdict CONFIRMED in
  both runs. The loader's weight-storage accounting is exact in
  both arms (ratio 0.489 in both runs).

## Bottom line

All three verdicts reproduce under the frozen pass lines: H1
REFUTED, H2 CONFIRMED, H3 CONFIRMED. Every timing-independent
measure (perplexity, NLL, suite token ids and scores, weight bytes,
VmHWM) reproduces exactly or within 0.1 MiB; only the throughput
ratio's magnitude moved (0.594 to 0.429), in the direction that
strengthens the H1 refutation, because absolute speed is a property
of the machine and this VM is not the machine of record.
