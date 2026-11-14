# Plan: cpp-inference-engine-lab

Pipeline: spec, plan, build, test, review, ship. Pre-registration
before results, always.

1. Freeze: write SPEC.md, PREREGISTRATION.md, data/PROVENANCE.md;
   download the checkpoint and tokenizer files; produce the frozen
   token-id data with tools/prepare_data.py; commit and push the
   freeze before any benchmark code runs.
2. Build the engine: safetensors parser with its JSON reader, tensor
   ops and thread pool, GPT-2 forward (full and KV-cache incremental),
   int8 weight-only quantization path (dequantize on the fly),
   benchmark harness CLI.
3. Test: unit tests against hand-computed values; tiny random-weight
   model end-to-end against an independent NumPy reference
   (tools/make_tiny_model.py, tools/reference_logits.py); KV-cache
   equivalence test.
4. Run the frozen protocol on this machine, both arms as separate
   processes; save raw outputs under results/.
5. Write RESULTS.md with per-hypothesis verdicts by the frozen rules
   and the full numbers; write README.md (research question, method,
   headline results, societal impact). No numeric grade in the repo.
6. Ship: CI (build, unit tests, tiny-model smoke generation) and
   CodeQL green on the results commit; Jev claim_check on the
   RESULTS.md claims through the credential-safe connector route,
   with the spec and results saved outside the repo.
