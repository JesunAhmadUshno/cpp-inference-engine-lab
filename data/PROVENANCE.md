# Data provenance

All token-id data in this directory was produced once by
`tools/prepare_data.py` (see PREREGISTRATION.md, tokenization scope)
and committed. The C++ engine consumes these files; it never tokenizes.

## Model checkpoint (not committed, downloaded at run time)

- Model: GPT-2 small (124M parameters), repo `openai-community/gpt2`
  on Hugging Face, revision 607a30d783dfa663caf39e06633721c8d4cfcd7e.
- File: `model.safetensors`, 548,105,171 bytes.
- SHA-256: 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
- Licence: MIT (licence tag on the Hugging Face model card for
  openai-community/gpt2, checked 2026-11-13). GPT-2 weights were
  released by OpenAI.
- Tokenizer files from the same repo and revision:
  - `vocab.json`, SHA-256 196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
  - `merges.txt`, SHA-256 1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5

## Perplexity text

- Source: `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`,
  split `test`, on Hugging Face (the canonical WikiText-2
  distribution; WikiText is extracted from Wikipedia featured and
  good articles).
- Retrieval: the split's parquet file served by Hugging Face
  (wikitext-2-raw-v1/test/0.parquet, 732,610 bytes,
  SHA-256 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91).
- Frozen text form: the 4,358 rows of the split, in dataset order,
  joined with a single newline character between consecutive rows.
  Result: `assets/test.txt` locally, 1,292,013 bytes,
  SHA-256 aca2f46735043bcfd0a44eca981d04627b9cdf74c4c9a04bf0856d04066f58fc.
  Tokenized in full: 284,614 GPT-2 BPE tokens.
- `ppl_ids.txt` holds the first 8,192 of those token ids.

## Files in this directory

- `ppl_ids.txt`: 8,192 token ids (space separated), the frozen
  perplexity set.
- `prompts.txt`: the 12-prompt suite, verbatim, one prompt per line.
- `prompt_ids.txt`: GPT-2 BPE ids of each prompt, one line per prompt.
- `vocab.tsv`: id, TAB, lowercase hex of the token's raw byte string
  (the GPT-2 byte-decoder mapping applied to the vocab entry). Used by
  the engine's minimal display decode path only.

## Tokenizer tool

- `tools/prepare_data.py`, written for this repo: GPT-2 byte-level BPE
  (bytes-to-unicode alphabet, canonical pre-tokenization regex,
  merges in rank order). Run 2026-11-13 with Python 3.12.3 and the
  `regex` package version 2026.11.9. This tool is a reference
  tokenizer for data preparation; the study makes no claim about it
  beyond producing the frozen ids above.
