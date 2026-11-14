#!/usr/bin/env python3
"""Produce the frozen token-id data files for cpp-inference-engine-lab.

This is the reference tokenizer step described in PREREGISTRATION.md.
Token ids for the perplexity set and the prompt suite are produced ONCE
by this script and committed under data/. The C++ engine itself only
ever consumes and produces token ids (plus a minimal id-to-bytes decode
path via data/vocab.tsv for displaying generations).

The encoder below is an implementation of the GPT-2 byte-pair encoding
procedure written for this repo: byte-level alphabet via the standard
GPT-2 bytes-to-unicode map, the canonical GPT-2 pre-tokenization regex
(applied with the `regex` package, which supports Unicode classes), and
merges applied in rank order from merges.txt. It is a data-preparation
tool, not part of the engine, and the study does not claim it as a
production tokenizer.

Inputs (in ../assets/, downloaded public files; hashes in
data/PROVENANCE.md):
  - vocab.json, merges.txt  from openai-community/gpt2 (Hugging Face)
  - test.txt                wikitext-2 test rows of Salesforce/wikitext
                            (wikitext-2-raw-v1) joined by single newlines

Outputs (in ../data/):
  - ppl_ids.txt    first 8192 token ids of the test text, space separated
  - prompt_ids.txt one line per prompt, space separated ids
  - prompts.txt    the prompt suite, verbatim, one prompt per line
  - vocab.tsv      id <TAB> hex of the token's raw byte string

Requires: python3, the `regex` package (pip install regex).
"""

import json
import os
import struct
import sys

import regex as re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
DATA = os.path.join(ROOT, "data")

PPL_TOKENS = 8192

PROMPTS = [
    "The history of the Roman Empire is",
    "In a small village by the sea, there lived",
    "The quick brown fox jumps over the",
    "Scientists have discovered a new species of",
    "To make a good cup of coffee, you should",
    "The stock market rose sharply after",
    "Once upon a time, in a kingdom far away,",
    "The recipe calls for two cups of flour and",
    "When the train arrived at the station,",
    "Artificial intelligence will change the way we",
    "The old lighthouse keeper watched as the",
    "If you want to learn a new language,",
]

PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Encoder:
    def __init__(self, vocab_path, merges_path):
        with open(vocab_path, encoding="utf-8") as f:
            self.encoder = json.load(f)
        with open(merges_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        merges = [tuple(l.split()) for l in lines[1:] if l.strip()]
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        if not pairs:
            self.cache[token] = token
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text):
        ids = []
        for piece in PAT.findall(text):
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for tok in self.bpe(mapped).split(" "):
                ids.append(self.encoder[tok])
        return ids


def main():
    os.makedirs(DATA, exist_ok=True)
    enc = Encoder(
        os.path.join(ASSETS, "vocab.json"), os.path.join(ASSETS, "merges.txt")
    )

    with open(os.path.join(ASSETS, "test.txt"), encoding="utf-8") as f:
        text = f.read()
    full_ids = enc.encode(text)
    print(f"test text: {len(text)} chars, {len(full_ids)} tokens total")
    ppl_ids = full_ids[:PPL_TOKENS]
    with open(os.path.join(DATA, "ppl_ids.txt"), "w") as f:
        f.write(" ".join(str(i) for i in ppl_ids) + "\n")
    print(f"ppl_ids.txt: {len(ppl_ids)} ids, first 8: {ppl_ids[:8]}")

    with open(os.path.join(DATA, "prompts.txt"), "w") as f:
        for p in PROMPTS:
            f.write(p + "\n")
    with open(os.path.join(DATA, "prompt_ids.txt"), "w") as f:
        for p in PROMPTS:
            ids = enc.encode(p)
            f.write(" ".join(str(i) for i in ids) + "\n")
            print(f"prompt ({len(ids)} ids): {p!r}")

    byte_decoder = {v: k for k, v in enc.byte_encoder.items()}
    id_to_token = {i: t for t, i in enc.encoder.items()}
    with open(os.path.join(DATA, "vocab.tsv"), "w") as f:
        for i in range(len(id_to_token)):
            raw = bytes(byte_decoder[c] for c in id_to_token[i])
            f.write(f"{i}\t{raw.hex()}\n")
    print(f"vocab.tsv: {len(id_to_token)} entries")


if __name__ == "__main__":
    sys.exit(main())
