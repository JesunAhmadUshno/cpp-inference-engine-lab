#!/usr/bin/env python3
"""Stdlib-only GPT-2 byte-pair encoder and decoder for the CLI tools.

tools/prepare_data.py produced the study's frozen token files with an
encoder built on the third-party `regex` package. This module is the
same procedure (byte-level alphabet, the canonical GPT-2
pre-tokenization pattern, merges in rank order) implemented with the
Python standard library only: Unicode letter and number classes come
from unicodedata instead of regex character classes.

It is validated against the study's own frozen artifacts, not against
its own say-so: encoding the 12 suite prompts must reproduce
data/prompt_ids.txt exactly, and encoding the wikitext-2 test text
must reproduce the first 8,192 ids of data/ppl_ids.txt exactly.
tools/generate.py --self-test runs the prompt half of that check; the
full check was run at packaging time and is recorded in the README.

Scope: a command-line convenience for this repo's tools, held to the
same honesty rule as prepare_data.py. It is not claimed as a
production tokenizer.
"""

import json
import os
import unicodedata

CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _class(ch):
    if ch.isspace():
        return "space"
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        return "L"
    if cat.startswith("N"):
        return "N"
    return "other"


def pretokenize(text):
    """Split text per the GPT-2 pattern:
    's|'t|'re|'ve|'m|'ll|'d| ?L+| ?N+| ?[^\\sLN]+|\\s+(?!\\S)|\\s+
    """
    pieces = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "'":
            hit = None
            for c in CONTRACTIONS:
                if text.startswith(c, i):
                    hit = c
                    break
            if hit is not None:
                pieces.append(hit)
                i += len(hit)
                continue
        if ch.isspace():
            if i + 1 < n and not text[i + 1].isspace():
                cls = _class(text[i + 1])
                j = i + 1
                while j < n and _class(text[j]) == cls:
                    j += 1
                pieces.append(text[i:j])
                i = j
                continue
            j = i
            while j < n and text[j].isspace():
                j += 1
            if j == n:
                pieces.append(text[i:j])
                i = j
            else:
                pieces.append(text[i:j - 1])
                i = j - 1
            continue
        cls = _class(ch)
        j = i
        while j < n and _class(text[j]) == cls:
            j += 1
        pieces.append(text[i:j])
        i = j
    return pieces


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
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
        merges = [tuple(line.split()) for line in lines[1:] if line.strip()]
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.decoder = {}
        for tok, idx in self.encoder.items():
            self.decoder[idx] = bytes(self.byte_decoder[c] for c in tok)
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
            bigram = min(pairs,
                         key=lambda p: self.bpe_ranks.get(p, float("inf")))
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
                if (word[i] == first and i < len(word) - 1
                        and word[i + 1] == second):
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
        for piece in pretokenize(text):
            mapped = "".join(self.byte_encoder[b]
                             for b in piece.encode("utf-8"))
            for tok in self.bpe(mapped).split(" "):
                ids.append(self.encoder[tok])
        return ids

    def decode(self, ids):
        return b"".join(self.decoder.get(i, b"") for i in ids)


def default_encoder(root=None):
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return Encoder(os.path.join(root, "assets", "vocab.json"),
                   os.path.join(root, "assets", "merges.txt"))
