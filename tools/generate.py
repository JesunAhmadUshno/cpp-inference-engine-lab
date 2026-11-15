#!/usr/bin/env python3
"""Text-prompt front end for engine_generate.

Encodes a text prompt with the repo's stdlib GPT-2 tokenizer
(tools/gpt2_tokenizer.py, validated against the frozen data files),
runs the C++ engine_generate binary, and prints its output. The
engine itself still consumes token ids only; this script is the
tokenizer step, exactly as tools/prepare_data.py was for the study.

Needs assets/vocab.json and assets/merges.txt (download commands in
the README) and a built engine_generate binary.

Usage:
    python3 tools/generate.py --prompt "The history of the Roman Empire is"
    python3 tools/generate.py --prompt "..." --arm int8 --tokens 64
    python3 tools/generate.py --self-test
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gpt2_tokenizer import default_encoder  # noqa: E402


def self_test():
    enc = default_encoder(ROOT)
    prompts = [line.rstrip("\n")
               for line in open(os.path.join(ROOT, "data", "prompts.txt"))
               if line.strip()]
    want = [[int(x) for x in line.split()]
            for line in open(os.path.join(ROOT, "data", "prompt_ids.txt"))
            if line.strip()]
    bad = 0
    for prompt, expected in zip(prompts, want):
        if enc.encode(prompt) != expected:
            bad += 1
            print(f"MISMATCH: {prompt!r}")
    if bad == 0 and len(prompts) == len(want) == 12:
        print("self-test: all 12 frozen prompt encodings match "
              "data/prompt_ids.txt")
        return 0
    print(f"self-test: FAILED ({bad} mismatches, {len(prompts)} prompts)")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", help="text prompt to continue")
    ap.add_argument("--model", default=os.path.join(ROOT, "assets",
                                                    "model.safetensors"))
    ap.add_argument("--arm", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--bin", default=os.path.join(ROOT, "build",
                                                  "engine_generate"))
    ap.add_argument("--self-test", action="store_true",
                    help="check the tokenizer against the frozen prompts")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.prompt:
        ap.error("--prompt is required (or use --self-test)")

    enc = default_encoder(ROOT)
    ids = enc.encode(args.prompt)
    cmd = [args.bin, "--model", args.model, "--arm", args.arm,
           "--heads", str(args.heads), "--tokens", str(args.tokens),
           "--prompt-ids", " ".join(str(i) for i in ids),
           "--vocab", os.path.join(ROOT, "data", "vocab.tsv")]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
