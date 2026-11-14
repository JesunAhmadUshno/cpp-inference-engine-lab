// Generates src/results.json from the committed raw harness outputs
// (../../results/run_fp32.json and ../../results/run_int8.json).
// Every number on the site comes from those files; nothing is typed here
// except frozen protocol lines (from PREREGISTRATION.md) and display rounding.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const load = (name) => JSON.parse(readFileSync(join(here, '..', '..', 'results', name), 'utf8'));

// Round half up to d decimals, returned as a fixed-decimal string,
// matching the presentation rounding used in RESULTS.md.
const rhu = (x, d) => {
  const p = 10 ** d;
  const v = Math.round((Math.abs(x) + Number.EPSILON) * p) / p;
  return (x < 0 ? -v : v).toFixed(d);
};
const num = (x, d) => Number(rhu(x, d));
const commas = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

function arm(raw) {
  const t = raw.throughput;
  const suite = raw.suite.prompts.map((p) => ({
    prompt: p.prompt, rep20: p.rep20, distinct20: p.distinct20, sane: p.sane,
  }));
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  return {
    throughput: {
      promptTokens: t.prompt_tokens,
      generatedTokens: t.generated_tokens,
      reps: t.repetitions_tok_per_s,
      display: {
        reps: t.repetitions_tok_per_s.map((v) => rhu(v, 2)),
        mean: rhu(t.mean, 3), std: rhu(t.std, 3), median: rhu(t.median, 3),
        min: rhu(t.min, 3), max: rhu(t.max, 3),
      },
      meanRaw: t.mean,
    },
    weightStorageBytes: raw.weight_storage_bytes,
    weightStorageDisplay: commas(raw.weight_storage_bytes),
    vmhwmMib: raw.vmhwm_mib_after_throughput,
    vmhwmDisplay: rhu(raw.vmhwm_mib_after_throughput, 2),
    loadSeconds: raw.load_seconds,
    perplexity: {
      tokensInSet: raw.perplexity.tokens_in_set,
      predictedTokens: raw.perplexity.predicted_tokens,
      totalNllDisplay: rhu(raw.perplexity.total_nll, 2),
      valueDisplay: rhu(raw.perplexity.perplexity, 4),
      valueRaw: raw.perplexity.perplexity,
    },
    suite: {
      prompts: suite,
      rep20MeanDisplay: rhu(mean(suite.map((p) => p.rep20)), 4),
      distinct20MeanDisplay: rhu(mean(suite.map((p) => p.distinct20)), 4),
      saneCount: suite.filter((p) => p.sane).length,
      promptCount: suite.length,
    },
  };
}

const fp32 = arm(load('run_fp32.json'));
const int8 = arm(load('run_int8.json'));

const derived = {
  throughputRatio: num(int8.throughput.meanRaw / fp32.throughput.meanRaw, 3),
  throughputRatioDisplay: rhu(int8.throughput.meanRaw / fp32.throughput.meanRaw, 3),
  perplexityRelChangePctDisplay: rhu(
    ((int8.perplexity.valueRaw - fp32.perplexity.valueRaw) / fp32.perplexity.valueRaw) * 100, 3),
  vmhwmRatio: num(int8.vmhwmMib / fp32.vmhwmMib, 3),
  vmhwmRatioDisplay: rhu(int8.vmhwmMib / fp32.vmhwmMib, 3),
  weightStorageRatioDisplay: rhu(int8.weightStorageBytes / fp32.weightStorageBytes, 3),
};

// Frozen decision lines from PREREGISTRATION.md (protocol constants, not measurements).
const frozen = { h1MinRatio: 1.5, h2MaxRelPct: 5, h3MaxRatio: 0.6 };

const out = { fp32, int8, derived, frozen };
writeFileSync(join(here, '..', 'src', 'results.json'), JSON.stringify(out, null, 2) + '\n');
console.log('wrote src/results.json',
  'throughputRatio', derived.throughputRatioDisplay,
  'pplChangePct', derived.perplexityRelChangePctDisplay,
  'vmhwmRatio', derived.vmhwmRatioDisplay);
