// Known-answer check, run as part of `npm run build`.
// Recomputes the three hypothesis verdicts from src/results.json (which is
// generated from the committed raw harness outputs) and fails the build if
// the data does not support the verdicts the dashboard displays:
//   H1 REFUTED   (int8/fp32 throughput ratio below the frozen 1.5x line)
//   H2 CONFIRMED (perplexity relative change inside the frozen 5 percent line)
//   H3 CONFIRMED (peak-memory ratio at or below the frozen 0.60 line)
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const d = JSON.parse(readFileSync(join(here, 'results.json'), 'utf8'));

const fail = (msg) => { console.error('lab.check FAILED:', msg); process.exit(1); };

if (d.fp32.throughput.reps.length !== 7 || d.int8.throughput.reps.length !== 7)
  fail('expected 7 throughput repetitions per arm');
if (d.fp32.suite.promptCount !== 12 || d.int8.suite.promptCount !== 12)
  fail('expected 12 degeneration-suite prompts per arm');
if (d.fp32.perplexity.predictedTokens !== d.int8.perplexity.predictedTokens)
  fail('perplexity predicted-token counts differ between arms');

const h1Refuted = d.derived.throughputRatio < d.frozen.h1MinRatio;
const h2Confirmed = Math.abs(Number(d.derived.perplexityRelChangePctDisplay)) < d.frozen.h2MaxRelPct;
const h3Confirmed = d.derived.vmhwmRatio <= d.frozen.h3MaxRatio;
if (!h1Refuted) fail(`H1 would not be REFUTED: ratio ${d.derived.throughputRatio}`);
if (!h2Confirmed) fail(`H2 would not be CONFIRMED: change ${d.derived.perplexityRelChangePctDisplay}%`);
if (!h3Confirmed) fail(`H3 would not be CONFIRMED: ratio ${d.derived.vmhwmRatio}`);

console.log('lab.check OK: H1 REFUTED, H2 CONFIRMED, H3 CONFIRMED are supported by the committed data');
