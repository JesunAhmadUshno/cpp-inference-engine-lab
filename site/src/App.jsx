import data from './results.json';

const REPO = 'https://github.com/JesunAhmadUshno/cpp-inference-engine-lab';
const RESULTS_URL = REPO + '/blob/main/RESULTS.md';
const PREREG_URL = REPO + '/blob/main/PREREGISTRATION.md';

const { fp32, int8, derived, frozen } = data;

function VerdictCard({ id, title, verdict, good, measured, line }) {
  return (
    <div className="tile verdict">
      <div className="vhead">
        <span className="mono vlabel">{id}</span>
        <span className={'pill ' + (good ? 'good' : 'bad')}>{verdict}</span>
      </div>
      <h3>{title}</h3>
      <div className="num">{measured}</div>
      <div className="cap">{line}</div>
    </div>
  );
}

function RepChart() {
  const W = 680, H = 230, base = 190, top = 16;
  const max = Math.max(...fp32.throughput.reps, ...int8.throughput.reps) * 1.15;
  const y = (v) => base - (v / max) * (base - top);
  const groupW = (W - 40) / fp32.throughput.reps.length;
  const barW = Math.min(26, groupW / 2 - 6);
  const bars = [];
  fp32.throughput.reps.forEach((v, i) => {
    const cx = 30 + i * groupW + groupW / 2;
    bars.push(
      <g key={'f' + i}>
        <rect x={cx - barW - 2} y={y(v)} width={barW} height={base - y(v)} rx="3" fill="var(--primary)" />
        <rect x={cx + 2} y={y(int8.throughput.reps[i])} width={barW} height={base - y(int8.throughput.reps[i])} rx="3" fill="var(--warn)" />
        <text x={cx} y={base + 18} textAnchor="middle">rep {i + 1}</text>
      </g>
    );
  });
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Throughput per repetition, tokens per second, fp32 and int8">
        <line x1="30" y1={base} x2={W - 10} y2={base} className="axis" />
        {bars}
      </svg>
      <div className="legend">
        <span><i style={{ background: 'var(--primary)' }} />fp32 baseline</span>
        <span><i style={{ background: 'var(--warn)' }} />int8 (W8A32)</span>
      </div>
    </div>
  );
}

export default function App() {
  const t = [
    ['Mean', fp32.throughput.display.mean, int8.throughput.display.mean],
    ['Std dev', fp32.throughput.display.std, int8.throughput.display.std],
    ['Median', fp32.throughput.display.median, int8.throughput.display.median],
    ['Min', fp32.throughput.display.min, int8.throughput.display.min],
    ['Max', fp32.throughput.display.max, int8.throughput.display.max],
  ];
  return (
    <div className="wrap">
      <nav className="top">
        <a className="brand" href={REPO}>cpp-inference-engine-lab</a>
        <div className="links">
          <a href="#verdicts">Verdicts</a>
          <a href="#throughput">Throughput</a>
          <a href="#quality">Quality and memory</a>
          <a href="#suite">Degeneration suite</a>
          <a href={REPO}>GitHub repo</a>
        </div>
      </nav>

      <section className="card hero">
        <p className="eyebrow">Measured study, from-scratch C++20 engine</p>
        <h1>What does int8 quantization cost, per unit of throughput gained?</h1>
        <p className="sub">
          Research question: for a small open transformer model run CPU-only, what does int8
          quantization cost in output quality per unit of throughput gained, against the fp32
          baseline of the same from-scratch C++ engine? One engine, two arms: fp32, and W8A32
          (weight-only symmetric int8, per-output-channel scales, dequantize-on-the-fly).
          Hypotheses and decision lines were frozen in{' '}
          <a href={PREREG_URL}>PREREGISTRATION.md</a> before any benchmark ran.
        </p>
        <div className="actions">
          <a className="btn primary" href={RESULTS_URL}>Read RESULTS.md</a>
          <a className="btn ghost" href={REPO}>GitHub repository</a>
        </div>
        <div className="tiles">
          <div className="tile">
            <div className="num">{derived.throughputRatioDisplay}x</div>
            <div className="cap">int8 / fp32 throughput ratio (line: at least {frozen.h1MinRatio}x)</div>
          </div>
          <div className="tile">
            <div className="num">{derived.perplexityRelChangePctDisplay}%</div>
            <div className="cap">int8 perplexity change vs fp32 (line: under {frozen.h2MaxRelPct} percent)</div>
          </div>
          <div className="tile">
            <div className="num">{derived.vmhwmRatioDisplay}</div>
            <div className="cap">int8 / fp32 peak memory ratio (line: at most {frozen.h3MaxRatio.toFixed(2)})</div>
          </div>
          <div className="tile">
            <div className="num">{fp32.perplexity.predictedTokens.toLocaleString('en-US')}</div>
            <div className="cap">predicted tokens in the frozen perplexity set, both arms</div>
          </div>
        </div>
      </section>

      <section id="verdicts" className="card">
        <div className="head">
          <div>
            <p className="eyebrow">Hypothesis verdicts</p>
            <h2>Frozen rules, measured outcomes</h2>
          </div>
          <span className="pill info">Verdicts follow the frozen rules exactly</span>
        </div>
        <div className="grid3">
          <VerdictCard
            id="H1" verdict="REFUTED" good={false}
            title="int8 at least 1.5x tokens/sec"
            measured={`${derived.throughputRatioDisplay}x measured`}
            line={`int8 mean ${int8.throughput.display.mean} tok/s vs fp32 mean ${fp32.throughput.display.mean} tok/s. Frozen line: at least ${frozen.h1MinRatio}x. The int8 arm is slower in this engine, and the verdict says so.`}
          />
          <VerdictCard
            id="H2" verdict="CONFIRMED" good={true}
            title="int8 perplexity increase under 5 percent"
            measured={`${derived.perplexityRelChangePctDisplay}% measured`}
            line={`int8 perplexity ${int8.perplexity.valueDisplay} vs fp32 ${fp32.perplexity.valueDisplay}. Frozen line: under ${frozen.h2MaxRelPct} percent relative increase.`}
          />
          <VerdictCard
            id="H3" verdict="CONFIRMED" good={true}
            title="int8 peak memory at most 60 percent of fp32"
            measured={`${derived.vmhwmRatioDisplay} measured`}
            line={`int8 peak ${int8.vmhwmDisplay} MiB vs fp32 ${fp32.vmhwmDisplay} MiB (VmHWM). Frozen line: at most ${frozen.h3MaxRatio.toFixed(2)}.`}
          />
        </div>
      </section>

      <section id="throughput" className="card">
        <div className="head">
          <div>
            <p className="eyebrow">Throughput</p>
            <h2>Generation speed, every repetition</h2>
          </div>
          <span className="pill warn">H1 refuted on this measure</span>
        </div>
        <p className="sub">
          Prompt: first {fp32.throughput.promptTokens} ids of the perplexity set.{' '}
          {fp32.throughput.generatedTokens} greedy tokens per repetition, KV cache on, one
          warm-up, {fp32.throughput.reps.length} timed repetitions; time includes prefill.
          Mean ratio int8 / fp32: {derived.throughputRatioDisplay}.
        </p>
        <RepChart />
        <table>
          <thead>
            <tr><th>Arm</th>{fp32.throughput.display.reps.map((_, i) => <th key={i}>Rep {i + 1}</th>)}</tr>
          </thead>
          <tbody>
            <tr><td>fp32 (tok/s)</td>{fp32.throughput.display.reps.map((v, i) => <td key={i}>{v}</td>)}</tr>
            <tr><td>int8 (tok/s)</td>{int8.throughput.display.reps.map((v, i) => <td key={i}>{v}</td>)}</tr>
          </tbody>
        </table>
        <table>
          <thead><tr><th>Statistic</th><th>fp32</th><th>int8</th></tr></thead>
          <tbody>
            {t.map(([k, a, b]) => <tr key={k}><td>{k}</td><td>{a}</td><td>{b}</td></tr>)}
          </tbody>
        </table>
        <h3 className="block">Interpretation (not a measured claim)</h3>
        <p className="sub wide">
          In this engine the int8 path dequantizes every weight on every use, inside the inner
          loop. Under the frozen build (baseline x86-64 instruction set, no AVX or VNNI), the
          per-element int8-to-fp32 conversion costs more than the memory traffic it saves at
          this model size and batch size 1, so the smaller footprint does not turn into speed
          here. A true integer dot-product kernel or a wider instruction set could change this;
          neither was in the frozen design.
        </p>
        <h3 className="block">Run record</h3>
        <p className="sub wide">
          The fp32 arm completed in one run. The int8 arm's first attempt was interrupted by an
          environment restart after its throughput repetitions; that attempt is preserved in the
          repo as <code>results/run_int8_attempt1_interrupted.log</code> and produced no results
          file, so the arm was re-run in full under the same frozen protocol. The H1 verdict is
          the same under either run's mean. Every headline number on this page is from the
          completed runs only.
        </p>
      </section>

      <section id="quality" className="card">
        <div className="head">
          <div>
            <p className="eyebrow">Quality and memory</p>
            <h2>Perplexity and peak memory, both arms</h2>
          </div>
        </div>
        <div className="grid2">
          <div>
            <h3>Perplexity (wikitext-2 test slice)</h3>
            <table>
              <thead><tr><th>Arm</th><th>Total NLL</th><th>Perplexity</th></tr></thead>
              <tbody>
                <tr><td>fp32</td><td>{fp32.perplexity.totalNllDisplay}</td><td>{fp32.perplexity.valueDisplay}</td></tr>
                <tr><td>int8</td><td>{int8.perplexity.totalNllDisplay}</td><td>{int8.perplexity.valueDisplay}</td></tr>
              </tbody>
            </table>
            <p className="hint">
              First {fp32.perplexity.tokensInSet.toLocaleString('en-US')} GPT-2 BPE tokens of the
              frozen set, blocks of 1,024, {fp32.perplexity.predictedTokens.toLocaleString('en-US')}{' '}
              predicted tokens. Relative change: {derived.perplexityRelChangePctDisplay} percent.
            </p>
          </div>
          <div>
            <h3>Peak memory and weight storage</h3>
            <table>
              <thead><tr><th>Arm</th><th>VmHWM (MiB)</th><th>Weight storage (bytes)</th></tr></thead>
              <tbody>
                <tr><td>fp32</td><td>{fp32.vmhwmDisplay}</td><td>{fp32.weightStorageDisplay}</td></tr>
                <tr><td>int8</td><td>{int8.vmhwmDisplay}</td><td>{int8.weightStorageDisplay}</td></tr>
              </tbody>
            </table>
            <p className="hint">
              VmHWM ratio: {derived.vmhwmRatioDisplay}. Weight storage ratio:{' '}
              {derived.weightStorageRatioDisplay}. Weight storage is the loader's exact byte
              accounting.
            </p>
          </div>
        </div>
      </section>

      <section id="suite" className="card">
        <div className="head">
          <div>
            <p className="eyebrow">Degeneration suite</p>
            <h2>Mechanical scores, descriptive only</h2>
          </div>
          <span className="pill info">No hypothesis rests on this suite</span>
        </div>
        <p className="sub">
          {fp32.suite.promptCount} frozen prompts, 64 greedy tokens each. rep20: fraction of the
          last 20 generated tokens that repeat a token from the first 44 generated positions.
          distinct20: unique ids among the last 20, divided by 20. sane: all 64 tokens produced
          with finite logits.
        </p>
        <table>
          <thead><tr><th>Arm</th><th>rep20 (mean)</th><th>distinct20 (mean)</th><th>Sane</th></tr></thead>
          <tbody>
            <tr><td>fp32</td><td>{fp32.suite.rep20MeanDisplay}</td><td>{fp32.suite.distinct20MeanDisplay}</td><td>{fp32.suite.saneCount} of {fp32.suite.promptCount}</td></tr>
            <tr><td>int8</td><td>{int8.suite.rep20MeanDisplay}</td><td>{int8.suite.distinct20MeanDisplay}</td><td>{int8.suite.saneCount} of {int8.suite.promptCount}</td></tr>
          </tbody>
        </table>
        <p className="sub wide">
          Both arms loop heavily under greedy decoding (rep20 near 1.0): that is GPT-2 small's
          known greedy behaviour, and the int8 arm tracks the fp32 arm closely on these
          mechanical scores. No broader quality claim is made.
        </p>
        <h3 className="block">Study setup</h3>
        <p className="sub wide">
          Model: GPT-2 small (124M), <code>openai-community/gpt2</code>, licence tag MIT
          (Hugging Face model card). Engine: this repo's from-scratch C++20 engine (own
          safetensors parser, tensor ops, attention, KV cache, quantization path; standard
          library only; no inference library linked, vendored, or wrapped). Machine: AMD EPYC
          9D25, 4 vCPUs, about 7.7 GiB RAM, Ubuntu 24.04, g++ 13.3.0, 4 engine threads, greedy
          decoding everywhere. Correctness: fp32 logits match an independent float64 NumPy
          reference within 4.2e-07 on the tiny end-to-end fixture, and the engine reproduces
          the reference perplexity on real weights to 5 significant figures.
        </p>
      </section>

      <footer>
        <p>
          Study scope only: one small model, one machine, one quantization scheme, greedy
          decoding. Nothing here measures production serving, batching, GPUs, or other models,
          and no production experience is claimed. Full record:{' '}
          <a href={RESULTS_URL}>RESULTS.md</a> · <a href={REPO}>GitHub repo</a>. Every number on
          this page is generated at build time from the repo's committed raw harness outputs.
        </p>
      </footer>
    </div>
  );
}
