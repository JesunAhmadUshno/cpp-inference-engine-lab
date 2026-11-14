// GPT-2 forward pass, written for this repo.
//
// Pre-LayerNorm transformer blocks, tanh-approximation GELU,
// causal multi-head attention, tied input/output embeddings.
// Two execution modes share the same weights and math:
//   - forward_all: full-sequence forward (perplexity).
//   - forward_step: incremental decoding with a KV cache
//     (generation), which the tests prove equivalent to
//     forward_all on a tiny model.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/safetensors.hpp"
#include "engine/tensor.hpp"

namespace engine {

struct Config {
  int vocab = 50257;
  int ctx = 1024;
  int embd = 768;
  int heads = 12;
  int layers = 12;
  float ln_eps = 1e-5f;

  int head_dim() const { return embd / heads; }
};

// One linear layer y = W x + b, weights either fp32 or in the
// W8A32 int8 form (see PREREGISTRATION.md).
struct Linear {
  bool is_q = false;
  Matrix w;    // fp32 path, compute orientation [out][in]
  QMatrix qw;  // int8 path, compute orientation [out][in]
  std::vector<float> bias;
  int in = 0;
  int out = 0;

  void apply(const float* x, float* y) const {
    if (is_q) {
      matvec_q(qw, x, y);
    } else {
      matvec(w, x, y);
    }
    for (int j = 0; j < out; j++) y[j] += bias[static_cast<size_t>(j)];
  }

  void apply_batch(const float* x, int t, int ldx, float* y, int ldy) const {
    if (is_q) {
      matmat_q(qw, x, t, ldx, y, ldy);
    } else {
      matmat(w, x, t, ldx, y, ldy);
    }
    add_bias(y, bias.data(), t, out, ldy);
  }

  size_t bytes() const {
    return (is_q ? qw.bytes() : w.bytes()) + bias.size() * sizeof(float);
  }
};

struct Block {
  std::vector<float> ln1_g, ln1_b;
  Linear c_attn;   // embd -> 3*embd
  Linear c_proj;   // embd -> embd
  std::vector<float> ln2_g, ln2_b;
  Linear mlp_fc;   // embd -> 4*embd
  Linear mlp_proj; // 4*embd -> embd
};

struct Model {
  Config cfg;
  Matrix wte;  // [vocab][embd], also the tied lm head
  Matrix wpe;  // [ctx][embd]
  std::vector<Block> blocks;
  std::vector<float> lnf_g, lnf_b;
  size_t weight_bytes = 0;
};

// Load one stored Conv1D weight (stored orientation [in][out]) as a
// Linear in compute orientation [out][in], fp32 or quantized.
inline Linear load_linear(const SafetensorsFile& st, const std::string& name,
                          bool quantize) {
  const TensorInfo& info = st.info(name + ".weight");
  if (info.shape.size() != 2) {
    throw std::runtime_error("expected 2D weight for " + name);
  }
  const int in = static_cast<int>(info.shape[0]);
  const int out = static_cast<int>(info.shape[1]);
  Linear lin;
  lin.in = in;
  lin.out = out;
  lin.bias = st.read_f32(name + ".bias");
  if (!quantize) {
    std::vector<float> stored = st.read_f32(name + ".weight");
    lin.w = Matrix(out, in);
    pool().parallel_for(out, [&](int64_t j0, int64_t j1) {
      for (int64_t j = j0; j < j1; j++) {
        float* dst = lin.w.row(static_cast<int>(j));
        for (int i = 0; i < in; i++) {
          dst[i] = stored[static_cast<size_t>(i) * out + j];
        }
      }
    });
  } else {
    lin.is_q = true;
    lin.qw.rows = out;
    lin.qw.cols = in;
    lin.qw.q.resize(static_cast<size_t>(out) * in);
    lin.qw.scale.assign(out, 0.0f);
    // Pass 1: per-output-channel max |w|. Stored element e sits at
    // row i = e / out (input feature), column j = e % out (output
    // channel), and lands at q[j * in + i].
    {
      std::vector<float> mx(out, 0.0f);
      int64_t i = 0;
      int j = 0;
      st.for_each_f32_chunk(name + ".weight",
                            [&](const float* data, size_t n, size_t) {
                              for (size_t e = 0; e < n; e++) {
                                float v = std::fabs(data[e]);
                                if (v > mx[static_cast<size_t>(j)]) {
                                  mx[static_cast<size_t>(j)] = v;
                                }
                                if (++j == out) {
                                  j = 0;
                                  i++;
                                }
                              }
                            });
      for (int c = 0; c < out; c++) {
        float s = mx[static_cast<size_t>(c)] / 127.0f;
        lin.qw.scale[static_cast<size_t>(c)] = (s == 0.0f) ? 1.0f : s;
      }
    }
    // Pass 2: quantize into compute orientation.
    {
      int64_t i = 0;
      int j = 0;
      st.for_each_f32_chunk(name + ".weight",
                            [&](const float* data, size_t n, size_t) {
                              for (size_t e = 0; e < n; e++) {
                                float v = std::round(
                                    data[e] / lin.qw.scale[static_cast<size_t>(j)]);
                                v = std::max(-127.0f, std::min(127.0f, v));
                                lin.qw.q[static_cast<size_t>(j) * in +
                                         static_cast<size_t>(i)] =
                                    static_cast<int8_t>(v);
                                if (++j == out) {
                                  j = 0;
                                  i++;
                                }
                              }
                            });
    }
  }
  return lin;
}

inline Model load_model(const std::string& path, bool quantize, int heads) {
  SafetensorsFile st(path);
  Model m;
  const TensorInfo& wte_info = st.info("wte.weight");
  const TensorInfo& wpe_info = st.info("wpe.weight");
  m.cfg.vocab = static_cast<int>(wte_info.shape[0]);
  m.cfg.embd = static_cast<int>(wte_info.shape[1]);
  m.cfg.ctx = static_cast<int>(wpe_info.shape[0]);
  m.cfg.heads = heads;
  int layers = 0;
  while (st.has("h." + std::to_string(layers) + ".ln_1.weight")) layers++;
  m.cfg.layers = layers;
  if (layers == 0 || m.cfg.embd % heads != 0) {
    throw std::runtime_error("unexpected checkpoint geometry");
  }

  m.wte = Matrix(m.cfg.vocab, m.cfg.embd);
  {
    std::vector<float> v = st.read_f32("wte.weight");
    m.wte.a = std::move(v);
  }
  m.wpe = Matrix(m.cfg.ctx, m.cfg.embd);
  {
    std::vector<float> v = st.read_f32("wpe.weight");
    m.wpe.a = std::move(v);
  }
  m.weight_bytes = m.wte.bytes() + m.wpe.bytes();

  for (int l = 0; l < layers; l++) {
    std::string p = "h." + std::to_string(l) + ".";
    Block b;
    b.ln1_g = st.read_f32(p + "ln_1.weight");
    b.ln1_b = st.read_f32(p + "ln_1.bias");
    b.c_attn = load_linear(st, p + "attn.c_attn", quantize);
    b.c_proj = load_linear(st, p + "attn.c_proj", quantize);
    b.ln2_g = st.read_f32(p + "ln_2.weight");
    b.ln2_b = st.read_f32(p + "ln_2.bias");
    b.mlp_fc = load_linear(st, p + "mlp.c_fc", quantize);
    b.mlp_proj = load_linear(st, p + "mlp.c_proj", quantize);
    m.weight_bytes += b.ln1_g.size() * 4 + b.ln1_b.size() * 4 +
                      b.ln2_g.size() * 4 + b.ln2_b.size() * 4 +
                      b.c_attn.bytes() + b.c_proj.bytes() + b.mlp_fc.bytes() +
                      b.mlp_proj.bytes();
    m.blocks.push_back(std::move(b));
  }
  m.lnf_g = st.read_f32("ln_f.weight");
  m.lnf_b = st.read_f32("ln_f.bias");
  m.weight_bytes += m.lnf_g.size() * 4 + m.lnf_b.size() * 4;
  return m;
}

// ----------------------------------------------------------------
// Full-sequence forward. Returns the post-final-LayerNorm hidden
// states [T][embd]; logits for position t are wte . hidden[t].
inline std::vector<float> forward_all(const Model& m,
                                      const std::vector<int>& ids) {
  const int T = static_cast<int>(ids.size());
  const int D = m.cfg.embd;
  const int H = m.cfg.heads;
  const int dh = m.cfg.head_dim();
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(dh));

  std::vector<float> x(static_cast<size_t>(T) * D);
  for (int t = 0; t < T; t++) {
    const float* te = m.wte.row(ids[static_cast<size_t>(t)]);
    const float* pe = m.wpe.row(t);
    float* dst = x.data() + static_cast<size_t>(t) * D;
    for (int d = 0; d < D; d++) dst[d] = te[d] + pe[d];
  }

  std::vector<float> h(static_cast<size_t>(T) * D);
  std::vector<float> qkv(static_cast<size_t>(T) * 3 * D);
  std::vector<float> attn(static_cast<size_t>(T) * D);
  std::vector<float> f(static_cast<size_t>(T) * 4 * D);

  for (const Block& b : m.blocks) {
    h = x;
    layernorm(h.data(), T, D, D, b.ln1_g.data(), b.ln1_b.data(), m.cfg.ln_eps);
    b.c_attn.apply_batch(h.data(), T, D, qkv.data(), 3 * D);
    // Causal attention, one output row at a time.
    pool().parallel_for(T, [&](int64_t t0, int64_t t1) {
      std::vector<float> scores(static_cast<size_t>(T));
      for (int64_t t = t0; t < t1; t++) {
        const float* qrow = qkv.data() + t * 3 * D;
        float* orow = attn.data() + t * D;
        for (int hd = 0; hd < H; hd++) {
          const float* q = qrow + hd * dh;
          for (int64_t s = 0; s <= t; s++) {
            const float* k = qkv.data() + s * 3 * D + D + hd * dh;
            float dot = 0.0f;
            for (int d = 0; d < dh; d++) dot += q[d] * k[d];
            scores[static_cast<size_t>(s)] = dot * attn_scale;
          }
          softmax(scores.data(), static_cast<int>(t + 1));
          float* o = orow + hd * dh;
          for (int d = 0; d < dh; d++) o[d] = 0.0f;
          for (int64_t s = 0; s <= t; s++) {
            const float* v = qkv.data() + s * 3 * D + 2 * D + hd * dh;
            float p = scores[static_cast<size_t>(s)];
            for (int d = 0; d < dh; d++) o[d] += p * v[d];
          }
        }
      }
    });
    b.c_proj.apply_batch(attn.data(), T, D, h.data(), D);
    add_inplace(x.data(), h.data(), T * D);
    h = x;
    layernorm(h.data(), T, D, D, b.ln2_g.data(), b.ln2_b.data(), m.cfg.ln_eps);
    b.mlp_fc.apply_batch(h.data(), T, D, f.data(), 4 * D);
    gelu_inplace(f.data(), static_cast<int64_t>(T) * 4 * D);
    b.mlp_proj.apply_batch(f.data(), T, 4 * D, h.data(), D);
    add_inplace(x.data(), h.data(), T * D);
  }

  layernorm(x.data(), T, D, D, m.lnf_g.data(), m.lnf_b.data(), m.cfg.ln_eps);
  return x;
}

// Logits for one hidden row: tied embedding, no bias.
inline void logits_row(const Model& m, const float* hidden, float* logits) {
  matvec(m.wte, hidden, logits);
}

// ----------------------------------------------------------------
// Incremental decoding with a KV cache.
struct KVCache {
  int layers = 0;
  int ctx = 0;
  int embd = 0;
  std::vector<std::vector<float>> k;  // per layer [ctx][embd]
  std::vector<std::vector<float>> v;
  int len = 0;

  void init(const Model& m) {
    layers = m.cfg.layers;
    ctx = m.cfg.ctx;
    embd = m.cfg.embd;
    k.assign(layers, std::vector<float>(static_cast<size_t>(ctx) * embd, 0.0f));
    v.assign(layers, std::vector<float>(static_cast<size_t>(ctx) * embd, 0.0f));
    len = 0;
  }
};

// One decoding step: feed `token` at position `pos`, write the
// logits for the next token into `logits` (vocab floats).
inline void forward_step(const Model& m, KVCache& cache, int token, int pos,
                         float* logits) {
  const int D = m.cfg.embd;
  const int H = m.cfg.heads;
  const int dh = m.cfg.head_dim();
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(dh));

  std::vector<float> x(D), h(D), qkv(3 * D), attn(D), f(4 * D), proj(D);
  const float* te = m.wte.row(token);
  const float* pe = m.wpe.row(pos);
  for (int d = 0; d < D; d++) x[static_cast<size_t>(d)] = te[d] + pe[d];

  for (int l = 0; l < m.cfg.layers; l++) {
    const Block& b = m.blocks[static_cast<size_t>(l)];
    h = x;
    layernorm(h.data(), 1, D, D, b.ln1_g.data(), b.ln1_b.data(), m.cfg.ln_eps);
    b.c_attn.apply(h.data(), qkv.data());
    float* kc = cache.k[static_cast<size_t>(l)].data() + static_cast<size_t>(pos) * D;
    float* vc = cache.v[static_cast<size_t>(l)].data() + static_cast<size_t>(pos) * D;
    std::memcpy(kc, qkv.data() + D, D * sizeof(float));
    std::memcpy(vc, qkv.data() + 2 * D, D * sizeof(float));
    std::vector<float> scores(static_cast<size_t>(pos) + 1);
    for (int hd = 0; hd < H; hd++) {
      const float* q = qkv.data() + hd * dh;
      for (int s = 0; s <= pos; s++) {
        const float* k = cache.k[static_cast<size_t>(l)].data() +
                         static_cast<size_t>(s) * D + hd * dh;
        float dot = 0.0f;
        for (int d = 0; d < dh; d++) dot += q[d] * k[d];
        scores[static_cast<size_t>(s)] = dot * attn_scale;
      }
      softmax(scores.data(), pos + 1);
      float* o = attn.data() + hd * dh;
      for (int d = 0; d < dh; d++) o[d] = 0.0f;
      for (int s = 0; s <= pos; s++) {
        const float* v = cache.v[static_cast<size_t>(l)].data() +
                         static_cast<size_t>(s) * D + hd * dh;
        float p = scores[static_cast<size_t>(s)];
        for (int d = 0; d < dh; d++) o[d] += p * v[d];
      }
    }
    b.c_proj.apply(attn.data(), proj.data());
    add_inplace(x.data(), proj.data(), D);
    h = x;
    layernorm(h.data(), 1, D, D, b.ln2_g.data(), b.ln2_b.data(), m.cfg.ln_eps);
    b.mlp_fc.apply(h.data(), f.data());
    gelu_inplace(f.data(), 4 * D);
    b.mlp_proj.apply(f.data(), proj.data());
    add_inplace(x.data(), proj.data(), D);
  }
  cache.len = pos + 1;

  h = x;
  layernorm(h.data(), 1, D, D, m.lnf_g.data(), m.lnf_b.data(), m.cfg.ln_eps);
  logits_row(m, h.data(), logits);
}

// Greedy generation with the KV cache. Returns the generated token
// ids (exactly n_new of them) and sets all_finite to false if any
// logit vector contained a NaN or an infinity.
inline std::vector<int> generate_greedy(const Model& m,
                                        const std::vector<int>& prompt,
                                        int n_new, bool& all_finite) {
  KVCache cache;
  cache.init(m);
  std::vector<float> logits(static_cast<size_t>(m.cfg.vocab));
  all_finite = true;
  auto check_finite = [&] {
    for (float v : logits) {
      if (!std::isfinite(v)) all_finite = false;
    }
  };
  int pos = 0;
  for (size_t i = 0; i < prompt.size(); i++) {
    forward_step(m, cache, prompt[i], pos++, logits.data());
  }
  check_finite();
  std::vector<int> out;
  out.reserve(static_cast<size_t>(n_new));
  for (int g = 0; g < n_new; g++) {
    int next = argmax(logits.data(), m.cfg.vocab);
    out.push_back(next);
    if (g + 1 < n_new) {
      forward_step(m, cache, next, pos++, logits.data());
      check_finite();
    }
  }
  return out;
}

}  // namespace engine
