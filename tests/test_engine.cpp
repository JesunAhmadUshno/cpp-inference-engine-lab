// Unit and end-to-end tests for the from-scratch engine.
//
// Coverage:
//   - JSON parser on a safetensors-style header.
//   - Safetensors parser on a synthetic file written by the test.
//   - matvec / matmat / matvec_q against hand-computed values.
//   - layernorm, GELU (known values), softmax (known values).
//   - Quantization round trip: error at most scale/2 per channel.
//   - A single transformer block built in code whose expected
//     output is computed by hand (uniform attention case).
//   - On the committed tiny fixture (tests/fixtures):
//       * causality: position 0 logits do not depend on later ids.
//       * KV-cache incremental decoding matches full forward.
//       * End to end: fp32 logits within 2e-4 of the independent
//         NumPy reference, int8 logits within 0.1 (frozen gates).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "engine/gpt2.hpp"
#include "engine/json.hpp"
#include "engine/safetensors.hpp"

using namespace engine;

namespace {

int failures = 0;

void check(bool ok, const std::string& name) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str());
  if (!ok) failures++;
}

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

void test_json() {
  Json v = parse_json(
      "{\"a\": {\"dtype\": \"F32\", \"shape\": [2, 3], "
      "\"data_offsets\": [0, 24]}, \"b\": \"x\\ny\", \"c\": -1.5e2}");
  const Json* a = v.find("a");
  bool ok = a && a->find("dtype")->str == "F32" &&
            a->find("shape")->arr.size() == 2 &&
            a->find("shape")->arr[1].num == 3 &&
            a->find("data_offsets")->arr[1].num == 24 &&
            v.find("b")->str == "x\ny" && near(v.find("c")->num, -150.0, 1e-9);
  check(ok, "json parser: nested header values");
}

void test_safetensors_synthetic() {
  const std::string path = "/tmp/engine_test_synthetic.safetensors";
  // Two F32 tensors: m [2][3] values 0..5, v [4] values 10..13.
  std::string header =
      "{\"m\": {\"dtype\": \"F32\", \"shape\": [2, 3], "
      "\"data_offsets\": [0, 24]}, "
      "\"v\": {\"dtype\": \"F32\", \"shape\": [4], "
      "\"data_offsets\": [24, 40]}}";
  {
    std::ofstream out(path, std::ios::binary);
    uint64_t hlen = header.size();
    out.write(reinterpret_cast<const char*>(&hlen), 8);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    float vals[10] = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13};
    out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
  }
  SafetensorsFile st(path);
  std::vector<float> m = st.read_f32("m");
  std::vector<float> v = st.read_f32("v");
  bool ok = st.info("m").shape == std::vector<int64_t>({2, 3}) &&
            m.size() == 6 && v.size() == 4;
  for (int i = 0; i < 6 && ok; i++) ok = m[static_cast<size_t>(i)] == i;
  for (int i = 0; i < 4 && ok; i++) ok = v[static_cast<size_t>(i)] == 10 + i;
  check(ok, "safetensors: synthetic file round trip");
}

void test_matvec() {
  Matrix w(2, 3);
  w.a = {1, 2, 3, 4, 5, 6};
  float x[3] = {1, 0, -1};
  float y[2] = {0, 0};
  matvec(w, x, y);
  check(near(y[0], -2, 1e-6) && near(y[1], -2, 1e-6),
        "matvec: hand-computed values");

  float X[6] = {1, 0, -1, 2, 1, 0};  // two rows
  float Y[4] = {0, 0, 0, 0};
  matmat(w, X, 2, 3, Y, 2);
  check(near(Y[0], -2, 1e-6) && near(Y[1], -2, 1e-6) && near(Y[2], 4, 1e-6) &&
            near(Y[3], 13, 1e-6),
        "matmat: hand-computed values");
}

void test_matvec_q() {
  QMatrix q;
  q.rows = 2;
  q.cols = 3;
  q.q = {1, -2, 3, 127, 0, -127};
  q.scale = {0.5f, 0.01f};
  float x[3] = {2, 1, -1};
  float y[2] = {0, 0};
  matvec_q(q, x, y);
  // y0 = 0.5 * (2 - 2 - 3) = -1.5 ; y1 = 0.01 * (254 + 0 + 127) = 3.81
  check(near(y[0], -1.5, 1e-6) && near(y[1], 3.81, 1e-5),
        "matvec_q: hand-computed dequantized values");
}

void test_layernorm() {
  float x[4] = {1, 2, 3, 4};
  float gain[4] = {1, 1, 1, 1};
  float bias[4] = {0, 0, 0, 0};
  layernorm(x, 1, 4, 4, gain, bias, 1e-5f);
  // mean 2.5, var 1.25, inv std = 1/sqrt(1.25001) = 0.894424
  bool ok = near(x[0], -1.341636, 1e-4) && near(x[1], -0.447212, 1e-4) &&
            near(x[2], 0.447212, 1e-4) && near(x[3], 1.341636, 1e-4);
  check(ok, "layernorm: hand-computed values");
}

void test_gelu_softmax() {
  bool ok = near(gelu_tanh(0.0f), 0.0, 1e-7) &&
            near(gelu_tanh(1.0f), 0.841192, 1e-5) &&
            near(gelu_tanh(-1.0f), -0.158808, 1e-5) &&
            near(gelu_tanh(2.0f), 1.954558, 1e-4);
  float s[3] = {1, 1, 1};
  softmax(s, 3);
  ok = ok && near(s[0], 1.0 / 3, 1e-6) && near(s[2], 1.0 / 3, 1e-6);
  float s2[2] = {0, 2};
  softmax(s2, 2);
  ok = ok && near(s2[0], 0.119203, 1e-5) && near(s2[1], 0.880797, 1e-5);
  check(ok, "gelu and softmax: known values");
}

void test_quantize_roundtrip() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Matrix w(64, 48);
  for (float& v : w.a) v = dist(rng);
  QMatrix q = quantize_rows(w);
  bool ok = true;
  for (int j = 0; j < w.rows && ok; j++) {
    for (int i = 0; i < w.cols; i++) {
      float deq = q.q[static_cast<size_t>(j) * w.cols + i] *
                  q.scale[static_cast<size_t>(j)];
      float err = std::fabs(deq - w.a[static_cast<size_t>(j) * w.cols + i]);
      if (err > q.scale[static_cast<size_t>(j)] / 2 + 1e-6f) ok = false;
    }
  }
  check(ok, "quantize: round-trip error at most scale/2 per channel");
}

// One transformer block with weights chosen so the expected output
// is computable by hand. Token and position embeddings are constant
// rows, so LayerNorm-1 output is exactly its bias [0.7, -0.2] at
// every position. c_attn has q = k = 0 (attention scores all zero,
// hence uniform over the causal past) and v = identity, so the
// attention output is [0.7, -0.2] at every position. c_proj is the
// identity. The MLP path then maps [1, -1] (the LayerNorm-2 output,
// deviations +-0.45 over std 0.45) through fc rows [1,0],[0,1],
// [1,1],[0,0] and GELU, and proj picks the first two coordinates.
// Final LayerNorm of [1.541169, -0.358800] has deviations
// +-0.9499845 over std 0.94999, giving [1, -1] up to 1e-3.
void test_block_hand_computed() {
  Model m;
  m.cfg.vocab = 5;
  m.cfg.embd = 2;
  m.cfg.heads = 1;
  m.cfg.layers = 1;
  m.cfg.ctx = 16;
  m.wte = Matrix(5, 2);
  for (int r = 0; r < 5; r++) {
    m.wte.row(r)[0] = 0.3f;
    m.wte.row(r)[1] = 0.3f;
  }
  m.wpe = Matrix(16, 2);
  for (int r = 0; r < 16; r++) {
    m.wpe.row(r)[0] = -0.3f;
    m.wpe.row(r)[1] = -0.3f;
  }
  Block b;
  b.ln1_g = {1, 1};
  b.ln1_b = {0.7f, -0.2f};
  b.c_attn.in = 2;
  b.c_attn.out = 6;
  b.c_attn.w = Matrix(6, 2);
  b.c_attn.w.a = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1};
  b.c_attn.bias.assign(6, 0.0f);
  b.c_proj.in = 2;
  b.c_proj.out = 2;
  b.c_proj.w = Matrix(2, 2);
  b.c_proj.w.a = {1, 0, 0, 1};
  b.c_proj.bias.assign(2, 0.0f);
  b.ln2_g = {1, 1};
  b.ln2_b = {0, 0};
  b.mlp_fc.in = 2;
  b.mlp_fc.out = 4;
  b.mlp_fc.w = Matrix(4, 2);
  b.mlp_fc.w.a = {1, 0, 0, 1, 1, 1, 0, 0};
  b.mlp_fc.bias.assign(4, 0.0f);
  b.mlp_proj.in = 4;
  b.mlp_proj.out = 2;
  b.mlp_proj.w = Matrix(2, 4);
  b.mlp_proj.w.a = {1, 0, 0, 0, 0, 1, 0, 0};
  b.mlp_proj.bias.assign(2, 0.0f);
  m.blocks.push_back(std::move(b));
  m.lnf_g = {1, 1};
  m.lnf_b = {0, 0};

  std::vector<float> hidden = forward_all(m, {0, 1, 2});
  bool ok = hidden.size() == 6;
  for (int t = 0; t < 3 && ok; t++) {
    ok = near(hidden[static_cast<size_t>(t) * 2], 1.0, 3e-3) &&
         near(hidden[static_cast<size_t>(t) * 2 + 1], -1.0, 3e-3);
  }
  check(ok, "block: hand-computed single block output [1, -1]");
}

std::vector<int> read_ids(const std::string& path) {
  std::ifstream in(path);
  std::vector<int> ids;
  int v;
  while (in >> v) ids.push_back(v);
  return ids;
}

std::vector<float> last_logits(const Model& m, const std::vector<int>& ids) {
  std::vector<float> hidden = forward_all(m, ids);
  std::vector<float> logits(static_cast<size_t>(m.cfg.vocab));
  logits_row(m, hidden.data() + (ids.size() - 1) * m.cfg.embd, logits.data());
  return logits;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    d = std::max(d, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return d;
}

void test_fixtures(const std::string& dir) {
  std::string model_path = dir + "/model.safetensors";
  std::vector<int> ids = read_ids(dir + "/ids.txt");

  std::ifstream in(dir + "/expected_logits.json");
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  Json expected = parse_json(text);
  const Json* arr = expected.find("logits_last");
  std::vector<double> ref;
  for (const Json& v : arr->arr) ref.push_back(v.num);

  Model fp32 = load_model(model_path, false, 4);
  std::vector<float> logits = last_logits(fp32, ids);
  double d_fp32 = 0.0;
  for (size_t i = 0; i < ref.size(); i++) {
    d_fp32 = std::max(d_fp32, std::fabs(logits[i] - ref[i]));
  }
  std::printf("  [info] fp32 vs NumPy reference max abs diff: %.2e\n", d_fp32);
  check(d_fp32 <= 2e-4, "e2e fp32: logits match NumPy reference within 2e-4");
  check(argmax(logits.data(), fp32.cfg.vocab) ==
            static_cast<int>(expected.find("argmax_last")->num),
        "e2e fp32: argmax matches reference");

  Model int8 = load_model(model_path, true, 4);
  std::vector<float> logits_q = last_logits(int8, ids);
  double d_q = 0.0;
  for (size_t i = 0; i < ref.size(); i++) {
    d_q = std::max(d_q, std::fabs(logits_q[i] - ref[i]));
  }
  std::printf("  [info] int8 vs NumPy reference max abs diff: %.2e\n", d_q);
  check(d_q <= 0.1, "e2e int8: logits within 0.1 of reference");

  // Causality: position 0 must not depend on later tokens.
  std::vector<int> ids_b = ids;
  ids_b.back() = (ids_b.back() + 1) % fp32.cfg.vocab;
  std::vector<float> h_a = forward_all(fp32, ids);
  std::vector<float> h_b = forward_all(fp32, ids_b);
  double d0 = 0.0;
  for (int d = 0; d < fp32.cfg.embd; d++) {
    d0 = std::max(d0, std::fabs(static_cast<double>(h_a[static_cast<size_t>(d)]) -
                                h_b[static_cast<size_t>(d)]));
  }
  check(d0 <= 1e-6, "causality: position 0 independent of later tokens");

  // KV-cache equivalence: stepping through the ids must reproduce
  // the full-forward last-position logits.
  KVCache cache;
  cache.init(fp32);
  std::vector<float> step_logits(static_cast<size_t>(fp32.cfg.vocab));
  for (size_t i = 0; i < ids.size(); i++) {
    forward_step(fp32, cache, ids[i], static_cast<int>(i), step_logits.data());
  }
  double d_kv = max_abs_diff(step_logits, logits);
  std::printf("  [info] KV cache vs full forward max abs diff: %.2e\n", d_kv);
  check(d_kv <= 1e-5, "kv cache: incremental logits match full forward");
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixtures = argc > 1 ? argv[1] : "tests/fixtures";
  test_json();
  test_safetensors_synthetic();
  test_matvec();
  test_matvec_q();
  test_layernorm();
  test_gelu_softmax();
  test_quantize_roundtrip();
  test_block_hand_computed();
  test_fixtures(fixtures);
  std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
              failures, failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
