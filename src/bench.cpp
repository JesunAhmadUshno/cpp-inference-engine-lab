// Benchmark harness for cpp-inference-engine-lab.
//
// Implements the frozen protocol of PREREGISTRATION.md:
//   - throughput: first 64 ids of data/ppl_ids.txt as prompt, 128
//     greedy tokens, one warm-up, 7 timed repetitions.
//   - peak memory: VmHWM of this process read right after the
//     timed repetitions, plus the loader's exact weight bytes.
//   - perplexity: data/ppl_ids.txt (8,192 ids), non-overlapping
//     blocks of 1,024, 8,184 predicted tokens.
//   - degeneration: the 12-prompt suite, 64 greedy tokens each,
//     rep20 / distinct20 / sane as frozen.
// Each arm runs as its own process invocation. Raw results are
// written as JSON to --out. A --smoke mode generates a few tokens
// from caller-supplied ids (used by CI on a tiny random model).
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "engine/gpt2.hpp"

using namespace engine;

namespace {

std::string g_arg(int argc, char** argv, const std::string& name,
                    const std::string& dflt) {
  for (int i = 1; i + 1 < argc; i++) {
    if (name == argv[i]) return argv[i + 1];
  }
  return dflt;
}

bool has_flag(int argc, char** argv, const std::string& name) {
  for (int i = 1; i < argc; i++) {
    if (name == argv[i]) return true;
  }
  return false;
}

std::vector<int> read_ids_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open ids file: " + path);
  std::vector<int> ids;
  int v;
  while (in >> v) ids.push_back(v);
  return ids;
}

std::vector<std::vector<int>> read_prompt_ids(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open prompt ids: " + path);
  std::vector<std::vector<int>> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::vector<int> ids;
    int v;
    while (ss >> v) ids.push_back(v);
    out.push_back(ids);
  }
  return out;
}

std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open file: " + path);
  std::vector<std::string> out;
  std::string line;
  while (std::getline(in, line)) out.push_back(line);
  return out;
}

std::vector<std::string> read_vocab(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open vocab: " + path);
  std::vector<std::string> out;
  std::string line;
  while (std::getline(in, line)) {
    auto tab = line.find('\t');
    if (tab == std::string::npos) continue;
    int id = std::stoi(line.substr(0, tab));
    std::string hex = line.substr(tab + 1);
    std::string bytes;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
      bytes.push_back(static_cast<char>(
          std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    if (static_cast<int>(out.size()) <= id) out.resize(id + 1);
    out[static_cast<size_t>(id)] = bytes;
  }
  return out;
}

std::string json_escape_bytes(const std::string& s) {
  std::string out;
  char buf[8];
  for (unsigned char c : s) {
    if (c == '"') {
      out += "\\\"";
    } else if (c == '\\') {
      out += "\\\\";
    } else if (c == '\n') {
      out += "\\n";
    } else if (c >= 0x20 && c < 0x7F) {
      out.push_back(static_cast<char>(c));
    } else {
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    }
  }
  return out;
}

double peak_rss_mib() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      long kb = std::strtol(line.c_str() + 6, nullptr, 10);
      return kb / 1024.0;
    }
  }
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss / 1024.0;
}

struct Stats {
  double mean, stddev, median, mn, mx;
};

Stats stats_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  Stats s{};
  double sum = 0.0;
  for (double x : v) sum += x;
  s.mean = sum / v.size();
  double sq = 0.0;
  for (double x : v) sq += (x - s.mean) * (x - s.mean);
  s.stddev = v.size() > 1 ? std::sqrt(sq / (v.size() - 1)) : 0.0;
  s.median = v[v.size() / 2];
  s.mn = v.front();
  s.mx = v.back();
  return s;
}

std::string fmt(double v, int prec = 6) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path = g_arg(argc, argv, "--model", "");
    std::string arm = g_arg(argc, argv, "--arm", "fp32");
    std::string data_dir = g_arg(argc, argv, "--data", "data");
    std::string out_path = g_arg(argc, argv, "--out", "");
    int heads = std::stoi(g_arg(argc, argv, "--heads", "12"));
    bool quantize = (arm == "int8");
    if (model_path.empty() || (arm != "fp32" && arm != "int8")) {
      std::cerr << "usage: engine_bench --model PATH --arm fp32|int8 "
                   "[--heads N] [--data DIR] [--out results.json]\n"
                   "       engine_bench --model PATH --arm fp32|int8 "
                   "--smoke --prompt-ids \"1 2 3\" --gen N\n";
      return 2;
    }

    auto t0 = std::chrono::steady_clock::now();
    Model model = load_model(model_path, quantize, heads);
    auto t1 = std::chrono::steady_clock::now();
    double load_s = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "[bench] arm=" << arm << " loaded in " << fmt(load_s, 2)
              << " s, weight storage " << model.weight_bytes << " bytes\n";

    if (has_flag(argc, argv, "--smoke")) {
      std::string pids = g_arg(argc, argv, "--prompt-ids", "");
      int gen = std::stoi(g_arg(argc, argv, "--gen", "8"));
      std::vector<int> prompt;
      std::istringstream ss(pids);
      int v;
      while (ss >> v) prompt.push_back(v);
      bool finite = true;
      std::vector<int> out = generate_greedy(model, prompt, gen, finite);
      std::cout << "generated:";
      for (int id : out) std::cout << " " << id;
      std::cout << "\nfinite_logits: " << (finite ? "yes" : "no") << "\n";
      return finite ? 0 : 1;
    }

    // ---- Throughput (frozen protocol) ----
    std::vector<int> ppl_ids = read_ids_file(data_dir + "/ppl_ids.txt");
    std::vector<int> thr_prompt(ppl_ids.begin(), ppl_ids.begin() + 64);
    {
      bool finite = true;
      generate_greedy(model, thr_prompt, 128, finite);  // warm-up, untimed
    }
    std::vector<double> rep_tps;
    std::vector<double> rep_secs;
    for (int r = 0; r < 7; r++) {
      bool finite = true;
      auto a = std::chrono::steady_clock::now();
      generate_greedy(model, thr_prompt, 128, finite);
      auto b = std::chrono::steady_clock::now();
      double secs = std::chrono::duration<double>(b - a).count();
      rep_secs.push_back(secs);
      rep_tps.push_back(128.0 / secs);
      std::cerr << "[bench] throughput rep " << r << ": " << fmt(secs, 3)
                << " s, " << fmt(128.0 / secs, 3) << " tok/s\n";
    }
    Stats tps = stats_of(rep_tps);
    double vmhwm = peak_rss_mib();
    std::cerr << "[bench] VmHWM after throughput: " << fmt(vmhwm, 1)
              << " MiB\n";

    // ---- Perplexity (frozen protocol) ----
    double total_nll = 0.0;
    int64_t predicted = 0;
    {
      std::vector<float> logits(static_cast<size_t>(model.cfg.vocab));
      const int B = 1024;
      for (size_t start = 0; start + 1 < ppl_ids.size(); start += B) {
        int len = static_cast<int>(
            std::min<size_t>(B, ppl_ids.size() - start));
        std::vector<int> block(ppl_ids.begin() + start,
                               ppl_ids.begin() + start + len);
        std::vector<float> hidden = forward_all(model, block);
        for (int t = 0; t + 1 < len; t++) {
          logits_row(model,
                     hidden.data() + static_cast<size_t>(t) * model.cfg.embd,
                     logits.data());
          double mx = logits[0];
          for (int i = 1; i < model.cfg.vocab; i++) {
            mx = std::max(mx, static_cast<double>(logits[static_cast<size_t>(i)]));
          }
          double sum = 0.0;
          for (int i = 0; i < model.cfg.vocab; i++) {
            sum += std::exp(static_cast<double>(logits[static_cast<size_t>(i)]) - mx);
          }
          double log_z = mx + std::log(sum);
          total_nll += log_z - logits[static_cast<size_t>(block[static_cast<size_t>(t + 1)])];
          predicted++;
        }
        std::cerr << "[bench] perplexity block at " << start << " done, "
                  << "running ppl "
                  << fmt(std::exp(total_nll / predicted), 4) << "\n";
      }
    }
    double ppl = std::exp(total_nll / static_cast<double>(predicted));

    // ---- Degeneration suite (frozen protocol) ----
    std::vector<std::vector<int>> suite = read_prompt_ids(data_dir + "/prompt_ids.txt");
    std::vector<std::string> prompt_text = read_lines(data_dir + "/prompts.txt");
    std::vector<std::string> vocab = read_vocab(data_dir + "/vocab.tsv");
    struct SuiteRow {
      double rep20, distinct20;
      bool sane;
      std::vector<int> gen;
      std::string text;
    };
    std::vector<SuiteRow> rows;
    double sum_rep = 0.0, sum_dist = 0.0;
    int sane_count = 0;
    for (size_t p = 0; p < suite.size(); p++) {
      bool finite = true;
      std::vector<int> gen = generate_greedy(model, suite[p], 64, finite);
      SuiteRow row{};
      row.gen = gen;
      row.sane = finite && gen.size() == 64;
      int rep_hits = 0;
      for (int i = 44; i < 64; i++) {
        for (int j = 0; j < 44; j++) {
          if (gen[static_cast<size_t>(i)] == gen[static_cast<size_t>(j)]) {
            rep_hits++;
            break;
          }
        }
      }
      row.rep20 = rep_hits / 20.0;
      std::vector<int> last20(gen.begin() + 44, gen.end());
      std::sort(last20.begin(), last20.end());
      int uniq = static_cast<int>(
          std::unique(last20.begin(), last20.end()) - last20.begin());
      row.distinct20 = uniq / 20.0;
      std::string text;
      for (int id : gen) text += vocab[static_cast<size_t>(id)];
      row.text = text;
      sum_rep += row.rep20;
      sum_dist += row.distinct20;
      if (row.sane) sane_count++;
      rows.push_back(row);
      std::cerr << "[bench] suite prompt " << p << ": rep20 "
                << fmt(row.rep20, 3) << ", distinct20 "
                << fmt(row.distinct20, 3) << ", sane " << row.sane << "\n";
    }
    double mean_rep = sum_rep / suite.size();
    double mean_dist = sum_dist / suite.size();

    // ---- JSON output ----
    std::string js;
    js += "{\n";
    js += "  \"arm\": \"" + arm + "\",\n";
    js += "  \"model_file\": \"" + model_path + "\",\n";
    js += "  \"load_seconds\": " + fmt(load_s, 3) + ",\n";
    js += "  \"weight_storage_bytes\": " + std::to_string(model.weight_bytes) + ",\n";
    js += "  \"throughput\": {\n    \"prompt_tokens\": 64,\n    \"generated_tokens\": 128,\n    \"repetitions_tok_per_s\": [";
    for (size_t i = 0; i < rep_tps.size(); i++) {
      if (i) js += ", ";
      js += fmt(rep_tps[i], 4);
    }
    js += "],\n    \"repetitions_seconds\": [";
    for (size_t i = 0; i < rep_secs.size(); i++) {
      if (i) js += ", ";
      js += fmt(rep_secs[i], 4);
    }
    js += "],\n";
    js += "    \"mean\": " + fmt(tps.mean, 4) +
          ", \"std\": " + fmt(tps.stddev, 4) +
          ", \"median\": " + fmt(tps.median, 4) +
          ", \"min\": " + fmt(tps.mn, 4) +
          ", \"max\": " + fmt(tps.mx, 4) + "\n  },\n";
    js += "  \"vmhwm_mib_after_throughput\": " + fmt(vmhwm, 2) + ",\n";
    js += "  \"perplexity\": {\n    \"tokens_in_set\": " +
          std::to_string(ppl_ids.size()) +
          ",\n    \"predicted_tokens\": " + std::to_string(predicted) +
          ",\n    \"total_nll\": " + fmt(total_nll, 4) +
          ",\n    \"perplexity\": " + fmt(ppl, 6) + "\n  },\n";
    js += "  \"suite\": {\n    \"prompts\": [\n";
    for (size_t p = 0; p < rows.size(); p++) {
      js += "      {\"prompt\": \"" + json_escape_bytes(prompt_text[p]) +
            "\", \"rep20\": " + fmt(rows[p].rep20, 4) +
            ", \"distinct20\": " + fmt(rows[p].distinct20, 4) +
            ", \"sane\": " + (rows[p].sane ? "true" : "false") +
            ", \"generated_ids\": [";
      for (size_t i = 0; i < rows[p].gen.size(); i++) {
        if (i) js += ", ";
        js += std::to_string(rows[p].gen[i]);
      }
      js += "], \"generated_text\": \"" + json_escape_bytes(rows[p].text) +
            "\"}";
      js += (p + 1 < rows.size()) ? ",\n" : "\n";
    }
    js += "    ],\n    \"mean_rep20\": " + fmt(mean_rep, 4) +
          ", \"mean_distinct20\": " + fmt(mean_dist, 4) +
          ", \"sane_count\": " + std::to_string(sane_count) + "\n  }\n}\n";

    if (!out_path.empty()) {
      std::ofstream out(out_path);
      out << js;
      std::cerr << "[bench] wrote " << out_path << "\n";
    }
    std::cout << js;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
