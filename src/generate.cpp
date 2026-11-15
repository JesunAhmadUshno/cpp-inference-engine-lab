// engine_generate: command-line generation front end for the engine.
//
// Loads a safetensors checkpoint (fp32 or W8A32 int8 arm), runs
// greedy generation from a prompt given as token ids, and prints the
// prompt ids, the generated ids, and the decoded text when a vocab
// file is supplied. The benchmark harness (engine_bench) stays the
// frozen study protocol; this tool is the everyday entry point.
//
// Tokenization scope is the repo's standing one (PREREGISTRATION.md):
// the engine consumes token ids. For text prompts use
// tools/generate.py, which encodes with the repo's GPT-2 tokenizer
// and calls this binary.
//
// Usage:
//   engine_generate --model assets/model.safetensors --arm fp32 \
//       --prompt-ids "464 2106 286 262" --tokens 40 \
//       --vocab data/vocab.tsv
//
// Options:
//   --model PATH      safetensors checkpoint (required)
//   --arm NAME        fp32 | int8 (default fp32)
//   --heads N         attention heads in the checkpoint (default 12)
//   --prompt-ids S    prompt as space-separated token ids
//   --prompt-file P   file whose first line is the prompt ids
//   --tokens N        tokens to generate (default 40)
//   --vocab PATH      id-to-bytes table (data/vocab.tsv format);
//                     when given, the decoded text is printed too
#include <chrono>
#include <cstdio>
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

std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> ids;
  std::istringstream ss(s);
  int v;
  while (ss >> v) ids.push_back(v);
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string model_path = g_arg(argc, argv, "--model", "");
    const std::string arm = g_arg(argc, argv, "--arm", "fp32");
    const int heads = std::stoi(g_arg(argc, argv, "--heads", "12"));
    const int tokens = std::stoi(g_arg(argc, argv, "--tokens", "40"));
    const std::string vocab_path = g_arg(argc, argv, "--vocab", "");
    std::string prompt_str = g_arg(argc, argv, "--prompt-ids", "");
    const std::string prompt_file = g_arg(argc, argv, "--prompt-file", "");
    if (!prompt_file.empty()) {
      std::ifstream in(prompt_file);
      if (!in) throw std::runtime_error("cannot open prompt file: " +
                                       prompt_file);
      std::getline(in, prompt_str);
    }
    if (model_path.empty() || prompt_str.empty() ||
        (arm != "fp32" && arm != "int8")) {
      std::cerr
          << "usage: engine_generate --model PATH [--arm fp32|int8] "
             "[--heads N]\n"
             "    (--prompt-ids \"1 2 3\" | --prompt-file PATH) "
             "[--tokens N] [--vocab PATH]\n";
      return 2;
    }
    const std::vector<int> prompt = parse_ids(prompt_str);
    if (prompt.empty()) throw std::runtime_error("empty prompt");

    auto t0 = std::chrono::steady_clock::now();
    Model model = load_model(model_path, arm == "int8", heads);
    auto t1 = std::chrono::steady_clock::now();
    bool finite = true;
    std::vector<int> gen = generate_greedy(model, prompt, tokens, finite);
    auto t2 = std::chrono::steady_clock::now();
    const double gen_s = std::chrono::duration<double>(t2 - t1).count();
    std::fprintf(stderr,
                 "[generate] arm=%s loaded in %.2f s, generated %d tokens "
                 "in %.2f s (%.3f tok/s), finite_logits=%s\n",
                 arm.c_str(),
                 std::chrono::duration<double>(t1 - t0).count(), tokens,
                 gen_s, tokens / gen_s, finite ? "yes" : "no");

    std::cout << "prompt_ids:";
    for (int id : prompt) std::cout << " " << id;
    std::cout << "\ngenerated_ids:";
    for (int id : gen) std::cout << " " << id;
    std::cout << "\n";
    if (!vocab_path.empty()) {
      std::vector<std::string> vocab = read_vocab(vocab_path);
      std::string text;
      for (int id : prompt)
        if (id >= 0 && static_cast<size_t>(id) < vocab.size())
          text += vocab[static_cast<size_t>(id)];
      for (int id : gen)
        if (id >= 0 && static_cast<size_t>(id) < vocab.size())
          text += vocab[static_cast<size_t>(id)];
      std::cout << "text: " << text << "\n";
    }
    return finite ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
