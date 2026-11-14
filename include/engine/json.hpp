// Minimal JSON parser, written for this repo.
//
// It exists for one job: reading the header of a safetensors file.
// It implements the JSON grammar (objects, arrays, strings with
// escapes, numbers, true/false/null) and nothing else. No external
// dependency.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine {

struct Json {
  enum class Type { Null, Bool, Num, Str, Arr, Obj };
  Type type = Type::Null;
  bool boolean = false;
  double num = 0.0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;

  const Json* find(const std::string& key) const {
    if (type != Type::Obj) return nullptr;
    for (const auto& kv : obj) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }
};

namespace detail {

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : s_(text) {}

  Json parse() {
    Json v = parse_value();
    skip_ws();
    if (pos_ != s_.size()) fail("trailing characters after JSON value");
    return v;
  }

 private:
  const std::string& s_;
  size_t pos_ = 0;

  [[noreturn]] void fail(const std::string& msg) const {
    throw std::runtime_error("JSON parse error at byte " + std::to_string(pos_) +
                             ": " + msg);
  }

  void skip_ws() {
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        pos_++;
      } else {
        break;
      }
    }
  }

  char peek() {
    skip_ws();
    if (pos_ >= s_.size()) fail("unexpected end of input");
    return s_[pos_];
  }

  Json parse_value() {
    char c = peek();
    if (c == '{') return parse_obj();
    if (c == '[') return parse_arr();
    if (c == '"') {
      Json v;
      v.type = Json::Type::Str;
      v.str = parse_string();
      return v;
    }
    if (c == 't') {
      expect("true");
      Json v;
      v.type = Json::Type::Bool;
      v.boolean = true;
      return v;
    }
    if (c == 'f') {
      expect("false");
      Json v;
      v.type = Json::Type::Bool;
      v.boolean = false;
      return v;
    }
    if (c == 'n') {
      expect("null");
      return Json{};
    }
    return parse_num();
  }

  void expect(const char* lit) {
    size_t n = std::char_traits<char>::length(lit);
    if (s_.compare(pos_, n, lit) != 0) fail("invalid literal");
    pos_ += n;
  }

  Json parse_obj() {
    Json v;
    v.type = Json::Type::Obj;
    pos_++;  // {
    if (peek() == '}') {
      pos_++;
      return v;
    }
    while (true) {
      if (peek() != '"') fail("expected object key");
      std::string key = parse_string();
      if (peek() != ':') fail("expected ':' in object");
      pos_++;
      Json val = parse_value();
      v.obj.emplace_back(std::move(key), std::move(val));
      char c = peek();
      if (c == ',') {
        pos_++;
        continue;
      }
      if (c == '}') {
        pos_++;
        return v;
      }
      fail("expected ',' or '}' in object");
    }
  }

  Json parse_arr() {
    Json v;
    v.type = Json::Type::Arr;
    pos_++;  // [
    if (peek() == ']') {
      pos_++;
      return v;
    }
    while (true) {
      v.arr.push_back(parse_value());
      char c = peek();
      if (c == ',') {
        pos_++;
        continue;
      }
      if (c == ']') {
        pos_++;
        return v;
      }
      fail("expected ',' or ']' in array");
    }
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  uint32_t parse_hex4() {
    if (pos_ + 4 > s_.size()) fail("bad \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
      char c = s_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') {
        v |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        v |= static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        v |= static_cast<uint32_t>(c - 'A' + 10);
      } else {
        fail("bad hex digit in \\u escape");
      }
    }
    return v;
  }

  std::string parse_string() {
    // Caller has verified s_[pos_] == '"'.
    pos_++;
    std::string out;
    while (true) {
      if (pos_ >= s_.size()) fail("unterminated string");
      char c = s_[pos_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (pos_ >= s_.size()) fail("unterminated escape");
        char e = s_[pos_++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            uint32_t cp = parse_hex4();
            if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 1 < s_.size() &&
                s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
              pos_ += 2;
              uint32_t lo = parse_hex4();
              if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              } else {
                append_utf8(out, cp);
                cp = lo;
              }
            }
            append_utf8(out, cp);
            break;
          }
          default: fail("invalid escape");
        }
      } else {
        out.push_back(c);
      }
    }
  }

  Json parse_num() {
    size_t start = pos_;
    if (pos_ < s_.size() && s_[pos_] == '-') pos_++;
    bool any = false;
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
          c == '+' || c == '-') {
        pos_++;
        any = true;
      } else {
        break;
      }
    }
    if (!any) fail("invalid number");
    Json v;
    v.type = Json::Type::Num;
    v.num = std::stod(s_.substr(start, pos_ - start));
    return v;
  }
};

}  // namespace detail

inline Json parse_json(const std::string& text) {
  return detail::JsonParser(text).parse();
}

}  // namespace engine
