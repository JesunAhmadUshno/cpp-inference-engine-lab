// Safetensors loader, written for this repo.
//
// Parses the safetensors container directly: an 8-byte little-endian
// header length, a JSON header describing each tensor (dtype, shape,
// byte offsets into the payload), then the raw payload. The JSON is
// read with this repo's own minimal parser (json.hpp).
//
// Tensors are streamed from the file with pread in bounded chunks, so
// the loader never holds a second full copy of the checkpoint: the
// fp32 path reads each tensor straight into its final storage, and
// the quantization path makes two streaming passes per matrix.
#pragma once

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/json.hpp"

namespace engine {

struct TensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0;  // offsets relative to the payload start
  uint64_t end = 0;

  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

class SafetensorsFile {
 public:
  explicit SafetensorsFile(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("cannot open checkpoint file: " + path);
    }
    uint8_t lenbuf[8];
    read_exact(lenbuf, 8, 0);
    uint64_t header_len = 0;
    for (int i = 0; i < 8; i++) {
      header_len |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
    }
    std::string header(header_len, '\0');
    read_exact(reinterpret_cast<uint8_t*>(header.data()), header_len, 8);
    payload_start_ = 8 + header_len;
    Json root = parse_json(header);
    if (root.type != Json::Type::Obj) {
      throw std::runtime_error("safetensors header is not a JSON object");
    }
    for (const auto& kv : root.obj) {
      if (kv.first == "__metadata__") continue;
      const Json& t = kv.second;
      TensorInfo info;
      const Json* dt = t.find("dtype");
      const Json* sh = t.find("shape");
      const Json* off = t.find("data_offsets");
      if (!dt || !sh || !off || off->arr.size() != 2) {
        throw std::runtime_error("malformed tensor entry for " + kv.first);
      }
      info.dtype = dt->str;
      for (const Json& d : sh->arr) info.shape.push_back(static_cast<int64_t>(d.num));
      info.begin = static_cast<uint64_t>(off->arr[0].num);
      info.end = static_cast<uint64_t>(off->arr[1].num);
      tensors_[kv.first] = std::move(info);
    }
  }

  ~SafetensorsFile() {
    if (fd_ >= 0) ::close(fd_);
  }

  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;

  bool has(const std::string& name) const { return tensors_.count(name) > 0; }

  const TensorInfo& info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      throw std::runtime_error("tensor not found in checkpoint: " + name);
    }
    return it->second;
  }

  std::vector<std::string> names() const {
    std::vector<std::string> out;
    for (const auto& kv : tensors_) out.push_back(kv.first);
    return out;
  }

  // Stream a float32 tensor in chunks. fn receives (data, count,
  // element offset of the chunk start). Chunk size is bounded.
  void for_each_f32_chunk(
      const std::string& name,
      const std::function<void(const float*, size_t, size_t)>& fn) const {
    const TensorInfo& t = info(name);
    if (t.dtype != "F32") {
      throw std::runtime_error("tensor " + name + " has dtype " + t.dtype +
                               ", expected F32");
    }
    const size_t total = static_cast<size_t>(t.numel());
    const size_t kChunk = 1 << 18;  // 262144 floats = 1 MiB per read
    std::vector<float> buf(std::min(kChunk, total));
    size_t done = 0;
    while (done < total) {
      size_t n = std::min(kChunk, total - done);
      read_exact(reinterpret_cast<uint8_t*>(buf.data()), n * 4,
                 payload_start_ + t.begin + done * 4);
      fn(buf.data(), n, done);
      done += n;
    }
  }

  // Read a whole float32 tensor into a fresh vector.
  std::vector<float> read_f32(const std::string& name) const {
    const TensorInfo& t = info(name);
    std::vector<float> out(static_cast<size_t>(t.numel()));
    for_each_f32_chunk(name,
                       [&](const float* data, size_t n, size_t offset) {
                         std::memcpy(out.data() + offset, data, n * 4);
                       });
    return out;
  }

 private:
  void read_exact(uint8_t* dst, size_t n, uint64_t offset) const {
    size_t done = 0;
    while (done < n) {
      ssize_t r = ::pread(fd_, dst + done, n - done,
                          static_cast<off_t>(offset + done));
      if (r <= 0) {
        throw std::runtime_error("short read on checkpoint file");
      }
      done += static_cast<size_t>(r);
    }
  }

  int fd_ = -1;
  uint64_t payload_start_ = 0;
  std::map<std::string, TensorInfo> tensors_;
};

}  // namespace engine
