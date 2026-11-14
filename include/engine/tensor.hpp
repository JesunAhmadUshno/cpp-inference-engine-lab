// Tensor operations, thread pool, and the int8 quantization path,
// written for this repo. Standard library only.
//
// Layout convention: the engine stores every linear weight in compute
// orientation, W[out][in] row-major (the transpose of the stored
// Conv1D matrices in the checkpoint). y[j] = sum_i W[j][i] * x[i].
// Matmul output rows are split across a fixed pool of 4 threads; each
// output element is accumulated by exactly one thread in a fixed
// order, so results are deterministic and thread-count independent.
#pragma once

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace engine {

// ----------------------------------------------------------------
// Thread pool: fixed size, generation-counter dispatch.
class Pool {
 public:
  explicit Pool(int threads) : n_(threads) {
    for (int i = 1; i < n_; i++) {
      workers_.emplace_back([this, i] { worker_loop(i); });
    }
  }

  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
      gen_++;
    }
    cv_work_.notify_all();
    for (auto& t : workers_) t.join();
  }

  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  int size() const { return n_; }

  // Run fn(chunk_id) for chunk_id in [0, n_), caller takes chunk 0,
  // and block until every chunk finished.
  void run(const std::function<void(int)>& fn) {
    {
      std::lock_guard<std::mutex> lk(m_);
      task_ = fn;
      completed_ = 0;
      gen_++;
    }
    cv_work_.notify_all();
    fn(0);
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] { return completed_ == n_ - 1; });
  }

  // Split [0, total) into n_ contiguous ranges and apply fn per range.
  void parallel_for(int64_t total, const std::function<void(int64_t, int64_t)>& fn) {
    if (total <= 0) return;
    int64_t chunk = (total + n_ - 1) / n_;
    run([&](int cid) {
      int64_t lo = cid * chunk;
      int64_t hi = std::min(total, lo + chunk);
      if (lo < hi) fn(lo, hi);
    });
  }

 private:
  void worker_loop(int id) {
    uint64_t seen = 0;
    while (true) {
      std::function<void(int)> task;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_work_.wait(lk, [&] { return gen_ != seen; });
        seen = gen_;
        if (stop_) return;
        task = task_;
      }
      task(id);
      {
        std::lock_guard<std::mutex> lk(m_);
        completed_++;
      }
      cv_done_.notify_all();
    }
  }

  int n_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  std::function<void(int)> task_;
  int completed_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

// The engine's single pool: 4 threads, frozen by PREREGISTRATION.md.
inline Pool& pool() {
  static Pool p(4);
  return p;
}

// ----------------------------------------------------------------
// Dense fp32 matrix, row-major [rows][cols].
struct Matrix {
  int rows = 0;
  int cols = 0;
  std::vector<float> a;

  Matrix() = default;
  Matrix(int r, int c) : rows(r), cols(c), a(static_cast<size_t>(r) * c, 0.0f) {}

  const float* row(int r) const { return a.data() + static_cast<size_t>(r) * cols; }
  float* row(int r) { return a.data() + static_cast<size_t>(r) * cols; }
  size_t bytes() const { return a.size() * sizeof(float); }
};

// Quantized matrix: int8 payload in compute orientation [out][in],
// one fp32 scale per output row (per-output-channel symmetric).
struct QMatrix {
  int rows = 0;  // out features
  int cols = 0;  // in features
  std::vector<int8_t> q;
  std::vector<float> scale;  // rows entries

  size_t bytes() const { return q.size() + scale.size() * sizeof(float); }
};

// Quantize one fp32 matrix (compute orientation) per output row:
// scale_j = max_i |W[j][i]| / 127, q = clamp(round(W / scale)).
inline QMatrix quantize_rows(const Matrix& w) {
  QMatrix out;
  out.rows = w.rows;
  out.cols = w.cols;
  out.q.resize(w.a.size());
  out.scale.resize(w.rows);
  pool().parallel_for(w.rows, [&](int64_t j0, int64_t j1) {
    for (int64_t j = j0; j < j1; j++) {
      const float* row = w.row(static_cast<int>(j));
      float mx = 0.0f;
      for (int i = 0; i < w.cols; i++) mx = std::max(mx, std::fabs(row[i]));
      float s = mx / 127.0f;
      if (s == 0.0f) s = 1.0f;  // all-zero row: scale is arbitrary, values are 0
      out.scale[static_cast<size_t>(j)] = s;
      int8_t* qr = out.q.data() + static_cast<size_t>(j) * w.cols;
      for (int i = 0; i < w.cols; i++) {
        float v = std::round(row[i] / s);
        v = std::max(-127.0f, std::min(127.0f, v));
        qr[i] = static_cast<int8_t>(v);
      }
    }
  });
  return out;
}

// ----------------------------------------------------------------
// y = W x (fp32). W is [out][in], x has in elements, y has out.
inline void matvec(const Matrix& w, const float* x, float* y) {
  pool().parallel_for(w.rows, [&](int64_t j0, int64_t j1) {
    for (int64_t j = j0; j < j1; j++) {
      const float* row = w.row(static_cast<int>(j));
      float acc = 0.0f;
      for (int i = 0; i < w.cols; i++) acc += row[i] * x[i];
      y[j] = acc;
    }
  });
}

// y = W x with dequantize-on-the-fly int8 weights (W8A32 scheme):
// y[j] = scale_j * sum_i q[j][i] * x[i], fp32 accumulation.
inline void matvec_q(const QMatrix& w, const float* x, float* y) {
  pool().parallel_for(w.rows, [&](int64_t j0, int64_t j1) {
    for (int64_t j = j0; j < j1; j++) {
      const int8_t* row = w.q.data() + static_cast<size_t>(j) * w.cols;
      float acc = 0.0f;
      for (int i = 0; i < w.cols; i++) acc += static_cast<float>(row[i]) * x[i];
      y[j] = acc * w.scale[static_cast<size_t>(j)];
    }
  });
}

// Y = X W^T for a batch of row vectors: X is [t][in] row-major with
// leading dimension ldx, Y is [t][out] with leading dimension ldy.
inline void matmat(const Matrix& w, const float* x, int t, int ldx, float* y,
                   int ldy) {
  pool().parallel_for(t, [&](int64_t t0, int64_t t1) {
    for (int64_t r = t0; r < t1; r++) {
      const float* xr = x + r * ldx;
      float* yr = y + r * ldy;
      for (int j = 0; j < w.rows; j++) {
        const float* wrow = w.row(j);
        float acc = 0.0f;
        for (int i = 0; i < w.cols; i++) acc += wrow[i] * xr[i];
        yr[j] = acc;
      }
    }
  });
}

inline void matmat_q(const QMatrix& w, const float* x, int t, int ldx, float* y,
                     int ldy) {
  pool().parallel_for(t, [&](int64_t t0, int64_t t1) {
    for (int64_t r = t0; r < t1; r++) {
      const float* xr = x + r * ldx;
      float* yr = y + r * ldy;
      for (int j = 0; j < w.rows; j++) {
        const int8_t* wrow = w.q.data() + static_cast<size_t>(j) * w.cols;
        float acc = 0.0f;
        for (int i = 0; i < w.cols; i++) acc += static_cast<float>(wrow[i]) * xr[i];
        yr[j] = acc * w.scale[static_cast<size_t>(j)];
      }
    }
  });
}

// ----------------------------------------------------------------
// Elementwise and row ops.

inline void add_inplace(float* dst, const float* src, int n) {
  for (int i = 0; i < n; i++) dst[i] += src[i];
}

inline void add_bias(float* x, const float* bias, int rows, int cols, int ld) {
  for (int r = 0; r < rows; r++) {
    float* row = x + static_cast<size_t>(r) * ld;
    for (int c = 0; c < cols; c++) row[c] += bias[c];
  }
}

// LayerNorm over each row of x ([rows][cols], leading dimension ld),
// GPT-2 form: biased variance, epsilon inside the square root.
inline void layernorm(float* x, int rows, int cols, int ld, const float* gain,
                      const float* bias, float eps) {
  pool().parallel_for(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; r++) {
      float* row = x + r * ld;
      double mean = 0.0;
      for (int c = 0; c < cols; c++) mean += row[c];
      mean /= cols;
      double var = 0.0;
      for (int c = 0; c < cols; c++) {
        double d = row[c] - mean;
        var += d * d;
      }
      var /= cols;
      double inv = 1.0 / std::sqrt(var + eps);
      for (int c = 0; c < cols; c++) {
        row[c] = static_cast<float>((row[c] - mean) * inv * gain[c] + bias[c]);
      }
    }
  });
}

// GELU, tanh approximation (gelu_new, the GPT-2 default activation).
inline float gelu_tanh(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  float inner = c * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

inline void gelu_inplace(float* x, int64_t n) {
  pool().parallel_for(n, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; i++) x[i] = gelu_tanh(x[i]);
  });
}

// Softmax over a row of n logits, in place, max-subtracted.
inline void softmax(float* x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; i++) mx = std::max(mx, x[i]);
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    x[i] = std::exp(x[i] - mx);
    sum += x[i];
  }
  float inv = static_cast<float>(1.0 / sum);
  for (int i = 0; i < n; i++) x[i] *= inv;
}

// Index of the maximum element; ties go to the lowest index.
inline int argmax(const float* x, int n) {
  int best = 0;
  for (int i = 1; i < n; i++) {
    if (x[i] > x[best]) best = i;
  }
  return best;
}

}  // namespace engine
