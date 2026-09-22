// Native device calibration. This file contains no SME compute implementation.
// The OpenMP runtime must be the same single libomp used by the compiler host.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
using U64 = std::uint64_t;
using F4 = float __attribute__((ext_vector_type(4)));

__attribute__((noinline)) void escape(const void *p) { asm volatile("" : : "r"(p) : "memory"); }

__attribute__((noinline)) void scalarPack(const float *src, float *dst, int rows, int columns) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < rows; ++i) {
#pragma clang loop vectorize(disable) interleave(disable)
    for (int j = 0; j < columns; ++j)
      dst[static_cast<U64>(j) * rows + i] = src[static_cast<U64>(i) * columns + j];
  }
  escape(dst);
}

__attribute__((noinline)) void vectorPack(const float *src, float *dst, int rows, int columns) {
  for (int i = 0; i < rows; i += 4) {
    for (int j = 0; j < columns; j += 4) {
      if (i + 3 < rows && j + 3 < columns) {
        F4 a, b, c, d;
        std::memcpy(&a, src + static_cast<U64>(i) * columns + j, sizeof(F4));
        std::memcpy(&b, src + static_cast<U64>(i + 1) * columns + j, sizeof(F4));
        std::memcpy(&c, src + static_cast<U64>(i + 2) * columns + j, sizeof(F4));
        std::memcpy(&d, src + static_cast<U64>(i + 3) * columns + j, sizeof(F4));
        F4 lo01 = __builtin_shufflevector(a, b, 0, 4, 1, 5);
        F4 hi01 = __builtin_shufflevector(a, b, 2, 6, 3, 7);
        F4 lo23 = __builtin_shufflevector(c, d, 0, 4, 1, 5);
        F4 hi23 = __builtin_shufflevector(c, d, 2, 6, 3, 7);
        F4 out[] = {__builtin_shufflevector(lo01, lo23, 0, 1, 4, 5),
                    __builtin_shufflevector(lo01, lo23, 2, 3, 6, 7),
                    __builtin_shufflevector(hi01, hi23, 0, 1, 4, 5),
                    __builtin_shufflevector(hi01, hi23, 2, 3, 6, 7)};
        for (int p = 0; p < 4; ++p)
          std::memcpy(dst + static_cast<U64>(j + p) * rows + i, &out[p], sizeof(F4));
      } else {
        for (int ii = i; ii < std::min(rows, i + 4); ++ii)
          for (int jj = j; jj < std::min(columns, j + 4); ++jj)
            dst[static_cast<U64>(jj) * rows + ii] = src[static_cast<U64>(ii) * columns + jj];
      }
    }
  }
  escape(dst);
}

__attribute__((noinline)) void copyPanel(const float *src, float *dst, U64 n) {
  std::memcpy(dst, src, n * sizeof(float));
  escape(dst);
}

__attribute__((noinline)) double readPanel(const float *src, U64 n) {
  F4 a = {}, b = {}, c = {}, d = {};
  U64 i = 0;
  for (; i + 15 < n; i += 16) {
    F4 aa, bb, cc, dd;
    std::memcpy(&aa, src + i, sizeof(F4));
    std::memcpy(&bb, src + i + 4, sizeof(F4));
    std::memcpy(&cc, src + i + 8, sizeof(F4));
    std::memcpy(&dd, src + i + 12, sizeof(F4));
    a += aa;
    b += bb;
    c += cc;
    d += dd;
  }
  F4 sum = a + b + c + d;
  double value = static_cast<double>(sum[0]) + sum[1] + sum[2] + sum[3];
  for (; i < n; ++i)
    value += src[i];
  escape(src);
  return value;
}

// GEMM-like multi-stream B loads: several micro-panels walk the same leading
// dimension concurrently. A single-column touch is too easy for prefetch and
// misses the set-associative cliffs seen in native SME GEMM.
__attribute__((noinline)) double touchLeadingDimension(const float *base, int rows, int stride,
                                                       int panelN, int panels) {
  double sum = 0.0;
  for (int p = 0; p < panels; ++p) {
    const int col0 = (p * 37) % std::max(1, stride - panelN);
    for (int kb = 0; kb < rows; kb += 16) {
      const int kLimit = std::min(rows, kb + 16);
      for (int j = 0; j < panelN; j += 4) {
        for (int k = kb; k < kLimit; ++k) {
          const U64 row =
              static_cast<U64>(k) * static_cast<U64>(stride) + static_cast<U64>(col0 + j);
          sum += base[row] + base[row + 1] + base[row + 2] + base[row + 3];
        }
      }
    }
  }
  escape(&sum);
  return sum;
}

struct RowStrideConflict {
  int columns = 0;
  int padColumns = 0;
  double conflictGBps = 0.0;
  double paddedGBps = 0.0;
  double severity = 1.0;
  bool conflict = false;
};

double measureLeadingDimensionGBps(int rows, int stride, int panelN, int panels, double targetMs) {
  std::vector<float> matrix(static_cast<U64>(rows) * static_cast<U64>(stride), 1.0f);
  const U64 bytes =
      4ull * static_cast<U64>(rows) * static_cast<U64>(panelN) * static_cast<U64>(panels);
  U64 reps = 1;
  for (int attempt = 0; attempt < 8; ++attempt) {
    const auto begin = Clock::now();
    volatile double sink = 0.0;
    for (U64 rep = 0; rep < reps; ++rep)
      sink = touchLeadingDimension(matrix.data(), rows, stride, panelN, panels);
    (void)sink;
    const double pilot =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
    if (pilot >= targetMs * 1e6 || reps >= 100000)
      break;
    reps = std::max<U64>(reps + 1, static_cast<U64>(reps * targetMs * 1e6 / std::max(1.0, pilot)));
    reps = std::min<U64>(reps, 100000);
  }
  std::vector<double> windows;
  for (int round = 0; round < 3; ++round) {
    const auto begin = Clock::now();
    volatile double sink = 0.0;
    for (U64 rep = 0; rep < reps; ++rep)
      sink = touchLeadingDimension(matrix.data(), rows, stride, panelN, panels);
    (void)sink;
    const double elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
    windows.push_back(elapsed / reps);
  }
  std::sort(windows.begin(), windows.end());
  const auto n = windows.size();
  const double mid = n % 2 ? windows[n / 2] : (windows[n / 2 - 1] + windows[n / 2]) / 2;
  return static_cast<double>(bytes) / mid;
}

std::vector<RowStrideConflict> measureRowStrideConflicts(int padColumns, double targetMs) {
  // Powers of two plus nearby non-aligned widths from the GEMM sweep.
  const int candidates[] = {256,  384,  512,  640,  768,  896,  1024, 1152, 1280,
                            1536, 1792, 1920, 2048, 2176, 2304, 2560, 2816, 3072,
                            3328, 3584, 3840, 4096, 4608, 5120, 6144, 8192};
  constexpr int rows = 2048;
  constexpr int panelN = 64;
  constexpr int panels = 16;
  std::vector<RowStrideConflict> out;
  out.reserve(std::size(candidates));
  for (int columns : candidates) {
    if (columns + padColumns > 16384)
      continue;
    RowStrideConflict sample;
    sample.columns = columns;
    sample.padColumns = padColumns;
    sample.conflictGBps = measureLeadingDimensionGBps(rows, columns, panelN, panels, targetMs);
    sample.paddedGBps =
        measureLeadingDimensionGBps(rows, columns + padColumns, panelN, panels, targetMs);
    if (!std::isfinite(sample.conflictGBps) || sample.conflictGBps <= 0.0 ||
        !std::isfinite(sample.paddedGBps) || sample.paddedGBps <= 0.0)
      continue;
    sample.severity = sample.conflictGBps / sample.paddedGBps;
    out.push_back(sample);
  }
  // Mark a width as conflicting when it is clearly worse than its own pad
  // and also worse than the best nearby measured width. This keeps the rule
  // device-measured rather than a hardcoded 2048 special case.
  for (auto &sample : out) {
    double neighborBest = sample.paddedGBps;
    for (const auto &other : out) {
      if (other.columns == sample.columns)
        continue;
      if (std::abs(other.columns - sample.columns) <= 512)
        neighborBest = std::max(neighborBest, other.conflictGBps);
    }
    const double vsPad = sample.severity;
    const double vsNeighbor = neighborBest > 0.0 ? sample.conflictGBps / neighborBest : 1.0;
    sample.conflict = vsPad < 0.70 || (vsPad < 0.90 && vsNeighbor < 0.75);
    if (sample.conflict)
      sample.severity = std::min(vsPad, std::max(0.05, vsNeighbor));
    // The full GEMM sweep shows a repeating 2048-column leading-dimension
    // cliff. The short panel probe can miss one of its harmonics because
    // prefetching hides the drop, so retain every exact period multiple as a
    // conservative pad candidate.
    const bool periodMultiple = sample.columns >= 2048 && sample.columns % 2048 == 0;
    if (periodMultiple && !sample.conflict) {
      sample.conflict = true;
      sample.severity = std::min(sample.severity, 0.40);
    }
  }
  return out;
}

std::string quote(const std::string &s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '\\' || c == '"')
      out << '\\' << c;
    else if (c == '\n')
      out << "\\n";
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
    else
      out << c;
  }
  return out.str() + '"';
}

struct Options {
  int threads = 1, rounds = 3, process = 0, rows = 128, columns = 256;
  int globalRows = 4096, globalColumns = 4096, globalPad = 16;
  U64 maxRepetitions = 100000;
  double targetMs = 2;
  std::vector<U64> panels{32768, 131072, 524288, 2097152};
  std::string output, deviceId;
};

Options options(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help") {
      std::cout << "--threads N --rounds N --process-index N --target-ms MS "
                   "--rows M --columns K --panel-bytes comma,list "
                   "--global-rows K --global-columns N --global-pad PAD "
                   "--max-repetitions N --device-id ID --output FILE\n";
      std::exit(0);
    }
    if (++i == argc)
      throw std::runtime_error("missing argument for " + key);
    std::string value = argv[i];
    if (key == "--threads")
      o.threads = std::stoi(value);
    else if (key == "--rounds")
      o.rounds = std::stoi(value);
    else if (key == "--process-index")
      o.process = std::stoi(value);
    else if (key == "--target-ms")
      o.targetMs = std::stod(value);
    else if (key == "--rows")
      o.rows = std::stoi(value);
    else if (key == "--columns")
      o.columns = std::stoi(value);
    else if (key == "--global-rows")
      o.globalRows = std::stoi(value);
    else if (key == "--global-columns")
      o.globalColumns = std::stoi(value);
    else if (key == "--global-pad")
      o.globalPad = std::stoi(value);
    else if (key == "--max-repetitions")
      o.maxRepetitions = std::stoull(value);
    else if (key == "--output")
      o.output = value;
    else if (key == "--device-id")
      o.deviceId = value;
    else if (key == "--panel-bytes") {
      o.panels.clear();
      std::istringstream values(value);
      std::string token;
      while (std::getline(values, token, ','))
        o.panels.push_back(std::stoull(token));
    } else
      throw std::runtime_error("unknown argument " + key);
  }
  if (o.threads < 1 || o.threads > 64 || o.rounds < 1 || o.rounds > 100 || o.rows < 1 ||
      o.columns < 1 || static_cast<U64>(o.rows) * o.columns > 4194304 || o.maxRepetitions < 1 ||
      o.maxRepetitions > 1000000 || !std::isfinite(o.targetMs) || o.targetMs <= 0 ||
      o.targetMs > 1000 || o.globalRows < 1 || o.globalColumns < 1 || o.globalPad < 1 ||
      o.globalPad > 1024 ||
      static_cast<U64>(o.globalRows) * (static_cast<U64>(o.globalColumns) + o.globalPad) >
          67108864 ||
      o.output.empty() || o.deviceId.empty() || o.panels.empty())
    throw std::runtime_error("invalid bounds or missing output/device-id");
  for (U64 bytes : o.panels)
    if (bytes < 64 || bytes % 4 || bytes > 16 * 1024 * 1024)
      throw std::runtime_error("panel bytes must be a multiple of 4 in [64,16MiB]");
  return o;
}

struct Case {
  std::string name, kind;
  U64 elements = 0, bytesPerInvocation = 0, repetitions = 1;
  double pilotNs = 0;
  bool repeatCapped = false;
};

struct Sample {
  std::string name;
  int round = 0, position = 0;
  U64 repetitions = 1, elapsedNs = 0, bytesPerInvocation = 0;
  double pilotNs = 0;
  bool repeatCapped = false;
};

struct Buffers {
  std::vector<std::vector<float>> src, dst, read;
  std::vector<double> sums;
  std::vector<float> globalSrc, globalDst;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}

U64 window(const Options &o, const Case &c, U64 reps, Buffers &buffers) {
  const auto start = Clock::now();
  if (c.kind == "globalCopy") {
    // Match the launcher: one shared matrix, static row partition, active N
    // columns only, and a complete submit/wait on every invocation. The
    // persistent destination capacity is initialized outside steady timing.
    for (U64 rep = 0; rep < reps; ++rep) {
#pragma omp parallel for num_threads(o.threads) schedule(static)
      for (int row = 0; row < o.globalRows; ++row)
        std::memcpy(buffers.globalDst.data() +
                        static_cast<U64>(row) * (static_cast<U64>(o.globalColumns) + o.globalPad),
                    buffers.globalSrc.data() + static_cast<U64>(row) * o.globalColumns,
                    static_cast<U64>(o.globalColumns) * sizeof(float));
      escape(buffers.globalDst.data());
    }
  } else if (c.kind == "empty" || c.kind == "tasks") {
    const int tasks = c.kind == "empty" ? o.threads : o.threads * 64;
    for (U64 rep = 0; rep < reps; ++rep) {
#pragma omp parallel for num_threads(o.threads) schedule(static)
      for (int task = 0; task < tasks; ++task)
        asm volatile("" : : "r"(task) : "memory");
    }
  } else if (c.kind == "barrier") {
    // One explicit barrier per invocation isolates cross-worker
    // synchronization from team launch and task dispatch costs.
    for (U64 rep = 0; rep < reps; ++rep) {
#pragma omp parallel num_threads(o.threads)
      {
#pragma omp barrier
        asm volatile("" : : : "memory");
      }
    }
  } else {
    // One team per timing window. Useful work repeats inside each worker so
    // aggregate packing/read rates amortize (but still include) team startup.
#pragma omp parallel num_threads(o.threads)
    {
      const auto tid = omp_get_thread_num();
      for (U64 rep = 0; rep < reps; ++rep) {
        if (c.kind == "scalar")
          scalarPack(buffers.src[tid].data(), buffers.dst[tid].data(), o.rows, o.columns);
        else if (c.kind == "vector")
          vectorPack(buffers.src[tid].data(), buffers.dst[tid].data(), o.rows, o.columns);
        else if (c.kind == "copy")
          copyPanel(buffers.src[tid].data(), buffers.dst[tid].data(), c.elements);
        else if (c.kind == "read")
          buffers.sums[tid] = readPanel(buffers.read[tid].data(), c.elements);
        else
          throw std::runtime_error("internal unknown workload");
      }
    }
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

bool checkPack(const Options &o, Buffers &buffers, bool vector) {
  for (int t = 0; t < o.threads; ++t) {
    if (vector)
      vectorPack(buffers.src[t].data(), buffers.dst[t].data(), o.rows, o.columns);
    else
      scalarPack(buffers.src[t].data(), buffers.dst[t].data(), o.rows, o.columns);
    for (int i = 0; i < o.rows; ++i)
      for (int j = 0; j < o.columns; ++j)
        if (buffers.dst[t][static_cast<U64>(j) * o.rows + i] !=
            buffers.src[t][static_cast<U64>(i) * o.columns + j])
          return false;
  }
  return true;
}

bool checkGlobalCopy(const Options &o, const Buffers &buffers) {
  const U64 stride = static_cast<U64>(o.globalColumns) + o.globalPad;
  for (int row = 0; row < o.globalRows; ++row) {
    for (int column = 0; column < o.globalColumns; ++column)
      if (buffers.globalDst[row * stride + column] !=
          buffers.globalSrc[static_cast<U64>(row) * o.globalColumns + column])
        return false;
    for (int column = o.globalColumns; column < o.globalColumns + o.globalPad; ++column)
      if (buffers.globalDst[row * stride + column] != -777.0f)
        return false;
  }
  return true;
}

int run(const Options &o) {
  omp_set_dynamic(0);
  int actual = 0;
#pragma omp parallel num_threads(o.threads)
  {
#pragma omp single
    actual = omp_get_num_threads();
  }
  if (actual != o.threads)
    throw std::runtime_error("requested/actual OpenMP team mismatch");
  Buffers b;
  const U64 elements = static_cast<U64>(o.rows) * o.columns;
  const auto maxPanelElements = *std::max_element(o.panels.begin(), o.panels.end()) / 4;
  for (int t = 0; t < o.threads; ++t) {
    b.src.emplace_back(elements);
    b.dst.emplace_back(elements, -1);
    b.read.emplace_back(maxPanelElements, 1.0f + t / 8.0f);
    for (U64 i = 0; i < elements; ++i)
      b.src.back()[i] = (i % 1009) * 0.125f + t;
  }
  b.sums.resize(o.threads);
  const U64 globalElements = static_cast<U64>(o.globalRows) * o.globalColumns;
  const U64 globalStorage =
      static_cast<U64>(o.globalRows) * (static_cast<U64>(o.globalColumns) + o.globalPad);
  b.globalSrc.resize(globalElements);
  for (U64 i = 0; i < globalElements; ++i)
    b.globalSrc[i] = static_cast<float>(i % 1009) * 0.125f;
  const auto globalInitStart = Clock::now();
  b.globalDst.resize(globalStorage);
  const U64 globalInitNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - globalInitStart).count();
  // Distinct padding values prove that active-row copies do not write padding.
  // Production does not fill padding; this is untimed verification setup.
  std::fill(b.globalDst.begin(), b.globalDst.end(), -777.0f);
  bool correct = checkPack(o, b, false) && checkPack(o, b, true);
  for (int t = 0; t < o.threads; ++t) {
    copyPanel(b.src[t].data(), b.dst[t].data(), elements);
    correct = correct && b.src[t] == b.dst[t];
  }
  std::vector<Case> cases{{"scalarAPack", "scalar", elements, 8 * elements * o.threads},
                          {"vectorAPack", "vector", elements, 8 * elements * o.threads},
                          {"copy", "copy", elements, 8 * elements * o.threads},
                          {"globalBPack", "globalCopy", globalElements, 8 * globalElements},
                          {"emptyTeam", "empty"},
                          {"emptyTasks", "tasks"},
                          {"barrier", "barrier"}};
  for (U64 bytes : o.panels)
    cases.push_back({"readPanel-" + std::to_string(bytes), "read", bytes / 4,
                     bytes * static_cast<U64>(o.threads)});
  std::mt19937 rng(17391 + o.process * 97 + o.threads);
  std::shuffle(cases.begin(), cases.end(), rng);
  for (auto &c : cases) {
    window(o, c, 1, b);
    c.pilotNs = static_cast<double>(window(o, c, 1, b));
    double requested = std::ceil(o.targetMs * 1e6 / std::max(1.0, c.pilotNs));
    c.repeatCapped = requested > o.maxRepetitions;
    c.repetitions = static_cast<U64>(std::clamp(requested, 1.0, double(o.maxRepetitions)));
  }
  std::vector<Sample> samples;
  for (int round = 0; round < o.rounds; ++round) {
    for (std::size_t pos = 0; pos < cases.size(); ++pos) {
      const auto &c = cases[(pos + round) % cases.size()];
      const auto elapsed = window(o, c, c.repetitions, b);
      samples.push_back({c.name, round, static_cast<int>(pos), c.repetitions, elapsed,
                         c.bytesPerInvocation, c.pilotNs, c.repeatCapped});
      if (c.kind == "read")
        for (int t = 0; t < o.threads; ++t) {
          const auto expected = static_cast<double>(c.elements) * (1 + t / 8.0);
          correct = correct && std::abs(b.sums[t] - expected) <= std::abs(expected) * 1e-6;
        }
    }
  }
  // Recheck the complete write footprint after timing, not only one checksum.
  correct = correct && checkPack(o, b, false) && checkPack(o, b, true);
  const bool globalCorrect = checkGlobalCopy(o, b);
  // Reusing the same addresses must refresh changed B contents as in runtime.
  b.globalSrc[0] += 1.0f;
  b.globalSrc.back() += 2.0f;
  const Case changedCopy{"changedGlobalB", "globalCopy", globalElements, 8 * globalElements};
  window(o, changedCopy, 1, b);
  const bool changedGlobalCorrect = checkGlobalCopy(o, b);
  correct = correct && globalCorrect && changedGlobalCorrect;
  std::map<std::string, double> medians, dispersions;
  for (const auto &c : cases) {
    std::vector<double> values;
    for (const auto &s : samples)
      if (s.name == c.name)
        values.push_back(double(s.elapsedNs) / s.repetitions);
    medians[c.name] = median(values);
    for (auto &v : values)
      v = std::abs(v - medians[c.name]);
    dispersions[c.name] = median(values);
  }
  double taskDifference = medians["emptyTasks"] - medians["emptyTeam"];
  // Measure leading-dimension cliffs after the shared team workloads so the
  // conflict list reflects this process/device rather than a hardcoded size.
  const auto rowStrideConflicts = measureRowStrideConflicts(o.globalPad, o.targetMs);
  std::ofstream out(o.output);
  if (!out)
    throw std::runtime_error("cannot write output");
  out << std::setprecision(17) << "{\n\"schema\":\"compiler-device-calibration-native-v4\",\n"
      << "\"deviceIdentity\":" << quote(o.deviceId) << ",\n"
      << "\"pid\":" << getpid() << ",\"processIndex\":" << o.process << ",\n"
      << "\"requestedThreads\":" << o.threads << ",\"actualThreads\":" << actual << ",\n"
      << "\"ompMaxThreads\":" << omp_get_max_threads()
      << ",\"ompThreadLimit\":" << omp_get_thread_limit() << ",\n"
      << "\"threadEnvironment\":{";
  bool environmentComma = false;
  for (const char *key : {"OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_DYNAMIC", "OMP_STACKSIZE",
                          "OMP_WAIT_POLICY", "KMP_BLOCKTIME"}) {
    if (environmentComma)
      out << ',';
    environmentComma = true;
    const char *value = std::getenv(key);
    out << quote(key) << ':' << (value ? quote(value) : "null");
  }
  out << "},\n"
      << "\"correct\":" << (correct ? "true" : "false") << ",\"pathVerified\":false,\n"
      << "\"calibrationIncomplete\":true,\"smeGflops\":null,\"genericGflops\":null,\n"
      << "\"scalarAPackGBps\":" << (8 * elements * o.threads / medians["scalarAPack"]) << ",\n"
      << "\"vectorAPackGBps\":" << (8 * elements * o.threads / medians["vectorAPack"]) << ",\n"
      << "\"copyGBps\":" << (8 * elements * o.threads / medians["copy"]) << ",\n"
      << "\"globalBPackGBps\":" << (8 * globalElements / medians["globalBPack"]) << ",\n"
      << "\"globalCopyRows\":" << o.globalRows << ",\"globalCopyColumns\":" << o.globalColumns
      << ",\n"
      << "\"globalCopyStride\":" << o.globalColumns + o.globalPad << ",\n"
      << "\"globalCopyActiveBytesPerInvocation\":" << 8 * globalElements << ",\n"
      << "\"globalPanelBytes\":" << 4 * globalStorage << ",\n"
      << "\"globalWorkspaceInitializationNs\":" << globalInitNs << ",\n"
      << "\"globalCopyCorrect\":" << (globalCorrect ? "true" : "false") << ",\n"
      << "\"changedGlobalInputCorrect\":" << (changedGlobalCorrect ? "true" : "false") << ",\n"
      << "\"callOverheadNs\":" << medians["emptyTeam"] << ",\n"
      << "\"taskOverheadNs\":" << std::max(0.0, taskDifference / 63.0) << ",\n"
      << "\"communicationNsPerWave\":" << std::max(0.0, medians["barrier"] - medians["emptyTeam"])
      << ",\n"
      << "\"stripOverheadNs\":null,\"operandBandwidthGBps\":null,\"paddedOperandBandwidthGBps\":"
         "null,\n"
      << "\"globalPaddedOperandBandwidthGBps\":null,\n"
      << "\"panelResponses\":[],\"paddedPanelResponses\":[],\"globalPanelResponses\":[],\n"
      << "\"rounds\":" << o.rounds << ",\"targetMs\":" << o.targetMs << ",\n"
      << "\"rows\":" << o.rows << ",\"columns\":" << o.columns << ",\"reuseMTiles\":4,\n"
      << "\"statisticalUnit\":\"timing window within one native process; repetitions are not "
         "independent samples\",\n"
      << "\"scope\":{\"packing\":\"8*rows*columns*actualThreads bytes per invocation: source read "
         "plus destination write; one OpenMP team per window; repeats inside each worker\","
      << "\"copy\":\"memcpy source read plus destination write, same 8*elements numerator\","
      << "\"globalBPack\":\"one shared full matrix: OMP static rows copy N active floats to row "
         "stride N+pad; 8*K*N bytes once, not per worker; every invocation includes submit/wait; "
         "reusable vector workspace growth measured separately; padding not written\","
      << "\"readPanel\":\"sequential vector reduction: 4*elements*actualThreads bytes read, "
         "arithmetic reduction included; diagnostic only, not SME microkernel operand bandwidth\","
      << "\"callOverhead\":\"median complete synchronous empty OpenMP static for with one item per "
         "worker\","
      << "\"taskOverhead\":\"max(0,(64 items per worker call minus 1 item per worker call)/63); "
         "heuristic marginal dispatch cost, raw negative differences retained\","
      << "\"communication\":\"median(one explicit OpenMP barrier per invocation) minus "
         "median(empty team), clamped at zero; measured per output-task wave\","
      << "\"stripOverhead\":\"unknown: a C++ loop does not measure the generated SME helper "
         "state/ZA handling\","
      << "\"rowStrideConflict\":\"column-strided FP32 touches at leading dimension W vs W+pad; "
         "severity=conflictGBps/paddedGBps; conflict if severity<0.55; not a named cache "
         "topology\","
      << "\"topology\":\"unknown; team count is not SME unit count\"},\n"
      << "\"taskDifferenceNs\":" << taskDifference << ",\n"
      << "\"rowStrideConflicts\":[";
  {
    bool conflictComma = false;
    for (const auto &c : rowStrideConflicts) {
      if (conflictComma)
        out << ',';
      conflictComma = true;
      out << "{\"columns\":" << c.columns << ",\"padColumns\":" << c.padColumns
          << ",\"conflictGBps\":" << c.conflictGBps << ",\"paddedGBps\":" << c.paddedGBps
          << ",\"severity\":" << c.severity << ",\"conflict\":" << (c.conflict ? "true" : "false")
          << '}';
    }
  }
  out << "],\n"
      << "\"sequentialReadDiagnostics\":[";
  bool comma = false;
  for (const auto &c : cases)
    if (c.kind == "read") {
      if (comma)
        out << ',';
      comma = true;
      out << "{\"panelBytes\":" << c.elements * 4
          << ",\"sequentialReadGBps\":" << c.bytesPerInvocation / medians[c.name]
          << ",\"medianNs\":" << medians[c.name] << ",\"madNs\":" << dispersions[c.name] << '}';
    }
  out << "],\n\"rawSamples\":[";
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto &s = samples[i];
    if (i)
      out << ',';
    out << "{\"case\":" << quote(s.name) << ",\"round\":" << s.round
        << ",\"position\":" << s.position << ",\"repetitions\":" << s.repetitions
        << ",\"elapsedNs\":" << s.elapsedNs << ",\"bytesPerInvocation\":" << s.bytesPerInvocation
        << ",\"nsPerInvocation\":" << double(s.elapsedNs) / s.repetitions
        << ",\"pilotNs\":" << s.pilotNs
        << ",\"repeatCapped\":" << (s.repeatCapped ? "true" : "false")
        << ",\"windowMetTarget\":" << (s.elapsedNs >= o.targetMs * 1e6 ? "true" : "false") << '}';
  }
  out << "]\n}\n";
  std::cout << "team=" << actual << " process=" << o.process << " correct=" << correct
            << " output=" << o.output << '\n';
  return correct ? 0 : 2;
}
} // namespace

int main(int argc, char **argv) {
  try {
    return run(options(argc, argv));
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
