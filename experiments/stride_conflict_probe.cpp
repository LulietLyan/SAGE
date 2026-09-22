#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static inline void escape(const void *p) { asm volatile("" : : "g"(p) : "memory"); }

static double touch(const float *base, int rows, int stride, int panel, int panels) {
  double sum = 0.0;
  for (int p = 0; p < panels; ++p) {
    const int col0 = (p * 37) % std::max(1, stride - panel);
    for (int kb = 0; kb < rows; kb += 16)
      for (int j = 0; j < panel; j += 4)
        for (int k = kb; k < std::min(rows, kb + 16); ++k) {
          const auto row = static_cast<size_t>(k) * stride + col0 + j;
          sum += base[row] + base[row + 1] + base[row + 2] + base[row + 3];
        }
  }
  escape(&sum);
  return sum;
}

static double measure(int columns, int pad, int phase) {
  constexpr int rows = 2048, panel = 64, panels = 16;
  const int stride = columns + pad;
  const size_t elements = static_cast<size_t>(rows) * stride + phase + 64;
  std::vector<float> storage(elements, 1.0f);
  const float *base = storage.data() + phase;
  int reps = 1;
  for (;;) {
    auto begin = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
      touch(base, rows, stride, panel, panels);
    double ns =
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - begin).count();
    if (ns >= 2.0e6 || reps >= 10000) {
      const double bytes = 4.0 * rows * panel * panels * reps;
      return bytes / (ns / 1.0e9) / 1.0e9;
    }
    reps *= 2;
  }
}

int main() {
  std::cout << "columns,pad,phase,gbps\n";
  for (int columns = 1792; columns <= 2304; columns += 64)
    for (int pad : {0, 4, 8, 16, 32, 64})
      for (int phase : {0, 4, 8, 16})
        std::cout << columns << ',' << pad << ',' << phase << ',' << measure(columns, pad, phase)
                  << '\n';
}
