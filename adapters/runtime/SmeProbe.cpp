// Standalone SME/SVL execution evidence, not a GEMM backend or a benchmark.
// A caller launches this process to contain unsupported-instruction failures.
#include <array>
#include <cstdint>
#include <iostream>

#if !defined(__aarch64__)
#error "The SME execution probe requires an AArch64 target"
#endif

extern "C" std::uint64_t sme_probe_multiply(float *, const float *, const float *);

int main() {
  std::array<float, 3> output{-777.0f, -777.0f, -777.0f};
  float a = 2.0f, b = 3.0f;
  const auto width = sme_probe_multiply(&output[1], &a, &b);
  const float value = output[1];
  a = -4.0f;
  b = 1.25f;
  const auto secondWidth = sme_probe_multiply(&output[1], &a, &b);
  const bool guards = output[0] == -777.0f && output[2] == -777.0f;
  const bool passed = value == 6.0f && output[1] == -5.0f && guards && width == secondWidth &&
                      width >= 16 && width <= 256 && width % 16 == 0;
  std::cout << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"executed\":true,\"vector_length_bytes\":" << width << ",\"value\":" << value
            << ",\"changed_value\":" << output[1]
            << ",\"guard_untouched\":" << (guards ? "true" : "false") << "}\n";
  return passed ? 0 : 2;
}
