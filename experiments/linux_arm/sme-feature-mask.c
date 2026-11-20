// Linux runtime supplies the FMV feature word read by LLVM's pinned SME ABI.
// The imported SME ABI implementation consumes exactly these three bits.
#include <stdint.h>
#include <sys/auxv.h>

__attribute__((visibility("hidden"))) uint64_t __aarch64_cpu_features;

__attribute__((constructor))
static void load_linux_features(void) {
    unsigned long caps = getauxval(AT_HWCAP);
    unsigned long caps2 = getauxval(AT_HWCAP2);
    __aarch64_cpu_features =
        (((uint64_t)caps >> 22) & 1) << 30 |
        (((uint64_t)caps2 >> 23) & 1) << 42 |
        (((uint64_t)caps2 >> 37) & 1) << 57;
}
