"""Provide the standard bit-cast semantics with Clang on libstdc++ 10."""
from pathlib import Path
import sys

root = Path(sys.argv[1])
header = root / "src/include/tlfloat/bigint.hpp"
initial = header.read_text()
if "constexpr To bit_cast" not in initial:
    old = "namespace tlfloat {"
    assert old in initial
    new = """namespace tlfloat {
  template<typename To, typename From>
  constexpr To bit_cast(const From &value) noexcept {
#if defined(__cpp_lib_bit_cast)
    return std::bit_cast<To>(value);
#else
    return __builtin_bit_cast(To, value);
#endif
  }
"""
    for path in (
        header, root / "src/include/tlfloat/tlfloat.hpp", root / "src/tlfloat/printf.cpp",
        root / "src/tlfloat/misc.cpp", root / "src/tester/test_printf.cpp",
    ):
        text = path.read_text().replace("std::bit_cast<", "tlfloat::bit_cast<")
        if path == header:
            text = text.replace(old, new, 1)
        path.write_text(text)
