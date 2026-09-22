# Compiler source patch

The compiler starts from the public
[`triton-lang/triton-cpu` commit `8261cd0638ca4745bb14133ca586e3bce19654ef`](https://github.com/triton-lang/triton-cpu/commit/8261cd0638ca4745bb14133ca586e3bce19654ef).
`triton-cpu-sme.patch` adds the SME dialect/lowering, strict GEMM planning hook,
and compiler-owned CPU launcher preparation. It also retains the relevant
compiler regression tests and the macOS build-tool detection fix. CPU-only
CMake also supplies the upstream shared NVWS/AMD dialect and layout dependencies and omits
GPU instrumentation conversion; it does not register a GPU backend. Gluon
retains its common layout query; its AMD target query is compiled only when
the AMD backend is enabled. Proton keeps its common dialect and analysis;
backend-specific GPU conversions and their Python bindings follow the enabled
backend list. The standard
in-tree Python backend discovery constructor is corrected so installation
does not depend on another editable Triton package.

The patch changes 32 source/test/build files. The public base's generic Triton
frontend and CPU numerical lowering are retained. The project C++ policy and
early semantic pass are compiled directly into the CPU backend by setting
`SME_EXTENSION_ROOT`; their source remains under `lib/`, `include/` and
`adapters/`, rather than being duplicated in the patch.

Run `python scripts/build.py` to fetch and patch `.build/triton`, prepare the
dependencies, configure `.build/cmake` and compile all required objects from
source. The default build includes the Python compiler and CPU numerical
support libraries; it does not build unused GPU command-line tools.
`--configure-only` stops before compilation; `--jobs` defaults to two
and accepts one through four. No sibling checkout, old object file, cached
link command or prebuilt Triton compiler is used. Subsequent invocations use
normal Ninja incremental dependency checks inside this new build directory.

The native implementation currently targets macOS on Apple arm64. Build tools
and dependencies are found from the active Python environment, `CC`/`CXX`,
`LLVM_SYSPATH` and Homebrew's libomp headers. The pinned LLVM default is
`62b7cf9623fc310525f39ed69aaecc318a909731`, macOS arm64 package build 2; a fresh
download is checked against its upstream SHA-256. LLVM is a build dependency,
not a copied Triton result. Its LLVM/MLIR API is 23.0.0git, so an arbitrary
system LLVM version is not interchangeable.

SLEEF is explicitly fixed to public commit
[`7623d6cfa2712462880fa63a4d0f0b5f775d1a83`](https://github.com/shibatch/sleef/commit/7623d6cfa2712462880fa63a4d0f0b5f775d1a83),
matching the library source used when the compiler was developed. Its source
archive is checked against a fixed SHA-256. Its scalar helper dependency
TLFloat is pinned to public commit
[`4cc749ac08c910894a632a94afa68a157cb68d4c`](https://github.com/shibatch/tlfloat/commit/4cc749ac08c910894a632a94afa68a157cb68d4c)
and also downloaded with a fixed archive SHA-256. Both dependencies are built
from source; their optional tests are disabled. The SLEEF revision
deliberately overrides the base checkout's newer SLEEF gitlink. CMake rebuilds
SLEEF and the CPU support runtime. The generated launcher and PyTorch share
the active PyTorch installation's OpenMP library to avoid loading two runtimes.

Triton's license is retained in `LICENSE.triton`; fetched dependencies retain
their own licenses. The obsolete external SME runtime provider is omitted.
The supported compiler-plan path rejects external GEMM calls and generates
its SME arithmetic in each kernel's JIT binary. Hardware capability, emitted
instructions, actual execution, correctness and performance still require
their separate checks after a new build. Earlier measurements do not establish
the speed of a newly rebuilt compiler.
