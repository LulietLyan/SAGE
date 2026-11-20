#!/usr/bin/env bash
# Run under dsub from the payload directory. All installs are local to it.
set -euo pipefail
cd "$(dirname "$0")"
export LC_ALL=C
root=$PWD
build=$root/sage/.build
llvm=$build/llvm
omp=$build/openmp
py=$root/build-python/bin/python3.11
triton=$build/triton
arch=$(uname -m)
[[ "$arch" == aarch64 ]]
jobs=$(nproc)
[[ "$jobs" -le 48 ]]
date -Is
hostname -f
uname -srmo
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
( cd deps && sha256sum -c COMPILER_SHA256SUMS )
mkdir -p "$build" "$build/cache" "$build/json/include/nlohmann" "$root/build-python"
# The pinned SLEEF searches prefix/lib. GNUInstallDirs on Kylin uses lib64.
mkdir -p "$build/install/lib64"
if [[ ! -e "$build/install/lib" ]]; then
  ln -s lib64 "$build/install/lib"
fi
if [[ ! -x "$py" ]]; then
  tar -xf deps/python311-aarch64.tar.gz -C build-python --strip-components=1
fi
"$py" -m pip install --no-index --find-links deps/wheels cmake==4.1.2 ninja==1.13.0 \
  pybind11==3.0.1 setuptools==80.9.0 numpy==2.2.6 torch==2.7.1
"$py" -m pip check
export PATH="$llvm/bin:$root/build-python/bin:$PATH"
if [[ ! -f "$llvm/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
  mkdir -p "$llvm"
  tar -xf deps/llvm-aarch64-verified.tar.gz -C "$llvm" --strip-components=1
fi
export CC="$llvm/bin/clang" CXX="$llvm/bin/clang++"
"$CC" --version
llvm-config --version
llvm-config --targets-built
test -f "$llvm/lib/libLLVMAArch64CodeGen.a"
test -f "$llvm/lib/cmake/mlir/MLIRConfig.cmake"
if ! ldd "$CC" | tee "$root/clang-ldd.log" | (! grep -q 'not found'); then
  exit 4
fi
if [[ ! -f "$omp/include/omp.h" ]]; then
  mkdir -p "$omp/include" "$omp/lib" "$root/omp-archive"
  for package in "$root"/deps/libomp*.deb; do
    (cd "$root/omp-archive"; ar p "$package" data.tar.xz | tar xJf -)
  done
  header=$(find "$root/omp-archive" -type f -name omp.h -print -quit)
  runtime=$(find "$root/omp-archive" -type f -name libomp.so.5 -print -quit)
  cp "$header" "$omp/include/omp.h"
  cp "$runtime" "$omp/lib/libomp.so.5"
  ln -s libomp.so.5 "$omp/lib/libomp.so"
fi
export TRITON_LOCAL_LIBOMP_PATH="$omp"
bash "$root/prepare-cxx-runtime.sh"
export LD_LIBRARY_PATH="$build/cxx-runtime:$build/install/lib:$omp/lib:$llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
flags="-fopenmp=libomp -I$omp/include"
if [[ ! -f "$triton/CMakeLists.txt" ]]; then
  mkdir -p "$triton"
  tar -xf deps/triton-cpu-source.tar.gz -C "$triton" --strip-components=1
  ( cd "$triton"; git apply --check "$root/sage/patches/triton-cpu-sme.patch"; git apply "$root/sage/patches/triton-cpu-sme.patch" )
  "$py" patch-linux.py "$root/sage"
  mkdir -p "$triton/third_party/sleef"
  tar -xf deps/sleef-*.tar.gz -C "$triton/third_party/sleef" --strip-components=1
  mkdir -p "$triton/third_party/sleef/submodules/tlfloat"
  tar -xf deps/tlfloat-*.tar.gz -C "$triton/third_party/sleef/submodules/tlfloat" --strip-components=1
  mkdir -p "$triton/python/triton/language/extra"
  ln -s ../../../third_party/cpu/backend "$triton/python/triton/backends/cpu"
  ln -s ../../../../third_party/cpu/language/cpu "$triton/python/triton/language/extra/cpu"
fi
# Undo the exploratory SLEEF guard on resumed builds to restore the pinned source.
if grep -Fxq 'if(TARGET ext_tlfloat)' "$triton/third_party/sleef/src/libm/CMakeLists.txt"; then
  patch --batch --reverse -d "$triton/third_party/sleef" -p1 -i "$root/sleef-linux.patch"
fi
"$py" "$root/tlfloat-linux.py" "$triton/third_party/sleef/submodules/tlfloat"
cp deps/nlohmann-json-3.11.3.hpp "$build/json/include/nlohmann/json.hpp"
mkdir -p "$triton/python/triton/_C" "$build/native"
"$CXX" -O2 -march=armv9.2-a+sme "$root/sage/adapters/runtime/SmeProbe.cpp" \
  "$root/sage/adapters/runtime/SmeProbe.S" -o "$build/native/sme-probe"
"$build/native/sme-probe"
"$CXX" -std=c++17 -O3 -Wall -Wextra -Werror -fopenmp=libomp \
  -I"$omp/include" -L"$omp/lib" -Wl,-rpath,"$omp/lib" \
  "$root/sage/adapters/runtime/CompilerDeviceCalibration.cpp" -o "$build/native/compiler-device-calibration"
ldd "$build/native/compiler-device-calibration"
export SAGE_BUILD_CPU_LOCKS=$(mktemp -d "${TMPDIR:-/tmp}/sage-cpu-locks.XXXXXX")
trap 'find "$SAGE_BUILD_CPU_LOCKS" -maxdepth 1 -type f -delete; rmdir "$SAGE_BUILD_CPU_LOCKS" 2>/dev/null || true' EXIT
set -x
time cmake -S "$triton" -B "$build/cmake" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX="$build/install" \
  -DCMAKE_C_COMPILER_LAUNCHER="$py;$root/pin-compile.py" \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$py;$root/pin-compile.py" \
  -DPython3_EXECUTABLE="$py" -Dpybind11_DIR="$("$py" -m pybind11 --cmakedir)" \
  -DSME_EXTENSION_ROOT="$root/sage" -DTRITON_CODEGEN_BACKENDS=cpu \
  -DTRITON_BUILD_PYTHON_MODULE=ON -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_UT=OFF \
  -DTRITON_BUILD_WITH_CCACHE=OFF -DTRITON_OFFLINE_BUILD=ON \
  -DTRITON_VERSION=3.7.0+sme -DLLVM_SYSPATH="$llvm" \
  -DLLVM_DIR="$llvm/lib/cmake/llvm" -DMLIR_DIR="$llvm/lib/cmake/mlir" \
  -DJSON_SYSPATH="$build/json" -DTRITON_CACHE_PATH="$build/cache" \
  -DTRITON_WHEEL_DIR="$triton/python/triton" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$triton/python/triton/_C" \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$build/bin" \
  -DOpenMP_C_FLAGS="$flags" -DOpenMP_CXX_FLAGS="$flags" \
  -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY="$omp/lib/libomp.so" \
  -DSLEEF_ENABLE_SVE=OFF -DSLEEF_ENFORCE_PURECFMA_SCALAR=ON -DSLEEF_BUILD_TESTS=OFF \
  -DSLEEF_ENABLE_TLFLOAT=ON
time cmake --build "$build/cmake" --target triton TritonCPURuntime sleef --parallel "$jobs"
set +x
bash "$root/build-sme-abi.sh"
env PYTHONPATH="$triton/python" TRITON_BACKENDS_IN_TREE=1 "$py" - <<'PY'
from triton._C import libtriton
from triton.backends import backends
assert list(backends) == ["cpu"], backends
assert hasattr(libtriton.cpu.passes.ttcpuir, "add_plan_gemm")
print("TritonCPU compiler import and GEMM pass verified")
PY
sha256sum "$triton/python/triton/_C/libtriton.so" "$build/native/"*
ldd "$triton/python/triton/_C/libtriton.so"
date -Is
