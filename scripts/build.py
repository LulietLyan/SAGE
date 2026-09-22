#!/usr/bin/env python3
"""Fetch pinned sources and build the CPU compiler entirely inside .build/."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".build"
TRITON = BUILD / "triton"
UPSTREAM = "https://github.com/triton-lang/triton-cpu.git"
REVISION = "8261cd0638ca4745bb14133ca586e3bce19654ef"
SLEEF_REVISION = "7623d6cfa2712462880fa63a4d0f0b5f775d1a83"
SLEEF_URL = f"https://codeload.github.com/shibatch/sleef/tar.gz/{SLEEF_REVISION}"
SLEEF_SHA256 = "22affd077d1fa19f0052c39cfc14b8f2e1227da70dc7928e056bebca1d35ac53"
TLFLOAT_REVISION = "4cc749ac08c910894a632a94afa68a157cb68d4c"
TLFLOAT_SHA256 = "a8a2def5f1dcba6fb75d1d3485f4ef5aa27584e910d597852f634e831311e4b7"
PATCH = ROOT / "patches/triton-cpu-sme.patch"
LLVM_REVISION = "62b7cf9623fc310525f39ed69aaecc318a909731"
LLVM_URL = (
    "https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-62b7cf96-macos-arm64-2.tar.gz"
)
LLVM_SHA256 = "8ce251f8f89638e2d4af67a5a5bb027b2afa018d7a6ec8ad55b6b3c401aea745"


def run(arguments, *, cwd=ROOT, capture=False, check=True, env=None):
    if not capture:
        print("+ " + " ".join(map(str, arguments)), flush=True)
    return subprocess.run(
        list(map(str, arguments)),
        cwd=cwd,
        text=True,
        check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination):
    print("Downloading " + url, flush=True)
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def sources():
    if not TRITON.exists():
        TRITON.mkdir(parents=True)
        run(["git", "init", TRITON])
        run(["git", "remote", "add", "origin", UPSTREAM], cwd=TRITON)
    if not (TRITON / ".git").is_dir():
        raise RuntimeError("Existing .build/triton is not an isolated Git checkout")
    if (
        run(["git", "remote", "get-url", "origin"], cwd=TRITON, capture=True).stdout.strip()
        != UPSTREAM
    ):
        raise RuntimeError("Existing .build/triton does not have the recorded public upstream")
    revision = run(["git", "rev-parse", "HEAD"], cwd=TRITON, capture=True, check=False)
    if revision.returncode:
        if any(path.name != ".git" for path in TRITON.iterdir()):
            raise RuntimeError(
                "Incomplete checkout has independent files; refusing to replace them"
            )
        run(["git", "fetch", "--depth=1", "origin", REVISION], cwd=TRITON)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=TRITON)
    elif revision.stdout.strip() != REVISION:
        raise RuntimeError(
            "Existing .build/triton is not the pinned upstream revision; preserve it and use a new checkout"
        )
    forward = run(["git", "apply", "--check", PATCH], cwd=TRITON, capture=True, check=False)
    if forward.returncode == 0:
        run(["git", "apply", PATCH], cwd=TRITON)
    elif run(
        ["git", "apply", "--reverse", "--check", PATCH], cwd=TRITON, capture=True, check=False
    ).returncode:
        raise RuntimeError(
            "Compiler source does not match the complete patch; refusing to overwrite independent edits"
        )
    patched_paths = re.findall(r"^\+\+\+ b/(.+)$", PATCH.read_text(), re.M)
    # Check complete file contents against the public base plus patch, including
    # changes outside patch contexts, without modifying the checkout's index.
    with tempfile.TemporaryDirectory(prefix="patch-index-", dir=BUILD) as temporary:
        environment = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        run(["git", "read-tree", REVISION], cwd=TRITON, capture=True, env=environment)
        run(["git", "apply", "--cached", PATCH], cwd=TRITON, capture=True, env=environment)
        expected = run(
            ["git", "ls-files", "--stage", "--", *patched_paths],
            cwd=TRITON,
            capture=True,
            env=environment,
        ).stdout
        for line in expected.splitlines():
            entry, name = line.split("\t", 1)
            content = (TRITON / name).read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if blob != entry.split()[1]:
                raise RuntimeError("Compiler file differs from public base plus patch: " + name)
    allowed = set(patched_paths) | {"third_party/sleef"}
    dirty = run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=TRITON, capture=True
    ).stdout
    unexpected = [line[3:] for line in dirty.splitlines() if line[3:] not in allowed]
    if unexpected:
        raise RuntimeError(
            "Unexpected source changes in isolated checkout: " + ", ".join(unexpected)
        )
    sleef = TRITON / "third_party/sleef"
    source_archive("sleef", SLEEF_REVISION, SLEEF_SHA256, sleef)
    source_archive("tlfloat", TLFLOAT_REVISION, TLFLOAT_SHA256, sleef / "submodules/tlfloat")


def source_archive(project, revision, archive_sha256, destination):
    cache = BUILD / "cache"
    cache.mkdir(exist_ok=True)
    archive = cache / f"{project}-{revision}.tar.gz"
    if not archive.exists():
        download(f"https://codeload.github.com/shibatch/{project}/tar.gz/{revision}", archive)
    if sha(archive) != archive_sha256:
        raise RuntimeError(f"Pinned {project} source archive SHA-256 mismatch")
    with tarfile.open(archive) as source:
        members = source.getmembers()
        prefix = f"{project}-{revision}/"
        if not (destination / "CMakeLists.txt").exists():
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Incomplete {project} source contains independent files")
            with tempfile.TemporaryDirectory(prefix=f"{project}-source-", dir=BUILD) as temporary:
                source.extractall(temporary, filter="data")
                extracted = Path(temporary) / prefix.rstrip("/")
                if destination.exists():
                    destination.rmdir()
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(extracted, destination)
        for member in members:
            if not member.isfile():
                continue
            if not member.name.startswith(prefix):
                raise RuntimeError(f"Unexpected {project} archive path")
            path = destination / member.name[len(prefix) :]
            expected = source.extractfile(member).read()
            if not path.is_file() or path.read_bytes() != expected:
                raise RuntimeError(
                    f"{project} source differs from its pinned archive: " + str(path)
                )


def llvm_dependency(path):
    if not path.exists():
        with tempfile.TemporaryDirectory(prefix="llvm-download-", dir=BUILD) as temporary:
            directory = Path(temporary)
            archive = directory / "llvm.tar.gz"
            download(LLVM_URL, archive)
            if sha(archive) != LLVM_SHA256:
                raise RuntimeError("Pinned LLVM archive SHA-256 mismatch")
            unpack = directory / "unpack"
            unpack.mkdir()
            with tarfile.open(archive) as source:
                source.extractall(unpack, filter="data")
            candidates = [candidate.parent.parent for candidate in unpack.rglob("bin/llvm-config")]
            if len(candidates) != 1:
                raise RuntimeError("Unexpected pinned LLVM archive layout")
            shutil.move(str(candidates[0]), path)
            (path / "version.txt").write_text(LLVM_URL)
    for relative in (
        "lib/cmake/llvm/LLVMConfig.cmake",
        "lib/cmake/mlir/MLIRConfig.cmake",
        "lib/libLLVMAArch64CodeGen.a",
        "bin/clang++",
        "bin/FileCheck",
    ):
        if not (path / relative).is_file():
            raise RuntimeError("Incomplete LLVM toolchain: " + str(path / relative))
    marker = path / "version.txt"
    if marker.exists() and marker.read_text().strip() != LLVM_URL:
        raise RuntimeError("LLVM dependency is not the pinned package")
    llvm_config = (path / "lib/cmake/llvm/LLVMConfig.cmake").read_text()
    if "set(LLVM_PACKAGE_VERSION 23.0.0git)" not in llvm_config:
        raise RuntimeError("The compiler requires its pinned LLVM 23 development API")


def json_dependency():
    directory = BUILD / "json"
    if (directory / "include/nlohmann/json.hpp").is_file():
        return directory
    if directory.exists():
        raise RuntimeError("Existing .build/json has no required headers")
    with tempfile.TemporaryDirectory(prefix="json-download-", dir=BUILD) as temporary:
        archive = Path(temporary) / "include.zip"
        download("https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip", archive)
        directory.mkdir()
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                if not (directory / name).resolve().is_relative_to(directory):
                    raise RuntimeError("Unsafe JSON dependency archive path")
            source.extractall(directory)
    if not (directory / "include/nlohmann/json.hpp").is_file():
        raise RuntimeError("Unexpected nlohmann/json include archive layout")
    return directory


def executable(name, override=None):
    result = str(override) if override else shutil.which(name)
    if result and not Path(result).is_file():
        result = shutil.which(result)
    if not result or not Path(result).is_file():
        raise RuntimeError(f"Required build tool is missing: {name}")
    return Path(result).absolute()


def openmp():
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or not torch_spec.origin:
        raise RuntimeError(
            "Install the project's Python requirements, including PyTorch, before building"
        )
    library = Path(torch_spec.origin).parent / "lib/libomp.dylib"
    if not library.is_file():
        raise RuntimeError("The installed PyTorch must provide its single libomp.dylib runtime")
    header = (
        Path(os.environ["LIBOMP_INCLUDE_DIR"]) if os.environ.get("LIBOMP_INCLUDE_DIR") else None
    )
    if header is None and shutil.which("brew"):
        result = run(["brew", "--prefix", "libomp"], capture=True, check=False)
        if result.returncode == 0:
            header = Path(result.stdout.strip()) / "include"
    if header is None or not (header / "omp.h").is_file():
        raise RuntimeError(
            "Install libomp headers (brew install libomp), or set LIBOMP_INCLUDE_DIR"
        )
    prefix = BUILD / "openmp"
    prefix.mkdir(exist_ok=True)
    for name, target in (("include", header.resolve()), ("lib", library.parent.resolve())):
        link = prefix / name
        if link.is_symlink() and link.resolve() != target:
            link.unlink()
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        elif link.resolve() != target:
            raise RuntimeError("Unexpected existing OpenMP directory: " + str(link))
    return prefix, library.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configure-only", action="store_true", help="fetch, patch and configure without compiling"
    )
    parser.add_argument("--jobs", type=int, default=2, choices=range(1, 5))
    parser.add_argument(
        "--llvm", type=Path, default=Path(os.environ.get("LLVM_SYSPATH", BUILD / "llvm"))
    )
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        parser.error("The current native SME implementation supports macOS on Apple arm64")
    BUILD.mkdir(exist_ok=True)
    sources()
    llvm = args.llvm.resolve()
    llvm_dependency(llvm)
    json_dir = json_dependency()
    cmake, ninja = executable("cmake"), executable("ninja")
    cc = executable("clang", os.environ.get("CC") or llvm / "bin/clang")
    cxx = executable("clang++", os.environ.get("CXX") or llvm / "bin/clang++")
    try:
        import pybind11
    except ImportError as error:
        raise RuntimeError(
            "Install the project's Python requirements, including pybind11"
        ) from error
    omp_prefix, omp_library = openmp()
    sdk = run(["xcrun", "--show-sdk-path"], capture=True).stdout.strip()
    cmake_build = BUILD / "cmake"
    library_dir = TRITON / "python/triton/_C"
    library_dir.mkdir(exist_ok=True)
    # Materialize the standard CPU backend package locally, without sibling paths.
    for destination, origin in (
        ("python/triton/backends/cpu", "third_party/cpu/backend"),
        ("python/triton/language/extra/cpu", "third_party/cpu/language/cpu"),
    ):
        target, source = TRITON / destination, TRITON / origin
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and target.resolve() != source.resolve():
            raise RuntimeError("Existing Python backend link points outside the isolated source")
        if not target.exists():
            target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
    flags = "-fopenmp=libomp -I" + str(omp_prefix / "include")
    configure = [
        cmake,
        "-S",
        TRITON,
        "-B",
        cmake_build,
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCMAKE_INSTALL_PREFIX={BUILD / 'install'}",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
        f"-DCMAKE_C_COMPILER={cc}",
        f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DCMAKE_OSX_SYSROOT={sdk}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        f"-DSME_EXTENSION_ROOT={ROOT}",
        "-DTRITON_CODEGEN_BACKENDS=cpu",
        "-DTRITON_BUILD_PYTHON_MODULE=ON",
        "-DTRITON_BUILD_PROTON=OFF",
        "-DTRITON_BUILD_UT=OFF",
        "-DTRITON_BUILD_WITH_CCACHE=OFF",
        "-DTRITON_OFFLINE_BUILD=ON",
        "-DTRITON_VERSION=3.7.0+sme",
        f"-DLLVM_SYSPATH={llvm}",
        f"-DLLVM_DIR={llvm}/lib/cmake/llvm",
        f"-DMLIR_DIR={llvm}/lib/cmake/mlir",
        f"-DJSON_SYSPATH={json_dir}",
        f"-DTRITON_CACHE_PATH={BUILD / 'cache'}",
        f"-DTRITON_WHEEL_DIR={TRITON / 'python/triton'}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={library_dir}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={BUILD / 'bin'}",
        f"-DOpenMP_C_FLAGS={flags}",
        f"-DOpenMP_CXX_FLAGS={flags}",
        "-DOpenMP_C_LIB_NAMES=omp",
        "-DOpenMP_CXX_LIB_NAMES=omp",
        f"-DOpenMP_omp_LIBRARY={omp_library}",
        "-DSLEEF_ENABLE_SVE=OFF",
        "-DSLEEF_ENFORCE_PURECFMA_SCALAR=ON",
        "-DSLEEF_BUILD_TESTS=OFF",
        "-DSLEEF_ENABLE_TLFLOAT=ON",
    ]
    environment = dict(
        os.environ,
        CC=str(cc),
        CXX=str(cxx),
        CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
        TRITON_LOCAL_LIBOMP_PATH=str(omp_prefix),
    )
    run(configure, env=environment)
    record = dict(
        upstream=UPSTREAM,
        revision=REVISION,
        sleef_revision=SLEEF_REVISION,
        sleef_archive_sha256=SLEEF_SHA256,
        tlfloat_revision=TLFLOAT_REVISION,
        tlfloat_archive_sha256=TLFLOAT_SHA256,
        llvm_revision=LLVM_REVISION,
        llvm=str(llvm),
        patch_sha256=sha(PATCH),
        build_script_sha256=sha(Path(__file__)),
        python=sys.executable,
        project_compiler_sources={
            str(path.relative_to(ROOT)): sha(path)
            for path in [
                ROOT / "lib/Core/CompilerGemmPolicy.cpp",
                ROOT / "include/sme_plan/CompilerGemmPolicy.h",
                ROOT / "include/sme_plan/ProblemFacts.h",
                ROOT / "include/sme_plan/TargetCapabilities.h",
                *sorted((ROOT / "adapters/triton_cpu").glob("CompilerGemm*.cpp")),
                *sorted((ROOT / "adapters/triton_cpu").glob("CompilerGemm*.h")),
            ]
        },
        triton_python=str(TRITON / "python"),
        compiler_library=str(library_dir / "libtriton.so"),
        cc=str(cc),
        cxx=str(cxx),
        openmp=str(omp_prefix),
        openmp_library=str(omp_library),
        build_dir=str(cmake_build),
        configure_argv=list(map(str, configure)),
        complete_source_build=True,
        retained_triton_objects_used=False,
        built=False,
    )
    if not args.configure_only:
        run(
            [
                cmake,
                "--build",
                cmake_build,
                "--target",
                "triton",
                "TritonCPURuntime",
                "sleef",
                "--parallel",
                args.jobs,
            ],
            env=environment,
        )
        import_environment = dict(
            environment, PYTHONPATH=str(TRITON / "python"), TRITON_BACKENDS_IN_TREE="1"
        )
        run(
            [
                sys.executable,
                "-c",
                "import triton; from triton.backends import backends; "
                "from triton._C import libtriton; assert list(backends) == ['cpu']; "
                "assert hasattr(libtriton.cpu.passes.ttcpuir, 'add_plan_gemm'); print('CPU compiler import passed')",
            ],
            env=import_environment,
        )
        record.update(
            built=True, import_verified=True, compiler_sha256=sha(library_dir / "libtriton.so")
        )
    (BUILD / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        "Configured compiler sources"
        if args.configure_only
        else "Built compiler: " + record["compiler_sha256"],
        flush=True,
    )


if __name__ == "__main__":
    main()
