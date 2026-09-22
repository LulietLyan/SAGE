# 开发指南

[返回 README](../README.md) · [编译器补丁](../patches/README.md)

## 代码分工

独立 C++ 策略保留在 `include/` 和 `lib/`，不依赖 MLIR。`adapters/triton_cpu/` 把已证明的 GEMM 事实交给策略，并将决定写回 IR；`adapters/runtime/` 负责原生探测与校准。Python 包名仍为 `sme_plan`，分发名为 `sage-triton-cpu`，项目名称为 SAGE。

README 中的 SAGE 方法在代码与 CSV 中仍使用内部模式名 `ours`。

`patches/triton-cpu-sme.patch` 应始终对应说明中固定的上游版本，不要直接格式化统一 diff。更新补丁时，在目标开发环境校验其可应用性并保留上游许可。项目 C++ 源码通过 `SME_EXTENSION_ROOT` 编入 TritonCPU，无需复制到补丁。

## 格式与注释

- Python 使用 Ruff，100 列，配置位于 `pyproject.toml`。
- C++ 使用 LLVM 风格、2 空格缩进和 100 列，配置位于 `.clang-format`。
- `.editorconfig` 与 `.gitattributes` 统一 UTF-8 和 LF；补丁保留必需的上下文空格。
- 注释应解释假设、数值边界、数据所有权和测量口径。实测事实、启发式与尚未实现的分析应明确区分。

安装开发工具并进行静态检查：

```bash
python -m pip install -e '.[dev]'
ruff check python scripts experiments tests
ruff format --check python scripts experiments tests
git ls-files '*.cpp' '*.h' | xargs clang-format --dry-run --Werror
```

格式化会改变 Triton kernel 的源码标识，原有 profile 可能需要在目标机重新生成。不能通过替换 profile 中的哈希绕过这一检查。

## 测试入口

独立策略与 Python 测试：

```bash
python -m unittest discover -s tests/python
cmake -S . -B .build/core -DCMAKE_BUILD_TYPE=Debug
cmake --build .build/core
ctest --test-dir .build/core --output-on-failure
```

Python 测试覆盖校准家族隔离及 CLI 分派；独立 C++ 测试覆盖策略、资源限制、候选覆盖与排序。Apple arm64 上的 CMake 配置还需要 Torch、Homebrew 和 OpenMP。需要原生编译器/硬件的测试通过目标 Mac 上的 `./run.sh --test` 执行，包含尾部、别名拒绝、同地址输入变化、workspace 复用和布局检查。

独立策略测试或静态检查通过不代表 SME 运行时已验证，也不能证明性能提升。提交性能改动时请附目标设备、编译器/profile 标识、运行命令及原始测量。

## 仓库维护

源码、测试、补丁与说明进入 Git；`.build/`、`.venv/`、`output/`、Python 缓存与打包产物保持在 Git 之外。不要把设备 profile 当作通用默认配置发布。
