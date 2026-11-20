# Linux ARM64 GEMM 复跑

本目录的脚本从同一个调度作业目录运行。用 `source-lock.tsv` 和 SHA-256 清单核对本机下载的 ARM64 输入，再经严格校验主机的普通 SSH 传输。将 SAGE 归档解包到作业目录的 `sage/`。其余源码、RPM、DEB 和 wheel 放在 `deps/`，wheel 放在 `deps/wheels/`。两个 SHA-256 清单复制到 `deps/`。对应文件名以脚本和清单为准。

服务器端先用 `dqueue` 确认账号可提交，在实际分配节点运行 `sme-check.sh`。它需要同目录的 `isa.c`、`fmopa.c` 和 `SmeProbeEncoded.S`。`SmeProbeEncoded.S` 是固定 SAGE 版本的轻量执行探针的同指令 ELF 版本，可用系统 GCC 汇编，源指令在 `SmeProbe.S` 中。原生编译只接受 Linux 对用户态暴露 SME_F32F32 且 SVL 为 64 字节的节点。

## 构建

下例在作业目录执行。将 `cn12088` 改为获准且探针通过的节点。

```sh
dsub -q q_hpcapp -nl cn12088 -a 'numa[count=2,distribution=balance]' \
  -R 'cpu=24,mem=32768MB' -T 6000 --cwd "$PWD" -oo "$PWD/compiler-build.log" \
  /bin/bash "$PWD/build-linux.sh"
dsub -q q_hpcapp -nl cn12088 -a 'numa[count=1,distribution=pack]' \
  -R 'cpu=12,mem=4096MB' -T 1800 --cwd "$PWD" -oo "$PWD/openblas-sve.log" \
  /bin/bash "$PWD/build-openblas.sh" ARMV8SVE
```

OpenBLAS 的 `ARMV9SME` 构建同样使用 `build-openblas.sh`，依赖已解包的 Clang 23。Linux 构建保留固定 SLEEF 和 TLFloat 源码版本。TLFloat 通过 Clang 内建位转换适配 GCC 10 的 C++ 标准库。Linux JIT 审计使用 GNU Binutils 的实际 ELF 反汇编。SAGE 的 Python 审计接口通过 `tool-bin/llvm-objdump` 兼容调用该工具。

独立 PyTorch/OpenBLAS 基线可用与构建同版本的 Python。将 `deps/python311-aarch64.tar.gz` 解包至 `baseline-python/`，在计算作业内用该目录的 `bin/python3.11 -m pip install --no-index --find-links deps/wheels numpy==2.2.6 torch==2.7.1 setuptools==80.9.0` 安装。需要重现早期 wheel 检查时另装 `scipy-openblas32==0.3.34.0.0`。

## 验证和计时

以下命令放在经 `dsub` 获准的计算作业内执行。单 NUMA 需要至少 8 个同 NUMA CPU。跨 NUMA 8 线程需要两组各至少 4 个 CPU。示例 `numa[count=2,distribution=balance]` 和 16 核预算会给双 NUMA 各 8 核。主脚本通过 `numactl` 设置计算 CPU 和内存策略，OpenMP 工作线程按核绑定，pthread OpenBLAS 通过其公开 API 逐线程绑定并验证。

```sh
bash run-linux.sh experiment-v3/native-smoke.py native-checks
bash run-linux.sh experiment-v3/calibrate-bind.py single calibration-single
bash run-linux.sh experiment-v3/calibrate-bind.py cross calibration-cross

bash run-linux.sh experiment-v3/gemm.py --output formal-single --topology single \
  --methods generic sme-fixed ours openblas pytorch --threads 1 8 \
  --profile calibration-single/profile.json --openblas "$PWD/openblas-ARMV8SVE/lib/libopenblas.so"

bash run-linux.sh experiment-v3/gemm.py --output formal-cross --topology cross \
  --methods generic sme-fixed ours openblas pytorch --threads 8 \
  --profile calibration-cross/profile.json --openblas "$PWD/openblas-ARMV8SVE/lib/libopenblas.so"
```

调度消融使用相同的 holdout 计时器。重点方法是 `ours`、`fixed-control`、`vector-a`、`tile-m`、`k-strip`、`local-b` 和 `global-b`。可追加排序候选 `ours-rank-2`、`ours-rank-5` 等。定形状消融推荐 `--shapes 1024x1024x1024 2049x2049x2049`，候选配置和实际调用路径见 checks.csv 的证据列。

在本机使用 Python 3.11+ 和 matplotlib 生成汇总 CSV 与图。例 `python experiment-v3/summarize.py formal-single --data data --figures figures`。分析脚本检查校验行、进程和窗口的一致性，按每个独立进程的窗口中位数配对。每个输出目录须为新路径。
