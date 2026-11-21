# 实验指南

[返回 README](../README.md) · [模型与实现](model.md)

所有命令均从仓库根目录执行。`main` 的原生测量协议面向支持 SME 的目标 Mac；Linux Arm v9 的独立构建和测量协议见 [Linux Arm v9 复现说明](../experiments/linux_arm/README.md)。源码仓库不保存远程机器的原始测量数据，正式结果包位于配套论文目录的 `linux-arm-v9/`。

## 常用入口

| 命令 | 工作 |
| --- | --- |
| `./run.sh` | 构建、按需校准、完整 GEMM 测量、绘图与汇总。 |
| `./run.sh --quick` | GEMM 仅测 128、1024、4096，每个线程预算 1 个进程、3 个窗口。 |
| `./run.sh --recalibrate` | 更新设备校准，再进行完整测量。 |
| `./run.sh --test` | 构建、策略与 Python 检查、按需校准及原生运行时检查。 |
| `./run.sh --attention` | Attention 完整调用及分阶段测量。 |
| `./run.sh --kan` / `--swiglu` | KAN / SwiGLU 完整前向测量。 |
| `./run.sh --suite-v2` | 8 线程密集 GEMM、Linear、BMM 及汇总绘图。 |
| `./run.sh --plot-only` | 重绘已有 GEMM 数据并更新报告。 |
| `./run.sh --attention --plot-only` | 仅重绘已有 Attention 数据；KAN、SwiGLU 同理。 |
| `./run.sh --plot-only --enrich-figures` | 为已有 CSV 生成调度及跨算子图。 |

`--quick` 不缩减构建与校准。以上命令的分派见 [`scripts/run.py`](../scripts/run.py)。首次运行须有网络与足够磁盘空间；之后复用 `.build/`。编译器或 Triton GEMM 源码变更会触发重新校准，包括改变源码标识的格式化。

需要独立执行时，先按 README 完成目标机设置：

```bash
.venv/bin/python scripts/build.py --jobs 2
.venv/bin/python experiments/calibrate.py --output output
.venv/bin/python experiments/gemm.py --output output
.venv/bin/python experiments/plot.py --output output
```

`experiments/run_suite_v2.sh` 是已有环境下的补充批处理入口，还包括 Attention 和 SwiGLU；它不负责构建或校准。名称相近的 `./run.sh --suite-v2` 仅包括 GEMM、Linear、BMM。

## 测量口径

默认 [GEMM](../experiments/gemm.py) 比较 `generic`、`sme-fixed`、`ours`、`pytorch`，采用 128–4096 的 24 个方阵尺寸、1/8 线程预算、每预算 5 个独立进程及每方法每尺寸 4 个交错窗口。窗口目标为 25 ms；实际耗时、重复次数和调整原因均保存。

[共享计时器](../experiments/common.py) 测量完整同步调用，包括每次的打包、复制、提交与等待。每进程先取窗口中位数，再跨进程汇总；误差范围为 min–max，不是置信区间。加速比在同一进程内配对。吞吐量使用逻辑 `2*M*N*K / median_ns`。

这里的完整调用指已准备输入和缓冲区的稳态调用。GEMM 的输入生成、输出分配与正确性检查不计时。Attention 的编译器路径提前完成 `K.t().contiguous()` 与主要缓冲区分配，而 PyTorch SDPA 接收原始 K；其完整调用比较不包含编译器路径的这次布局准备成本。

校准样本不加入正式性能曲线。编译/缓存查询与首次调用另记；进程共享 JIT 缓存，因此不是多次独立冷启动。PyTorch 冷启动未测量。请求的线程预算不等于 SME 物理单元数，Accelerate 的内部调度也没有直接观测。

[编译检查](../python/sme_plan/compiler_backend.py) 检查 LLVM 入口、JIT 反汇编中的 `fmopa`、外部符号及实际调用。数值检查由各实验另外执行。GEMM、Attention 的主实验会记录完成状态与编译器/profile 身份；其他诊断脚本的证据完整性不同，不应把所有 CSV 视为同等严格的实验结果。

## 应用层算子

| 实验 | 编译器承担的部分 | 其余工作与测量边界 |
| --- | --- | --- |
| [Attention](../experiments/attention.py) | QKᵀ 与 PV 两次 GEMM。 | batch=heads=1、head dimension=64，默认 26 个长度（32–2048）；缩放与 softmax 在应用层。PyTorch 完整基线使用 SDPA。 |
| [KAN](../experiments/kan.py) | SiLU base 与 B-spline 系数的两次投影。 | 完整调用包含 basis expansion、激活和求和；GFLOP/s 只计两次稠密投影。 |
| [SwiGLU](../experiments/swiglu.py) | 三次稠密投影。 | 完整调用包含 SiLU 和逐元素乘法。 |
| [Linear](../experiments/linear.py) | 无 bias 的 `X @ W`。 | 与 `torch.mm` 对比。 |
| [BMM](../experiments/bmm.py) | 每个 batch slice 的 GEMM。 | 应用层逐 slice 调用，与 `torch.bmm` 对比。 |

这些实验不构成自动整图融合或非线性算子降级。Attention 完整 FLOPs 采用 `4*L*L*d + 6*L*L` 的计数约定，非线性阶段以延迟为主要指标。其独立 PyTorch QK 阶段当前包含缩放，而 scale 阶段还单独计时，因此阶段柱状图不应相加解释为完整 SDPA 延迟。

各入口的默认协议不同，不能统一标注为五进程结果：

| 入口 | 每线程预算进程数 | 每方法/用例窗口数 | 窗口目标 |
| --- | --- | --- | --- |
| 默认 GEMM | 5 | 4 | 25 ms |
| Attention | 3 | 3 | 15 ms |
| `--suite-v2` 密集 GEMM | 3 | 3 | 25 ms |
| Linear / BMM | 1 | 3 | 15 ms |
| KAN / SwiGLU | 1 | 3 | 10 ms |

缩减模式会另行减少用例或窗口；具体以保存的环境与样本文件为准。

## 输出与保存

默认目录是仓库根目录 `output/`；`--output` 可改为独立目录。相同目录下重跑可能覆盖或合并已有 CSV，跨机器或跨版本比较时应使用不同目录，并同时保存对应环境与校准文件。

| 文件 | 内容 |
| --- | --- |
| `profile.json`、`calibration.csv`、`device.csv` | profile 与独立设备/策略测量。 |
| `results.csv`、`samples.csv`、`checks.csv` | GEMM 汇总、逐窗口数据与正确性/后端检查。 |
| `environment.json`、`results.md` | GEMM 环境、完成状态及配对加速比汇总。 |
| `gemm-t{1,8}-{gflops,latency}.{pdf,svg,png}` | 默认 GEMM 图。 |
| `attention_results.csv`、`attention_phases.csv`、`attention_checks.csv` | Attention 完整与阶段数据。 |
| `{kan,swiglu,linear,bmm}.csv`、`*_samples.csv`、`*_environment.json` | 对应应用实验的数据与运行信息。 |
| `results_v2.csv`、`results_v2_samples.csv`、`results_v2_checks.csv`、`environment_v2.json` | 密集 GEMM 数据。 |
| `figures/` | 调度、跨算子与 suite 汇总图；具体导出格式依绘图入口而定。 |

数据、编译依赖与 JIT 缓存均被 `.gitignore` 排除。图由脚本从 CSV 生成，无需把 PNG/PDF 复制进源码仓库。

## 诊断脚本与已知限制

- [`rank_compare.py`](../experiments/rank_compare.py)：比较第 5、10、20 名及最末候选。现有 `--processes` 参数实际上在**同一进程**内重复测量，不能作为独立进程证据；使用前应核查数值检查列，脚本不会因 `passed=False` 自动中止。
- [`plot_diagnostics.py`](../experiments/plot_diagnostics.py)：从 `--root` 指定的 CSV 绘制步长、扩展性与候选排序图；保留各 `--*-stem` 参数。
- [`plot_cliff2048.py`](../experiments/plot_cliff2048.py)：比较已有 before/after CSV，输入版本与环境一致性需由实验者核查。
- [`stride_conflict_probe.cpp`](../experiments/stride_conflict_probe.cpp) 与 [`plot_stride_probe.py`](../experiments/plot_stride_probe.py)：标量加载诊断，不直接测量 SME；绘图通过 `--root` 指定含 `low_level_stride.csv` 的目录。
- 密集尺寸列表含 8192，而源码备注固定 SME 对该 N 有拒绝限制。默认四方法密集 suite 可能在该点终止；远程复现实验应按已有协议限制尺寸，或明确用 `--methods generic ours pytorch` 比较三方法，不能把缺测当作完整四方法结果。
- suite 汇总器可以合并多个 CSV，诊断图并非都校验完成状态与身份。应在同一版本的独立输出目录绘图，避免混入旧结果。

## Linux Arm v9 实验

Linux 实验使用 `linux-arm-v9` 分支，固定 SAGE 提交 `fa3881e104d5bf286ead4b6fe2651c51b7de4cd1` 和 TritonCPU 提交 `8261cd0638ca4745bb14133ca586e3bce19654ef`。复现目录包含离线依赖清单、Linux 补丁、SME 探针、构建脚本、OpenBLAS 基线脚本以及正式 GEMM 汇总器。脚本默认通过集群调度器提交到计算节点，不应在登录节点上执行长时间构建或正式计时。

正式结果覆盖单 NUMA 的 1/8 线程和跨 NUMA 的 8 线程，并以 generic、固定 SME、SAGE、OpenBLAS ARMV8SVE 和 PyTorch 2.7.1 进行同机配对。代表性 1024³ 和 2049³ 形状的 8 线程候选选择消融已补做，使用 14 个候选方法和 3 个独立进程。KML_BLAS 2.2.0 已确认导出 SGEMM，但在实际 Kunpeng 节点初始化时因 CPU ID 校验退出，因此报告中标记为阻塞。完整 CSV、环境记录、校准 profile 和图表随论文实验包保存，未完成的 KML 仍不作为性能结论。

### 协议审计与重绘

Linux 归档可以在不重新计时的情况下运行协议审计。审计只读取 `environment.json`、`samples.csv`、`checks.csv` 和 `process_results.csv`，检查完成状态、主机一致性、独立进程数、窗口数、目标窗口覆盖、数值检查以及编译/首次调用证据：

```sh
python experiments/linux_arm/protocol_audit.py \
  /path/to/formal-single-merged /path/to/formal-cross-merged \
  --json /path/to/protocol-audit.json \
  --markdown /path/to/protocol-audit.md
```

已有汇总 CSV 可直接重绘全部 shape sweep 和配对 speedup。脚本把矩形 shape 映射到 $(MNK)^{1/3}$，用独立进程 min--max 作为误差带，并用对数 speedup 轴避免 generic 基线遮住较小的 CPU/BLAS 比值：

```sh
python experiments/linux_arm/plot_protocol_review.py \
  --summary /path/to/formal-single-merged-summary.csv \
            /path/to/formal-cross-merged-summary.csv \
  --paired /path/to/formal-single-merged-paired.csv \
           /path/to/formal-cross-merged-paired.csv \
  --figures /path/to/figures/protocol-review
```

这两个脚本不提交作业、不访问登录节点上的正式 benchmark，也不修改原始 CSV。图注应把 min--max 标成进程范围，不能称为置信区间。正式计时仍通过 `dsub` 在计算节点完成，登录节点只用于轻量身份、队列和探针检查。
