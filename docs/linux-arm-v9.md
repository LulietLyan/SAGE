# Linux Arm v9 实验记录

这份页面对应 `linux-arm-v9` 分支。完整原始产物、环境快照、校准 profile、CSV 和图表保存在配套论文目录的 `SAGE-paper/linux-arm-v9` 目录中。源码仓库只提交复现实验脚本和补丁，避免把机器相关的结果文件混入代码历史。

## 固定版本

- SAGE `fa3881e104d5bf286ead4b6fe2651c51b7de4cd1`
- TritonCPU `8261cd0638ca4745bb14133ca586e3bce19654ef`
- LLVM/MLIR `23.0.0git` ARM64 构建包
- Python `3.11.14`
- OpenBLAS `0.3.34`
- PyTorch `2.7.1`

依赖和输入包的 SHA-256 清单位于 [`experiments/linux_arm`](../experiments/linux_arm) 中。该目录的脚本使用普通用户目录和集群调度器，不修改系统配置。

## 硬件与原生验证

登录节点只用于核对角色和提交队列。正式作业在 Donau `q_hpcapp` 计算节点执行。最终计时节点为 Kunpeng 920 72F5，608 个逻辑 CPU，32 个 NUMA 节点中 16 个有 CPU，Kylin Linux 5.10。Linux HWCAP/HWCAP2 暴露 SVE、SVE2、SME 和 SME_F32F32，SME2 未暴露。`PR_SME_GET_VL` 和实际 streaming 状态均为 64 字节。

最小探针执行了 FP32 `fmopa`，并检查输入输出哨兵。generic JIT 在两个形状上均未生成 `fmopa`，固定 SME JIT 生成 16 或 20 条 `fmopa`，两者的 FP32 数值检查均通过。构建、探针和 JIT 诊断日志位于论文产物的 `environment/`。

## 正式结果

正式 GEMM 使用完整同步调用计时。窗口包括打包、复制、任务提交、计算和等待，构建、校准、首次 JIT 和正确性检查单独记录。每个方法和形状使用 3 个独立进程、4 个交错窗口，校准样本与评估样本分离。

| 布局和线程 | SAGE / 固定 SME | SAGE / OpenBLAS ARMV8SVE | SAGE / PyTorch | SAGE / generic |
| --- | ---: | ---: | ---: | ---: |
| 单 NUMA，1 线程 | 1.477×，32/33 胜 | 1.735×，33/33 胜 | 1.776×，33/33 胜 | 24.062×，33/33 胜 |
| 单 NUMA，8 线程 | 1.299×，25/33 胜 | 0.870×，16/33 胜 | 1.285×，26/33 胜 | 12.930×，33/33 胜 |
| 跨 NUMA，8 线程 | 1.343×，31/33 胜 | 0.731×，8/33 胜 | 1.096×，19/33 胜 | 13.008×，33/33 胜 |

单 NUMA 的 8 线程结果显示 SAGE 超过固定 SME 和 PyTorch，但低于同机 OpenBLAS。跨 NUMA 时，OpenBLAS 领先，SAGE 仍超过固定 SME、PyTorch 和 generic。正式逐窗口 CSV 在 `SAGE-paper/linux-arm-v9/data/formal-single-merged/` 和 `formal-cross-merged/`。

## 基线与阻塞项

OpenBLAS ARMV8SVE 覆盖 33 个形状、单 NUMA 的 1/8 线程和跨 NUMA 的 8 线程。OpenBLAS ARMV9SME 作为附加诊断覆盖 8 个形状。PyTorch 2.7.1 记录了实际 wheel 和线程后端。

KML_BLAS 2.2.0 已确认包含 LP64 `cblas_sgemm`、线程设置接口和 NEON/SVE pthread 库，但在两个实际 Kunpeng SME 节点初始化时均因 CPU ID 校验退出。现有数据不能支持 KML_BLAS 与 Apple Accelerate 的直接性能结论，正式报告将其标记为阻塞。

Apple M4 只恢复了论文 ZIP 中的汇总和逐尺寸摘要，逐窗口 CSV 已不在归档中。Linux 结果单独校准和报告。Linux 已补做 8 线程、1024³ 和 2049³ 的 14 方法候选选择消融，单 NUMA 和跨 NUMA 各 3 个进程、4 个窗口，结果保存在论文包的 `data/ablation-single/`、`data/ablation-cross/` 和 `figures/ablation-regret.svg`。这组实验量化选择差距，没有把单个模型项解释为因果贡献。

## 复跑入口

```sh
cd experiments/linux_arm
bash run-linux.sh experiment-v3/native-smoke.py native-checks
bash run-linux.sh experiment-v3/calibrate-bind.py single calibration-single
bash run-linux.sh experiment-v3/calibrate-bind.py cross calibration-cross
bash run-linux.sh experiment-v3/gemm.py --output formal-single --topology single \
  --methods generic sme-fixed ours openblas pytorch --threads 1 8 \
  --profile calibration-single/profile.json \
  --openblas "$PWD/openblas-ARMV8SVE/lib/libopenblas.so"
```

长时间构建和正式测量应提交到获准的计算队列。`experiments/linux_arm/README.md` 记录了依赖布局、队列示例、线程亲和性和结果汇总命令。
