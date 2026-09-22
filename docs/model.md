# 模型与实现

[返回 README](../README.md) · [实验协议](experiments.md)

本文对应当前源码，整理了原始建模资料中的语义、成本和摊销分析。公式用于描述实现与假设；实测加速需要独立实验支持。

## 1. 编译边界与决策变量

输入为连续、行主序 FP32 矩阵，目标为覆盖写入 $C=AB$：

$$
C_{ij}=\sum_{k=0}^{K-1} A_{ik}B_{kj}.
$$

[`CompilerGemmPlan.cpp`](../adapters/triton_cpu/CompilerGemmPlan.cpp) 从 TTIR 检查静态维度、地址表达式、掩码、零初始累加器和完整 K 归约，再改写分块并复查。SME 路径要求 $M,N,K\ge32$。launcher 检查原始 grid、线程预算、地址溢出和 C 与输入的重叠；有效分配大小及调用期间输入不变仍由调用方保证。A、B 可以共享只读存储。

调度变量为

$$
\theta=(e,B_M,B_N,B_K,K_c,a,\ell,p),\qquad
\theta^*=\arg\min_{\theta\in\mathcal F}\widehat T(\theta).
$$

| 变量 | 实现含义 |
| --- | --- |
| $e$ | 普通 CPU 或 SME。 |
| $B_M,B_N,B_K$ | 输出宏块与 Triton K 归约块。 |
| $K_c$ | 完整 K 面板容量或顺序 K 分段容量。 |
| $a$ | 标量或向量 A 打包。 |
| $\ell$ | 原始 B、局部补齐 B 面板或全局补齐 B。 |
| $p$ | 不超过调用方预算的已校准线程数。 |

可行集 $\mathcal F$ 受语义、整数溢出、scratch 上限、编译器和设备标识以及已测量策略覆盖约束。当前校准线程数为 1、2、4、8；线程数不是 SME 物理单元数量。C++ 按 profile 中的策略与 tile 家族生成合法候选，并使用确定性的平局规则排序。见 [`CompilerGemmPolicy.cpp`](../lib/Core/CompilerGemmPolicy.cpp) 和 [`CompilerGemmProfile.cpp`](../adapters/triton_cpu/CompilerGemmProfile.cpp)。

## 2. 输出所有权、分段与布局

令 $c_M=\lceil M/B_M\rceil$、$c_N=\lceil N/B_N\rceil$，任务 $(u,v)$ 独占

$$
\Omega_{uv}=[uB_M,\min((u+1)B_M,M))\times
             [vB_N,\min((v+1)B_N,N)).
$$

任意输出元素的唯一拥有者是 $(\lfloor i/B_M\rfloor,\lfloor j/B_N\rfloor)$。所有 K 分段均属于该任务；这是顺序分段，不是并行 split-K。第 $s$ 段结束时的实数算术不变量为

$$
C^{(s)}_{ij}=\sum_{k=0}^{\min(sK_c,K)-1}A_{ik}B_{kj}.
$$

首段从零开始，后续段重载 FP32 部分和后继续累加。尾部谓词限制有效读写。A 打包及两种 B 布局在有效索引上满足

$$
\begin{aligned}
P_A[tB_M+r]&=A_{i_0+r,k_0+t},\\
P_B[t(B_N+\lambda)+c]&=B_{k_0+t,j_0+c},\\
G_B[k(N+\lambda)+j]&=B_{kj}.
\end{aligned}
$$

当前 $\lambda=512/32=16$。局部与全局 B 补齐互斥，逻辑 N 及 C 的行跨度保持不变；补齐列不参与计算。全局 workspace 复用分配容量，**每次调用仍复制当前 B 的值**。这些值映射与输出分区说明实数算术上的等价关系，不意味着不同后端的 FP32 结果逐位相等。对应实现位于 [TritonCPU 补丁](../patches/triton-cpu-sme.patch)。

对有限输入、顺序 round-to-nearest FP32 FMA，在无溢出且中间结果满足标准相对舍入模型的条件下，令 $u=2^{-24}$、$\gamma_K=Ku/(1-Ku)$ 且 $Ku<1$，通常的逐步舍入展开给出

$$
|\widehat C_{ij}-(AB)_{ij}|\le\gamma_K\sum_k|A_{ik}B_{kj}|.
$$

这一条件性分析不覆盖任意下溢、flush-to-zero 或 FPCR 状态。[数值检查](../experiments/common.py) 使用 PyTorch FP32 对照及最多 16 个确定性行的 FP64 对照；后者采用更保守的 $\gamma_{2K}$ 尺度，未构造定向舍入包络，因此属于有限样本诊断。

## 3. 完整调用成本

以下变量与 [`estimate()`](../lib/Core/CompilerGemmPolicy.cpp) 对应。时间为 ns，bytes/ns 与 FLOPs/ns 分别等于十进制 GB/s 与 GFLOP/s。

$$
J=c_Mc_N,\qquad L=\lceil J/p\rceil,\qquad
\eta=\frac{J}{Lp},\qquad z=\lceil K/K_c\rceil.
$$

$L$ 与 $\eta$ 是任务波次数与末波占用率的近似，不是硬件利用率测量。完整 K 候选取 $K_c=B_K\lceil K/B_K\rceil$，分段候选取选定的 `stripK`。实际降级对小 K 使用至少 $2B_K$ 的保守分配界。

面板占用与 scratch 约束为

$$
S=4B_MK_c+\mathbf1_{\ell=\mathrm{local}}4(B_N+\lambda)K_c,
\qquad S_{\mathrm{scratch}}=S+4B_MB_N.
$$

当前 A 面板及总 scratch 分别限制为 4 MiB；全局 B 分配另限制为 $4K(N+\lambda)\le256$ MiB。这些是实现资源限制，不是推断的缓存大小。32×32 原生微内核的工作量与操作数流量代理为

$$
\begin{aligned}
\overline M&=32\lceil M/32\rceil,\quad\overline N=32\lceil N/32\rceil,\\
F&=2\overline M\overline N K,\\
R&=4\overline M K\lceil N/32\rceil
  +4NK\lceil M/32\rceil+4MN(2z-1).
\end{aligned}
$$

$R$ 包含微内核边界的重复加载需求，并非 DRAM 流量；宏块由多个微块依次执行，不会整体驻留 ZA。

### 步长惩罚

源码的 `conflictScale(d)` 定义 $\phi(d)$：初始为 1；对匹配的 `rowStrideConflicts`，取与 `max(0.05, severity)` 的较小值；若 $d\ge2048$ 且为 2048 的整数倍，再取与 **0.40** 的较小值。后一项是固定启发式，即使该宽度没有校准记录也会生效，不能解释为已测得的缓存结构或纯数据驱动结论。

令 B 的实际行跨度 $d_B$ 在原始、局部、全局布局下分别为 $N$、$B_N+\lambda$、$N+\lambda$，$\phi_B=\phi(d_B)$。源码同时惩罚计算与操作数分支，并修正 A/B 准备成本：

$$
\begin{aligned}
\widehat T_{\mathrm{SME}}={}&
\max\left\{\frac{F}{\eta P_p\phi_B},
             \frac{R}{\eta\beta_{p,f}(S)\phi_B}\right\}\\
&+\frac{8MKc_N}{\eta a_{p,a}\phi(K)}
 +\mathbf1_{\ell=\mathrm{local}}\frac{8c_MKN}{\eta b_p\phi(N)}\\
&+\mathbf1_{\ell=\mathrm{global}}\frac{8KN}{g_p\phi(N)}
 +h_p+J\tau_p+L\rho_p+Lz\sigma_p.
\end{aligned}
$$

| 系数 | 来源与含义 |
| --- | --- |
| $P_p$、$\beta_{p,f}(S)$ | 线程组 $p$ 的算术响应与精确策略家族 $f$ 的面板响应。 |
| $a_{p,a}$、$b_p$ | A 打包与局部 B 复制速率。 |
| $g_p$ | 完整全局 B 复制区域的速率，已含该区域的提交与等待。 |
| $h_p$、$\tau_p$ | 空线程组开销与每个任务的边际提交开销。 |
| $\rho_p$、$\sigma_p$ | 有效波次通信惩罚与分段开销系数。 |

速率均为整个线程组的聚合速率，不再乘 $p$。全局复制每次调用只有一份，没有 $c_M$ 或 $1/\eta$ 因子。$\rho_p$ 来自独立 barrier 测量，但 launcher 不保证每个模型波次都有对应 barrier；当前拟合令 $\sigma_p=0$，未分离的分段开销留在残差中。

CPU 候选使用 $F=2MNK$、$R=4(MK+KN+MN)$，不含 SME 打包项，采用相同 max 加调度项结构和 $\phi(N)$ 修正。其操作数系数来自合并的原始 B SME 响应，是 CPU 内存响应的近似，不能视为独立测得的 CPU 带宽。报告吞吐量始终使用逻辑工作量 $2MNK/T_{\mathrm{measured}}$。

## 4. 校准与适用域

[`calibration.py`](../python/sme_plan/calibration.py) 当前使用尺寸 704、1664、3072、3456、4096，线程组 1、2、4、8，每组 3 个独立进程。精确策略家族为

$$
f=(B_M,B_N,B_K,a,\ell,\mathbf1_{\mathrm{stripK}>0}).
$$

从完整调用中扣除已知准备与调度项 $T_{\mathrm{known}}$，得到 $r=T_{\mathrm{call}}-T_{\mathrm{known}}$。仅 $r>\max(1\mathrm{ns},0.01T_{\mathrm{call}})$ 的样本参与拟合：

$$
\beta_{\mathrm{obs}}=R/(\eta r),\qquad P_{\mathrm{obs}}=F/(\eta r).
$$

两类响应来自同一残差，无法分别识别物理带宽与纯算术峰值。Python 的准备项扣除目前**没有**复现 C++ 的 $\phi$ 步长修正；拟合残差与生产估计并非严格互逆，需用独立测量检验排序效果。

面板响应在对数容量坐标上线性插值：

$$
w=\frac{\log(S/S_0)}{\log(S_1/S_0)},\qquad
\beta(S)=(1-w)\beta(S_0)+w\beta(S_1).
$$

这是对数横坐标插值，不是 log-log 插值。相同面板字节数不允许跨 tile、打包方式、B 布局或完整/分段 K 家族借用证据。每个家族整体至少需要 3 个有效独立进程，并非每个采样点都要求 3 个。分段候选必须留在已观察容量区间内；完整 K 候选允许端点钳制，属于需要验证的外推假设。

当前 3072 节点、2048 步长启发式、固定工作量下测得的复制速率以及 CPU 响应近似都限制了模型的可迁移性。校准与正式测量可以共享尺寸，例如 4096；样本分开并不自动证明对未见尺寸的泛化。

## 5. 保留的条件性分析

### CPU/SME 合作

若未来按行把比例 $\alpha$ 的工作分给 SME，且两条路径完全重叠、速率 $r_c,r_s$ 恒定、准备开销不依赖内部切分，则

$$
T(\alpha)=\max\{(1-\alpha)F/r_c,\alpha F/r_s\},\quad
\alpha^*=\frac{r_s}{r_c+r_s},\quad T^*=\frac{F}{r_c+r_s}.
$$

有限合法切分点上比较交点相邻点及端点即可求该简化模型的最小值。争用、不完全重叠和额外准备会破坏假设。**当前实现仅选择纯 CPU 或纯 SME，不执行此混合调度。**

### 设置成本摊销

设调用前设置成本为 $H_{\mathrm{pre}}$，首次调用为 $T_{\mathrm{first}}$，稳态延迟为 $T$。对整数 $Q\ge1$：

$$
C(Q)=H_{\mathrm{pre}}+T_{\mathrm{first}}+(Q-1)T=H+QT,
\quad H=H_{\mathrm{pre}}+T_{\mathrm{first}}-T.
$$

比较两个方案，$\Delta=T_0-T_1$、$H_\Delta=H_1-H_0$，净收益为 $G(Q)=Q\Delta-H_\Delta$。当 $H_\Delta\ge0$ 且 $\Delta>0$，严格获益要求 $Q>H_\Delta/\Delta$。跨形状复用校准时只计一次共享校准成本，仍需保留各 kernel 的编译与首次调用成本。

当前实验没有测量完整且独立的冷启动成本，也没有提供数值化的盈亏平衡点。实测 min–max 或 MAD 不是置信界；不能据此声称校准必然回本。
