# H3 VDN SageAttention AMD 加速结论汇总

更新时间：2026-09-07

本文把现有实验整理成一份面向决策的索引，回答三个问题：

1. 哪些改动已经证明能够加速；
2. 哪些改动不能加速，或收益不足以保留；
3. 哪些改动虽然加速，但因 H3 音频质量不合格而不能进入 stable/`auto`。

详细过程与原始数据见
[`h3-vdn-sageattention-optimization.md`](h3-vdn-sageattention-optimization.md)，
接口、门禁和 Definition of Done 见
[`vdn-sageattention-amd-cpp-requirements.md`](vdn-sageattention-amd-cpp-requirements.md)。

## 1. 一页结论

| 问题 | 当前结论 |
|---|---|
| AMD gfx1201 上能否显著加速 H3 VDN attention？ | **能。** E27 在 GPU6 standalone 中相对精确 wave32 快约 **26.9x**；真实 50 层累计 SDPA 快 **16.61x**。 |
| 能否加速完整 H3 推理？ | **能。** 已测 8-NFE DiT 提升约 **2.06x--2.27x**，example_0 完整进程提升 **1.545x**。 |
| 能否设为 stable 或 `auto` 默认？ | **不能。** 三个真实 prompt 的音频质量门禁为 **0/3**；`auto` 必须继续使用精确 wave32。 |
| 当前应保留什么？ | 保留 E27 作为显式 `sage-i8-bf16` research/experimental mode；保留 wave32 作为 stable/`auto`。 |
| 当前不应继续做什么？ | 不再扩大 softmax batch、同 workgroup producer/consumer、layer/NFE/modality/head fallback、单纯布局转换等已系统失败的搜索。 |
| 下一条值得投入的路线是什么？ | 优先做 **H3 专用校准/训练**，使模型适应 INT8 QK + tile-softmax/BF16 PV；若必须保持精确轨迹，则需要新的持久化/融合执行模型。 |

最重要的边界是：**“能加速”不等于“能上线”。** E27 已通过算子性能、正确性、
真实 50 层、8-NFE、视频和 mux 门禁，但未通过音频质量门禁。

## 2. 测量范围与判定口径

当前 standalone production geometry：

```text
S=5338, H=56, D=128
video_start=986, frames=17, tokens_per_frame=256
radius=1, chunk=5, anchor_both=true
```

按 VDN mask 的有效 Q-K 对计数，总有效工作量为 708.002 GOP；R9700 的 INT8/BF16
混合精度理想峰值按 254.889 TOPS 计算。所有 operator 时间均包含 Q/K 量化，不能只
报告核心 QK/PV kernel。

本文使用三种不同判定，不能互相替代：

| 判定 | 含义 |
|---|---|
| `ACCEPT` | 独立性能收益达到门槛，且正确性/资源门禁通过。 |
| `RETAIN` / `MERGED` | 单项收益不足 3%，但缩短 ISA、降低资源或解锁后续收益，最终进入 E27 组合。 |
| `REJECT` | 性能回退、收益处于噪声、资源/容量违规，或 H3 质量未过线。 |

跨 GPU 的绝对时间只用于各自卡内比较。GPU4 是正式 promotion 卡；GPU5/GPU6 主要
用于研究与复核。

## 3. 已确认能够加速并进入 E27 的改动

### 3.1 最终有效加速栈

| 实验 | 改动 | 直接证据 | 为什么有效 | 最终处理 |
|---|---|---|---|---|
| E02 | online softmax 使用 device fast-exp | profile 20.093 → **18.338 ms**，降低 8.73% | 去掉 IEEE `expf` 的范围修正和临时序列 | `ACCEPT` |
| E03 | 有限概率使用 branch-free BF16 RNE pack | profile 18.338 → **16.889 ms**；相对 B0 降低 15.95% | 去掉通用 BF16 转换中的 NaN/舍入控制流 | `ACCEPT` |
| E13 + E14 | 移除 `allowed[8]` 长生命周期，并用 `exp(-INF)=0` 消除逐概率分支 | GPU5 profile **16.208 ms**；较 B5-E03 降低 4.08% | 减少概率阶段分支和跨阶段活跃值 | `ACCEPT` 组合 |
| E15 + E16 | 精简 probability fragment 跨 lane 重排；每 row 复用 denominator reciprocal | GPU median **16.130 ms**；VGPR 216 → 211，shuffle 108 → 52 | 单项收益小，但 ISA 更短、资源更低 | `RETAIN`，作为结构基础 |
| E21 | 32-row Q super-tile，2 wave、零 barrier | GPU5 profile 16.222 → **15.214 ms**，降低 6.21% | 相邻 Q tile 在同 workgroup 自然复用 K/V cache | `ACCEPT` |
| E25 + E26 | workgroup 共享 task，最终改为 uniform task direct load | E26 GPU5 profile **14.974 ms**；VGPR 208 → 192，LDS 3328 B → 0 | 消除每 lane 的 task 复制、LDS 和入口 barrier | `ACCEPT` 组合，首次达到 M2 |
| E27 | 按 row 完成归一化并立即写回 8 个 depth fragment | GPU5 profile **14.832 ms**；GPU4 正式 profile **14.892 ms** | 缩短输出尾段活跃区和 ISA，扩大 M2 裕量 | `RETAIN`，当前最终版本 |

E13、E15、E16、E25、E27 的部分单项收益低于 3%，不能单独宣称显著加速；它们的价值
是降低资源和缩短代码，并与后续改动组成最终 E27。不要把这些结构保留项重复包装成
独立性能突破。

### 3.2 累积结果

| 版本 | 设备/角色 | Profile total | GPU median | 有效吞吐 | 混合峰值占比 | 状态 |
|---|---|---:|---:|---:|---:|---|
| B0 | GPU4 初始 Sage MVP | 20.093 ms | 20.105 ms | 35.236 TOPS | 13.824% | 起点 |
| E03 | GPU4 首轮 accepted baseline | 16.889 ms | 17.117 ms | 41.921 TOPS | 16.447% | M1 |
| E14 | GPU5 accepted candidate | 16.208 ms | 16.280 ms | 43.682 TOPS | 17.138% | accepted |
| E21 | GPU4 历史 release candidate | 15.807 ms | 15.598 ms | 44.790 TOPS | 17.573% | 已被 E27 取代 |
| E27 | GPU4 当前 release baseline | **14.892 ms** | **15.121 ms** | 约 **47.6 TOPS** | 约 **18.7%** | operator promotion PASS |

同一 GPU4 口径下，B0 到 E27 的 profile total 降低 **25.88%（1.349x）**，GPU
median 降低 **24.79%（1.330x）**。这里比较的是初始 Sage MVP 与最终 Sage E27，
不是 E27 与精确 wave32 的倍数。

## 4. 已确认的真实 H3 加速

| 范围 | 精确 wave32 | E27 | 加速 | 质量/状态 |
|---|---:|---:|---:|---|
| GPU6 standalone production SDPA | 414.786 ms | 15.419 ms | **26.9x** | operator correctness PASS |
| GPU5 真实 50 层累计 SDPA | 14.865 s | 0.895 s | **16.61x** | 单 NFE 传播 PASS |
| GPU5 真实 50 层 forward | 28.404 s | 14.622 s | **1.943x** | 单 NFE 传播 PASS |
| example_0 8-NFE DiT | 246.230 s | 119.637 s | **2.058x** | 视频 PASS，音频 FAIL |
| example_0 完整进程 | 356.76 s | 230.88 s | **1.545x** | VAE/mux PASS，音频 FAIL |
| example_1 8-NFE DiT（GPU6） | 256.676 s | 123.530 s | **2.078x** | 视频 PASS，音频 FAIL |
| example_2 8-NFE | 389.590 s | 171.934 s | **2.266x** | audio latent 快速失败 |

结论：E27 的算子加速能够稳定传递到 50 层和完整 DiT；完整进程的收益被与 SDPA mode
无关的 VAE、mux 等阶段稀释。这些数据足以证明“有真实性能收益”，但不足以解除音频
质量阻塞。

## 5. 不能加速或不值得保留的性能路线

### 5.1 编译器、launch 与量化微调

| 实验 | 路线 | 结果 | 决策依据 |
|---|---|---|---|
| E01 | `launch_bounds(32,2)` | GPU median 20.026 ms，仅约 0.4% | 资源和 ISA 不变，低于 3% 门槛 |
| E05a | AMDGPU VGPR live-range pass | 仍为 216 VGPR | 没有改变静态资源，不进入计时 |
| E05b | occupancy-biased scheduler | 16.838 ms，对照 16.824 ms | 无收益 |
| E10 | wave-level Q/K quant reduction | 16.810 ms，仅改善 0.08% | 量化 barrier 下降但总时间处于噪声 |
| E23 | 重访 wave-level quant reduction | profile 15.154 ms，仅改善 0.39% | 未跨 M2，仍低于接受线 |

结论：当前约 0.86 ms 的 Q/K quant 不是最值得继续微调的热点；只改编译参数或 reduction
形式不能产生可复现的整体收益。

### 5.2 只降 VGPR、拆 wave 或 producer/consumer

| 实验 | 路线 | 资源变化 | 性能结果 | 结论 |
|---|---|---|---:|---|
| E06 | 2-wave K split + LDS partial-softmax merge | 56 B scratch | 24.874 ms | merge/LDS/scratch 成本过高 |
| E07 | K-loop 内重载 Q fragment | VGPR 216 → 204 | 17.997 ms | 未跨 occupancy 台阶，重复读 Q 反而变慢 |
| E08 | 2-wave output split，重复 QK/softmax | VGPR 216 → 144 | 20.578 ms | 重复 score 路径抵消低 VGPR 收益 |
| E09 | 1 producer + 2 PV consumer | VGPR 133、0 scratch | 22.676 ms | 每 K tile 两次 barrier 太贵 |
| E17 | 8-tile batched producer/consumer | VGPR 136、LDS 4672 B | 未计时 | 出现 56 B scratch，静态门禁失败 |
| E28 | exact-order，3 wave 整批 QK/PV | bitwise exact | 最佳 1438.686 ms | 比 wave32 慢 3.48x |
| E29 | exact-order，1 producer + 2 consumer | bitwise exact | 最佳 938.088 ms | 比 wave32 慢 2.27x |
| E30 | exact-order，1 producer + 1 consumer | VGPR 23、0 scratch、bitwise exact | 最佳约 693 ms | 仍比 wave32 慢约 68%，比 E27 慢约 45x |

结论：**降低 VGPR 本身不是加速。** 如果代价是重复 QK/softmax、频繁 barrier、LDS
系数流量或 producer/consumer 握手，性能会明显下降。现有同 workgroup wave
specialization 路线已经充分失败。

### 5.3 mask、tile 粒度与数据布局

| 实验 | 路线 | 结果 | 决策依据 |
|---|---|---|---|
| E04 | full-tile mask fast path | VGPR 216 → 228 | 双路径提高寄存器压力；污染性能点作废 |
| E12 | 16-bit VDN 有效位图 | 16.934 ms，慢约 0.7% | 边界比较不是显著热点 |
| E18 | full-sequence 专用 kernel | dense 实例 256 VGPR、32 B scratch | 编译期 specialization 仍恶化资源 |
| E19 | K 量化直接写 head-major | 16.135 ms，对照 16.130 ms | 无可测收益 |
| E20 | V 显式 pack 为 head-major | 16.784 ms；workspace 146.031 MiB | 变慢且超过 80 MiB contract |
| E22 | 64-row/4-wave Q super-tile | profile 15.378 ms | 略慢于 32-row E21 |
| E24 | 48-row/3-wave Q super-tile | profile 17.677 ms | 3-wave 调度明显回退 |

结论：当前最佳粒度是 **32-row/2-wave/零 barrier**。显式 K/V 重排不能抵消转换和容量
成本；mask interior/boundary 分支也不是当前主要瓶颈。

### 5.4 softmax 批处理与精确轨迹

| 实验 | 路线 | 结果 | 决策依据 |
|---|---|---|---|
| E11 | 每 2 个 K tile 合并一次 softmax 更新 | profile 17.452 ms，VGPR 230 | 标量节省抵不过寄存器和展开成本 |
| E31 | 逐 key softmax state + tile WMMA PV | 15.548 → 25.546 ms | 变慢 64.3%，RMSE 仅 0.361013% → 0.361007% |
| E28--E30 | 完整 exact-order producer/consumer | bitwise 正确但最快约 693 ms | 精确轨迹与现有 wave specialization 不能同时高吞吐 |

结论：继续调整 softmax batch 大小没有证据价值。只恢复逐 key softmax state 几乎不
改善误差；完整恢复 QK/PV 数值轨迹则会丢掉绝大部分甚至全部性能。

## 6. “会加速但不能上线”的路线

以下候选通常比全 wave32 快，也可能改善部分误差，但没有同时通过冻结的音频门禁，
所以仍归类为 production `REJECT`。

| 候选 | 性能收益 | 最好或关键质量结果 | 最终原因 |
|---|---:|---|---|
| E27 | DiT 约 2.06x--2.27x | example_0 audio 0.981459 / 19.232%；example_1 0.983433 / 18.1844% | 三 prompt 音频 0/3，不进入 stable |
| H01 | example_0 完整 wall 1.391x | audio 0.985745 / 16.874% | 有改善但仍未达到 correlation 0.99、RMSE 10% |
| H03 | 8-NFE 1.422x | audio latent 11.3370% | NFE 首尾精确调度没有修复累计误差 |
| A05 | DiT 1.761x | audio 0.987678 / 15.683% | 比 E27 更好但仍未过冻结媒体线 |
| A08 | 8-NFE 1.746x | audio latent 3.6599% | 单 NFE 改善没有跨 scheduler 保持，差于 A05 |

还测试并拒绝了：

- H02 anchor/global query 精确覆盖；
- A01/A07/A10/A11/A12 的 layer 或 NFE 精确调度；
- A02 row-scale、A03 F16-PV、A04 exact-QK、A06 modality fallback；
- A09 BF16 rocWMMA QK + F16 PV、A13 exact reduction tree；
- E32 分 head 精确覆盖。

E32 尤其说明不能继续搜索“少数坏 head”：七组 8-head 精确覆盖都把 audio RMSE 从纯
E27 的约 1.54% 放大到 4.88%--5.53%。E27 的 head-wise 误差在 output projection 中
存在抵消，局部混用精确/近似 head 会破坏这种抵消。

## 7. 不属于加速、但必须保留的改动

| 改动 | 作用 | 是否宣称加速 |
|---|---|---|
| rocSOLVER 前固定 stream boundary | 修复 example_1 Cholesky producer/solver 时序问题 | 否；只宣称稳定性修复，未观察到可测负收益 |
| 同 command group 的 64 轮 solve 压力测试 | 防止分两次 submit 隐藏时序问题 | 否 |
| T01 H3 定向 geometry、极端分数和 guard canary | 验证非对齐、短 interval、OOB、determinism | 否 |
| `H3_VDN_SOLVE_DIAGNOSTICS` | 失败矩阵 readback 与 CPU double Cholesky 诊断 | 否，默认关闭 |
| `VDN_SMOKE_SAGE_LAYER_ERRORS=1` | 观察 text/audio/video 逐层误差放大 | 否，诊断专用 |
| `auto` 保持 wave32 | 保证当前 production 音频质量 | 否，这是发布安全边界 |

## 8. 当前发布矩阵

| 模式 | 性能 | 正确性/稳定性 | 视频 | 音频 | 当前定位 |
|---|---|---|---|---|---|
| `wave32` | 慢 | PASS | reference | reference | stable |
| `auto` | 当前选择 wave32 | PASS | PASS | PASS | production 默认 |
| `sage-i8-bf16` E27 | 显著加速 | operator、50 层、8-NFE、Cholesky PASS | PASS | **FAIL** | explicit experimental/research |
| 已清理的 hybrid/实验 mode | 部分有速度或诊断价值 | 不完整或已失败 | 不统一 | FAIL | 不保留公开入口 |

## 9. 后续投入建议

### 9.1 应继续

1. **H3 专用校准/训练**：让模型直接适应 INT8 QK、tile online-softmax 和 BF16 WMMA
   PV 的联合误差；训练目标必须单独关注 audio residual/velocity，而不能只优化合并
   latent loss。
2. **训练感知的量化参数**：可以研究 layer/head/modality 级 scale 或 clipping，但应在
   校准目标中联合优化；不要再用运行时精确 fallback 规则代替校准。
3. **端到端音频优先门禁**：新候选按单 NFE audio → 8-NFE audio latent → 完整媒体
   correlation/RMSE/SI-SDR 分级快速失败，避免先消耗完整 VAE/mux 时间。
4. 如果目标改为 bitwise exact，则探索不同的 persistent/fused 执行模型；不要继续在
   当前同 workgroup producer/consumer 结构上增加 batch 或 wave。

### 9.2 暂停或停止

- 继续扫描 softmax batch=2/4/8/16/32；
- 继续拆 producer/consumer wave 或只追求更低 VGPR；
- 继续尝试 48/64-row super-tile；
- 继续做 K/V head-major 全量 pack；
- 继续组合 layer、NFE、modality、anchor 或 head 精确 fallback；
- 在没有训练/校准闭环前继续细调 F16/BF16 PV 或 Q/K scale；
- 用 seed 数量替代不同真实 prompt 的媒体门禁。

## 10. 新候选的最短验收路径

```text
静态资源/ISA
  -> operator correctness + OOB/canary + determinism
  -> production 10-iteration 卡内性能
  -> example_0 单 NFE，单独检查 audio
  -> 8-NFE audio latent
  -> 三个真实 prompt 的完整媒体门禁
  -> GPU4 干净窗口 promotion
  -> 才能讨论 stable/auto
```

任何一步失败都应记录后停止扩大实现；不能以 operator 加速、合并 latent cosine 或视频
PASS 覆盖音频 FAIL。

## 11. 当前最终决定

- **能够加速且应保留的实现**：E02、E03、E13/E14、E15/E16、E21、E25/E26、E27
  组成的最终 E27 栈。
- **能够端到端加速但不能上线的实现**：E27 以及部分 H/A hybrid；原因是音频质量，
  不是性能。
- **不能加速或不值得继续的方向**：编译器 flag、单纯降 VGPR、现有
  producer/consumer、softmax batch、mask fast path、显式 K/V pack、48/64-row tile、
  runtime 精确 fallback。
- **当前 production 选择**：`auto = wave32`；`sage-i8-bf16 = explicit experimental`。
- **下一阶段入口**：H3 专用校准/训练；没有校准数据时维持当前状态，不再扩大
  inference-only 混合规则搜索。
