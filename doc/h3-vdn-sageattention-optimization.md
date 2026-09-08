# H3 VDN SageAttention AMD 性能优化日志

本文档记录 `sage-i8-bf16` 在 AMD Radeon AI PRO R9700（`gfx1201`、
wave32）上的性能优化过程。需求、接口和发布门禁仍以
[`vdn-sageattention-amd-cpp-requirements.md`](vdn-sageattention-amd-cpp-requirements.md)
为准；只有已经复现并接受的结论才回填需求文档。

若只需查看“哪些路线有效、哪些路线应停止”的决策结论，见
[`h3-vdn-sageattention-acceleration-summary.md`](h3-vdn-sageattention-acceleration-summary.md)。

## 1. 优化目标与计算口径

当前 production geometry：

```text
S=5338, H=56, D=128
video_start=986, frames=17, tokens_per_frame=256
radius=1, chunk=5, anchor_both=true
```

按 VDN mask 的实际有效 Q-K 对计数，不使用 dense `S*S` 虚高吞吐：

| 项目 | 数值 |
|---|---:|
| 有效 Q-K 对 | 24,693,156 |
| dense Q-K 对 | 28,494,244 |
| mask 有效比例 | 86.6602% |
| INT8 QK 工作量 | 354.001 GOP |
| BF16 PV 工作量 | 354.001 GFLOP |
| 总有效工作量 | 708.002 GOP |

R9700 的[官方规格](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)
给出的 dense matrix 峰值为 INT8 383 TOPS、FP16 matrix
191 TFLOPS。暂按 gfx12 BF16 WMMA 与 FP16 matrix 同吞吐计，等量 QK/PV
的混合精度理想上限为：

```text
P_mixed = 2 / (1 / 383 + 1 / 191) = 254.889 TOPS
useful_efficiency = 708.002 GOP / elapsed / 254.889 TOPS
```

这个指标是包含 mask、online softmax、数据搬运和 tile padding 的算子级
有效效率，不等同于 profiler 的 WMMA pipe busy 或 CU occupancy。

阶段目标：

| 阶段 | profile total | 有效峰值占比 | 说明 |
|---|---:|---:|---|
| 基线 | 约 20.1 ms | 约 13.8% | 已复现 |
| M1 | <= 18.0 ms | >= 15.4% | 已由 E03 达成 |
| M2 | <= 15.0 ms | >= 18.5% | 进入结构性优化 |
| M3 | <= 12.0 ms | >= 23.1% | 激进目标 |

## 2. 测量纪律

每个候选必须遵守以下规则：

1. 每个测试会话固定一张已记录型号、PCI 地址和 Unique ID 的物理 GPU，并确认
   没有其他进程在该卡执行计算；发现量化 kernel 同步异常放大时整轮作废。
2. production benchmark 至少执行 3 轮，每轮 3 次 warmup、5 次计时；主指标
   取三轮 `gpu_ms median` 的中位数，同时保留 min/max。
3. 记录 Q quant、K quant、attention 和 profile total；不得只报告最快单点。
4. 输出 hash 必须保持确定，GPU correctness、CPU contract 和 ISA 检查必须通过。
5. 记录 VGPR、SGPR、group/private segment 和 scratch；出现 spill 的候选默认回退，
   除非端到端数据证明它仍有稳定净收益。
6. 一次实验只改变一个主要变量。失败结果同样记录，避免重复探索。

标准验证命令：

```sh
make -j
make contract-test
H3_PHYSICAL_GPU=4 make gpu-test
make isa
H3_PHYSICAL_GPU=4 make bench
```

切换显卡时必须显式替换 `H3_PHYSICAL_GPU`，不能把不同卡的绝对时间混成同一
基线；先在新卡上复现当前 accepted 版本，再比较该卡内的候选相对收益。

## 3. 基线 B0

2026-09-04 在 GPU4 空闲后连续复测三轮：

| Run | Q quant | K quant | Attention | Profile total | GPU median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0-1 | 0.472 ms | 0.398 ms | 19.242 ms | 20.112 ms | 20.105 ms | 19.881 ms | 21.118 ms |
| B0-2 | 0.470 ms | 0.396 ms | 19.187 ms | 20.052 ms | 19.979 ms | 19.794 ms | 21.077 ms |
| B0-3 | 0.473 ms | 0.398 ms | 19.223 ms | 20.093 ms | 20.160 ms | 19.862 ms | 21.129 ms |

三轮 GPU median 的中位数为 **20.105 ms**。按 profile 分项中位附近的
`20.093 ms` 计算，有效吞吐为 **35.236 TOPS**，混合峰值占比为
**13.824%**；attention-only 约 **14.46%**。

当前 attention kernel 资源与 ISA：

| 项目 | B0 |
|---|---:|
| VGPR | 216 |
| SGPR | 59 |
| Group segment | 1664 bytes |
| Private segment | 0 bytes |
| Scratch spill | 无 |
| I8 QK WMMA | 8 条静态展开指令 |
| BF16 PV WMMA | 8 条静态展开指令 |

曾在 GPU4 被另一条完整 H3 任务占用时测得 62.984 ms median，且 Q/K
量化分别异常升至 57.062/29.225 ms。该轮标记为 `CONTAMINATED`，不进入
任何性能趋势或峰值占比计算。

## 4. 当前瓶颈假设

按验证优先级排序：

| ID | 假设 | 证据 | 首个实验 | 风险 |
|---|---|---|---|---|
| O1 | 208 VGPR仍限制长K-loop隐藏延迟，但2-wave super-tile已有cache收益 | E21 ISA；8组FP32 PV accumulator仍全程存活 | 下一阶段只做不引入scratch的liveness缩短 | 高：E17已证明强压寄存器会spill |
| O2 | IEEE `expf` 的范围修正和临时值增加 softmax 延迟/寄存器 | ISA 中存在 `v_exp_f32`、`v_ldexp_f32` 及展开序列 | device fast-exp 已通过首轮门禁 | 中：尾部概率与跨 tile rescale 误差 |
| O2b | 通用 `hip_bfloat16` 转换为每个概率生成 NaN/舍入控制流 | E02 ISA 中每次 K tile 有 8 组展开转换序列 | finite probability branch-free RNE pack 已通过门禁 | 中：必须证明与标准 RNE 一致 |
| O3 | 每个 wave 串行处理约 190--334 个 K tile | kernel 循环结构 | 研究两阶段 partial softmax/reduction | 高：workspace、同步和额外写回 |
| O4 | 8 个 D16 PV accumulator 造成至少 64 个长期存活 VGPR | 源码和 ISA 寄存器区间 | 比较 2-wave output split 或分阶段 PV | 高：重复 QK/softmax 可能抵消 occupancy 收益 |
| O5 | K/V 为每个 Q tile 重复读取 | E21的32-row/2-wave自然cache复用使profile降6.21% | 已接受零barrier super-tile；显式pack/LDS均不采用 | 已收敛：32-row优于16/64-row |
| O6 | 独立 Q/K 量化约占 0.87 ms | profile 分项 | wave reduction/vectorized IO，之后再评估 fusion | 低，但总收益上限有限 |
| O7 | 完整 interval 内的 K tile 不需要逐元素 mask | production 每段只有首尾边界 tile | E04 uniform fast path | 已回退：控制流令 VGPR 增至 228 |

## 5. 实验记录

| ID | 日期 | 主要变量 | Correctness | VGPR / spill | GPU median | Profile total | 相对 B0 | 结论 |
|---|---|---|---|---|---:|---:|---:|---|
| B0 | 2026-09-04 | 原始 research MVP | PASS | 216 / 0 | 20.105 ms | 20.093 ms | 1.000x | 基线 |
| E01 | 2026-09-04 | `launch_bounds(32,2)` | PASS | 216 / 0 | 20.026 ms | 19.706 ms | 1.004x | 回退：资源/ISA 不变，低于 3% 门槛 |
| E02 | 2026-09-04 | online softmax 使用 device fast-exp | PASS | 216 / 0 | 18.466 ms | 18.338 ms | 1.089x | 接受：profile total -8.7% |
| E03 | 2026-09-04 | E02 + finite probability branch-free BF16 RNE pack | PASS | 216 / 0 | 17.117 ms | 16.889 ms | 1.175x | 接受：profile total -15.9%，M1 达成 |
| E04 | 2026-09-04 | E03 + full-tile mask fast path | PASS | 228 / 0 | 污染，不采信 | 污染，不采信 | n/a | 回退：VGPR +12，且测试期间 GPU4 被占用 |

### 5.1 E02 结果

E02 将 online softmax 的两个 `expf` 调用替换为 device fast-exp。编译后的
IEEE range-fix `v_ldexp_f32` 序列消失，VGPR 仍为 216 且无 scratch。三轮结果：

| Run | Q quant | K quant | Attention | Profile total | GPU median |
|---|---:|---:|---:|---:|---:|
| E02-1 | 0.465 ms | 0.399 ms | 17.396 ms | 18.260 ms | 18.530 ms |
| E02-2 | 0.466 ms | 0.393 ms | 17.480 ms | 18.338 ms | 18.466 ms |
| E02-3 | 0.466 ms | 0.391 ms | 17.670 ms | 18.527 ms | 18.412 ms |

GPU median 中位数为 **18.466 ms**，相对 B0 提升 **8.15%**；profile total
中位数为 **18.338 ms**，相对 B0 提升 **8.73%**。有效吞吐为
**38.608 TOPS**，混合峰值占比由 13.824% 提升至 **15.147%**。

CPU contract、GPU correctness、17-frame analogue、determinism、non-finite
与双 WMMA ISA 检查全部通过。production synthetic hash 从
`dc09f9a2822ee36e` 变化为 `b0e0a8e0eca80b1b`；这是近似 exp 的预期数值
变化，不用 hash 相等代替误差门禁。

### 5.2 E03 结果

直接使用 CK 中的 `v_cvt_pk_bf16_f32` 思路在 `gfx1201` 汇编器上报
`instruction not supported on this GPU`，该不可编译尝试已丢弃。当前 E03
改用对有限 `[0,1]` 概率等价的整数 RNE 打包，GPU correctness 指标与 B0/E02
保持一致，ISA 缩短且仍为 216 VGPR、0 scratch。

E03 初测期间 GPU4 出现另一条 H3 smoke 任务，Q quant 一度升至 1.449 ms，
该批总耗时已丢弃。任务结束、GPU4 空闲后重新执行的三轮结果如下：

| Run | Q quant | K quant | Attention | Profile total | GPU median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| E03-1 | 0.471 ms | 0.399 ms | 16.053 ms | 16.922 ms | 17.143 ms | 16.986 ms | 17.850 ms |
| E03-2 | 0.469 ms | 0.396 ms | 16.024 ms | 16.889 ms | 17.080 ms | 17.001 ms | 17.800 ms |
| E03-3 | 0.469 ms | 0.398 ms | 15.990 ms | 16.856 ms | 17.117 ms | 16.967 ms | 17.857 ms |

GPU median 中位数为 **17.117 ms**，相对 B0 提升 **14.86%**；profile total
中位数为 **16.889 ms**，相对 B0 提升 **15.95%**。有效吞吐为
**41.921 TOPS**，混合峰值占比提升到 **16.447%**；attention-only 中位数
为 16.024 ms，对应 **17.335%**。

最终复验再次通过 contract、GPU correctness、17-frame analogue、determinism、
non-finite 和双 WMMA ISA 门禁。production synthetic hash 为稳定的
`0f5208a843ba73aa`。E03 作为新的优化基线，下一阶段目标为 M2（15 ms）。

### 5.3 E04 回退

E04 为完整 K tile 增加 uniform fast path，试图只在 interval 首尾执行逐元素
mask。虽然 correctness 通过，但额外控制流使 VGPR 从 216 增至 228；测试期间
GPU4 又被另一条 production H3 任务占用，量化时间升至约 1.5/0.95 ms，性能
样本全部作废。由于静态资源已经朝错误方向变化，未等待新的独占窗口，直接回退
到 E03。后续若重访该方向，应在 host task 中直接编码 interior/boundary 类型，
避免 kernel 内双路径长期同时存活。

## 6. 接受与回退标准

一个候选只有同时满足以下条件才接受：

- 三轮 median 的中位数至少提升 3%，且没有一致性变差；
- contract、GPU correctness、determinism、non-finite 和 ISA 检查全部通过；
- 非对齐/suffix case 不出现 OOB，质量指标不越过 requirements 中的门禁；
- 不增加不受控的 workspace 或 H3 hot-loop host 开销；
- 最终仍需在同 Q/K/V 的 H3 50 层和完整 8-NFE E2E 中证明净收益。

低于 3% 的改动只在它能显著简化后续结构性优化时保留，否则回退。

## 7. GPU5 后续优化会话（已收口）

会话开始时间：2026-09-04。目标设备：

| 项目 | 数值 |
|---|---|
| Physical index | GPU5 |
| PCI bus | `0000:A3:00.0` |
| Unique ID | `0xbe778bc77f940084` |
| GPU / ISA | Radeon AI PRO R9700 / `gfx1201`, wave32 |
| 选择时状态 | VRAM 0%，约 15 W，GPU use 3% telemetry floor |

GPU5 只用于后续 standalone 优化实验；GPU4 的 E03 数据仍是 release baseline。
第一步是在 GPU5 上对 accepted E03 执行 correctness、ISA 和三轮 benchmark，
建立跨卡校准基线 `B5-E03`。

| ID | 状态 | 说明 |
|---|---|---|
| B5-E03 | PASS | GPU5 已复现 accepted E03，作为本会话基线 |

### 7.1 B5-E03 校准基线

| Run | Q quant | K quant | Attention | Profile total | GPU median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| B5-E03-1 | 0.469 ms | 0.395 ms | 16.061 ms | 16.924 ms | 16.824 ms | 16.709 ms | 16.892 ms |
| B5-E03-2 | 0.471 ms | 0.396 ms | 16.030 ms | 16.897 ms | 16.774 ms | 16.716 ms | 16.901 ms |
| B5-E03-3 | 0.469 ms | 0.393 ms | 15.936 ms | 16.799 ms | 16.839 ms | 16.802 ms | 17.713 ms |

GPU median 中位数为 **16.824 ms**，profile total 中位数为 **16.897 ms**。
correctness、17-frame analogue、determinism、non-finite、output hash 和双 WMMA
ISA 均与 GPU4 E03 一致。GPU4/GPU5 的 profile total 中位数只差约 0.05%，
后续候选以 `B5-E03` 作卡内相对比较，不改写 GPU4 release baseline。

### 7.2 GPU5 实验队列

| ID | 主要变量 | 状态 | 结果 |
|---|---|---|---|
| B5-E03 | accepted E03 | PASS | GPU median 16.824 ms；profile total 16.897 ms |
| E05a | AMDGPU VGPR live-range pass | REJECT | 216 VGPR / 0 spill，不改变资源元数据 |
| E05b | occupancy-biased scheduler | REJECT | GPU median 16.838 ms，低于 3% 接受线 |
| E06 | 2-wave K split + LDS partial-softmax merge | REJECT | GPU median 24.874 ms；56-byte scratch，明显回退 |
| E07 | K-loop 内重载 Q fragments | REJECT | 204 VGPR / 0 spill，但 GPU median 17.997 ms |
| E08 | 2-wave output split，重复 QK/softmax | REJECT | 144 VGPR / 0 spill，但 GPU median 20.578 ms |
| E09 | producer/consumer wave，共享 QK/softmax | REJECT | 133 VGPR / 0 spill，但 GPU median 22.676 ms |
| E10 | wave-level quant reduction | REJECT | Quant LDS 1024→32 bytes；GPU median 16.810 ms，收益不足1% |
| E11 | 每2个 K tile 合并一次 online-softmax 更新 | REJECT | 230 VGPR / 0 spill；GPU median 17.334 ms，回退约3% |
| E12 | 每 K tile 预计算16-bit VDN有效位图 | REJECT | 216 VGPR / 0 spill；GPU median 16.934 ms，无收益 |
| E13 | 移除跨softmax阶段存活的`allowed[8]` | MERGED | GPU median 16.528 ms（+1.76%）；并入E14 |
| E14 | 概率fast-exp去除逐元素有效性分支 | ACCEPT | GPU median 16.280 ms；profile total 16.208 ms，较B5-E03降4.08% |
| E15 | 精简probability fragment跨lane重排 | MERGED | 211 VGPR；并入E16结构保留版本 |
| E16 | 输出归一化按row复用倒数 | RETAIN | GPU median 16.130 ms；低资源/短ISA版本，GPU5 promotion候选 |
| E17 | 8-tile batched producer/consumer | REJECT | 136 VGPR、4672-byte LDS，但产生56-byte scratch |
| E18 | full-sequence task专用kernel | REJECT | masked为211 VGPR；dense实例256 VGPR并有32-byte scratch |
| E19 | K量化直接写head-major布局 | REJECT | correctness通过；GPU median 16.135 ms，与E16持平 |
| E20 | V显式pack为head-major布局 | REJECT | workspace 146.031 MiB；GPU median 16.784 ms，明显回退 |
| T01 | H3边界/mask/数值/OOB定向case | PASS | 新增4组geometry及output/workspace双侧canary，GPU5通过 |
| E21 | 32-row Q super-tile、2 wave零barrier | ACCEPT | GPU median 15.354 ms；profile total 15.214 ms，较E16降6.21% |
| E22 | 64-row Q super-tile、4 wave零barrier | REJECT | GPU median 15.383 ms、profile total 15.378 ms，慢于E21 |
| E23 | wave-level quant reduction（重访 E10） | REJECT | profile total 15.154 ms，仅提升0.39%，未跨 M2 |
| E24 | 48-row Q super-tile、3 wave | REJECT | profile total 17.677 ms，明显回退 |
| E25 | workgroup-shared task metadata | RETAIN | 202 VGPR/52 B LDS；profile total 15.047 ms |
| E26 | direct uniform task loads | ACCEPT | 192 VGPR/0 LDS/0 scratch；profile total 14.974 ms，M2 达成 |
| E27 | row-at-a-time output normalize | RETAIN | profile total 14.832 ms；扩大 M2 稳定裕量 |

E05 首先通过 `SAGE_AMDGPU_FLAGS` 隔离 AMDGPU 后端试验参数。该变量只作用于
gfx12 HIP kernel 和 ISA 输出；正式 accepted 构建默认仍为空，候选比较结束后
必须强制重建默认版本，避免把临时编译参数隐式留在 build artifact 中。

E05a 使用 `--amdgpu-opt-vgpr-liverange`，输出仍为 216 VGPR、59 SGPR、
0-byte private segment，未进入耗时测试。E05b 使用
`--amdgpu-schedule-metric-bias=100`，correctness 和 ISA 通过，三轮数据为：

| Run | Attention | Profile total | GPU median |
|---|---:|---:|---:|
| E05b-1 | 15.967 ms | 16.836 ms | 16.838 ms |
| E05b-2 | 16.148 ms | 17.018 ms | 16.778 ms |
| E05b-3 | 15.974 ms | 16.833 ms | 16.884 ms |

GPU median 中位数 16.838 ms，相对 B5-E03 的 16.824 ms 无提升；profile total
差异也不足 1%。E05b 回退，说明仅调整后端 scheduler 不能解决216 VGPR的结构性
liveness；后续不再在同类编译参数上消耗测试时间。

E06 使用2个 wave 分别处理一半 K tiles，再通过16.64 KiB LDS合并 partial
softmax/output。correctness 通过且误差没有恶化，但静态资源为约217 VGPR、
56-byte private scratch；一次快速性能判定得到 attention 24.039 ms、profile
total 24.880 ms、GPU median 24.874 ms，相对 B5-E03 明显回退。按快速失败规则
不再执行三轮，立即回退。结论：只切分 K 链而不同时缩小每个 wave 的8组 PV
accumulator，既不能改善 VGPR occupancy，又引入 LDS、merge 和 scratch 成本。

E07 不再让8组 Q fragments 跨整个 K-loop 常驻，而是在每个 K tile 内重载，
将 VGPR 从216降到204且保持0 scratch。correctness/hash完全一致，但一次快速
判定为 attention 17.068 ms、profile total 17.940 ms、GPU median 17.997 ms，
相对 B5-E03 慢约7%。这表明204 VGPR没有跨过足以补偿重复Q读取的occupancy
台阶；不执行三轮，立即恢复 Q fragment 常驻。

E08 让2个 wave各负责64个输出维度，每个wave只保留4组PV accumulator，
VGPR从216显著降到144且保持0 scratch；代价是两wave各自重复完整QK和softmax。
correctness/hash完全一致，但一次快速判定为attention 19.351 ms、profile total
20.225 ms、GPU median 20.578 ms，相对B5-E03慢约22%。因此立即回退。这个实验
证明 accumulator 拆分确实能跨VGPR档位，但后续方案必须让多个PV消费者共享
一次QK/softmax结果，不能重复score路径。

E09 由wave0计算一次QK/softmax，通过约576-byte显式LDS发布概率和row scale，
两个wave各维护4组PV accumulator。资源降至133 VGPR、0 scratch，correctness/hash
完全一致；但每个K tile需要两次workgroup barrier，一次快速判定为attention
21.396 ms、profile total 22.248 ms、GPU median 22.676 ms。同步成本超过VGPR收益，
立即回退。后续跨wave共享不能采用逐K-tile barrier；需要更粗粒度的阶段划分、
异步流水或改变中间表示。

E10 将Q/K量化的8轮shared reduction改为wave shuffle加两次barrier，量化kernel
LDS从1024降到32 bytes、VGPR从8增到9且无spill。三轮Q/K quant分别约
0.457/0.391、0.457/0.391、0.459/0.389 ms；GPU median为16.835/16.810/
16.796 ms，中位16.810 ms，相对B5-E03只改善0.08%，profile total落在噪声内。
低于3%门槛且没有解锁attention结构，因此回退。

E11 将相邻两个16-key tile合并为一次online-softmax状态更新，试图把`alpha`
指数和64个PV accumulator的重缩放频率减半。编译器将临时score LDS标量化，
group segment仍为1664 bytes、0 scratch，但VGPR由216增至230；correctness通过，
一次快速判定为attention 16.595 ms、profile total 17.452 ms、GPU median
17.334 ms，相对B5-E03分别回退约3.3%/3.0%。因此不执行三轮并回退。结论是扩大
softmax更新粒度带来的标量运算节省不足以抵消更高寄存器压力和双tile展开代码。

E11回退后在GPU5用ROCm 7.2.3 `rocprofv3`采集硬件计数器。kernel trace和
`GRBM_COUNT/GRBM_GUI_ACTIVE/SQ_BUSY_CYCLES/SQ_WAVES`可用，显示dispatch期间
GPU active为100%；但gfx1201上的`SQ_WAVE_CYCLES`、VALU、TA和GL2计数均返回0，
导致occupancy、pipe issue、memory busy等派生值也错误为0。本轮仅用于确认计数器
工具链限制，不把这些0值当作硬件利用率，也不据此接受或回退任何候选。

E12 每个K tile先生成一次16-bit VDN有效位图，再让8个score按位取有效性，替代
逐score执行`key < sequence && key >= begin && key < end`。资源仍为216 VGPR、
1664-byte group segment、0 scratch，correctness和production hash通过；一次快速
判定为attention 16.146 ms、profile total 17.021 ms、GPU median 16.934 ms，
相对B5-E03慢约0.7%。边界比较不是当前显著热点，候选立即回退。

E13 去掉跨output rescale阶段存活的`allowed[8]`，在概率阶段直接判断已经编码为
`-INFINITY`的无效score。资源仍为216 VGPR、1664-byte group segment、0 scratch，
correctness和production hash通过。三轮GPU median为16.581/16.528/16.482 ms，
中位16.528 ms，相对B5-E03提升1.76%；profile total为16.562/16.675/
16.505 ms，中位16.562 ms，提升1.98%。单独低于3%接受线，随后作为进一步消除
概率分支的基础并入已接受的E14组合。

E14 对有效query直接计算`exp(score - running_max)`，利用`exp(-INFINITY)=0`
自然清零mask位置；只对无效尾部query将减数置0，防止`-INF - -INF`产生NaN。
correctness、17-frame analogue、determinism、non-finite、production hash和双WMMA
ISA门禁全部通过，资源保持216 VGPR、1664-byte group segment、0 scratch。三轮结果：

| Run | Q quant | K quant | Attention | Profile total | GPU median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| E14-1 | 0.473 ms | 0.396 ms | 15.285 ms | 16.154 ms | 16.362 ms | 15.636 ms | 24.238 ms |
| E14-2 | 0.474 ms | 0.401 ms | 15.428 ms | 16.303 ms | 16.231 ms | 16.208 ms | 16.697 ms |
| E14-3 | 0.470 ms | 0.395 ms | 15.343 ms | 16.208 ms | 16.280 ms | 16.206 ms | 16.859 ms |

首轮max有单次离群点，但量化与其余median正常；三轮GPU median中位为
**16.280 ms**，相对B5-E03降低**3.23%**；profile total中位为**16.208 ms**，
降低**4.08%**，达到接受线。按profile total计算有效吞吐为**43.682 TOPS**，
混合峰值占比为**17.138%**；attention-only中位约15.343 ms，对应**18.104%**。
E13与E14作为一个组合接受，成为GPU5的新候选基线。

E15 重新推导BF16 probability的WMMA fragment映射，让每个lane只交换实际需要的
两组32-bit probability word，而不是交换全部四组。两次初始实现的fragment顺序
错误均被uniform-score correctness在benchmark前拦截，错误版本没有性能数据；
修正后全部correctness门禁通过。静态资源由E14的216 VGPR降至211 VGPR，SGPR
next-free由35降至25，仍为0 scratch。首轮attention 15.347 ms、profile total
16.226 ms、GPU median 16.169 ms；相对E14的卡内提升不足1%，随后并入E16，
按低资源、短ISA的结构优化例外保留。

E16 将输出归一化从每个64个FP32 fragment各自执行lane shuffle和除法，改为每个
row只计算一次denominator倒数，再以乘法复用。与E15组合后，静态lane shuffle
由108条降到52条，attention ISA主体明显缩短；资源保持211 VGPR、1664-byte
group segment、0 scratch。倒数乘法使production hash预期变化为稳定的
`d8fccefb0ea98938`，但correctness、17-frame analogue、determinism和non-finite
门禁均不变。三轮结果：

| Run | Q quant | K quant | Attention | Profile total | GPU median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| E16-1 | 0.474 ms | 0.401 ms | 15.222 ms | 16.097 ms | 16.138 ms | 16.125 ms | 16.595 ms |
| E16-2 | 0.471 ms | 0.402 ms | 15.356 ms | 16.229 ms | 16.129 ms | 16.094 ms | 16.499 ms |
| E16-3 | 0.469 ms | 0.396 ms | 15.357 ms | 16.222 ms | 16.130 ms | 16.063 ms | 16.519 ms |

GPU median中位为**16.130 ms**，profile total中位为**16.222 ms**。相对E14，
GPU median再降0.92%，profile total持平；相对B5-E03则分别降低**4.13%**和
**4.00%**。按profile total计算为**43.645 TOPS**、混合峰值占比**17.123%**；
attention-only中位15.356 ms，对应**18.089%**。E15/E16没有满足相对E14的3%
独立接受线，但显著缩短ISA并将VGPR从216降至211，因此按“解锁后续结构优化”
例外保留。它目前是GPU5 promotion候选；GPU4 release baseline仍是E03，待GPU4
独占复验后才升级需求文档中的正式基线。

下一项E17会重访E09，但把probability和row alpha按8个K tile批量写入LDS，
将producer/consumer同步从每tile两次barrier摊薄到每8 tile两次，同时让两个wave
各维护4组PV accumulator。当前已固定E16源码与上述GPU5数据为回退点，E17
batch=8进入实现；先检查VGPR/LDS/scratch和correctness，再决定是否进入计时。

E17 batch=8编译后为136 VGPR、4672-byte group segment，但出现56-byte private
segment/scratch。按预先约定的快速失败规则，不执行GPU correctness和性能计时，
也不继续batch=4/16扫描；源码dispatch立即恢复E16。与E06类似，spill来自同一
workgroup路径中producer Q fragment和consumer PV accumulator的重叠压力，说明
仅摊薄barrier还不足以让这个两wave结构成为可接受实现。下一步转入E18：通过独立
full-tile实现消除interior热循环中的mask/OOB分支，避免E04在同一kernel保留双路径
造成的228 VGPR膨胀。

E18 把full-sequence task和masked task拆成两个compile-time kernel实例，以避免
同一kernel内双路径同时存活。masked实例保持211 VGPR、0 scratch；但dense实例
恶化到256 VGPR并产生32-byte private scratch。该候选没有通过静态门禁，未运行
GPU测试并立即恢复E16 dispatch。控制流specialization暂不再继续，E19转而让K
量化直接写head-major布局，并由QK WMMA按该布局读取；这样没有额外转换kernel或
workspace，只改变K量化输出的地址顺序，用于验证同一head连续K tile的局部性。

E19 保持workspace大小和quant kernel数量不变，让K量化直接写head-major布局，
QK WMMA按相同布局读取。correctness、hash和ISA门禁通过，attention仍为211 VGPR、
0 scratch；首轮K quant 0.396 ms、attention 15.275 ms、profile total 16.137 ms、
GPU median 16.135 ms，与E16的16.130 ms完全持平。该布局没有可测收益，立即回退。
E20继续评估V head-major packing；由于V没有已有量化过程可顺带转换，必须新增pack
kernel和BF16 workspace，二者都计入端到端operator时间与容量结论。

E20 新增NHD→HND BF16 V pack kernel，并把packed V放入workspace后供PV WMMA读取。
GPU correctness、17-frame analogue、determinism和hash通过，但workspace从73.051
MiB增至146.031 MiB，直接违反80 MiB contract上限；包含pack的attention阶段为
15.869 ms、profile total 16.730 ms、GPU median 16.784 ms，相对E16也明显回退。
因此无需三轮即完整回退。K与V的head-major实验共同说明，在当前按head调度和缓存
行为下，布局转换不能抵消成本；后续访存优化必须来自跨Q tile实际复用，而不是
单纯重排同样的数据。

T01 在原有uniform-score、masked K/V leak、17-frame非对齐和determinism基础上新增：

| Case | 主要覆盖 | GPU5结果 |
|---|---|---|
| single-dense-tail | `S=13`、单interval、`q_count < 16` | rel-RMSE 0.003247，cosine 0.999994728 |
| one-key-intervals | `chunk=0`、radius=0、1-key极短interval | rel-RMSE 0，cosine 1.0 |
| anchor-off-multi-interval | `anchor_both=false`、多interval、非16对齐 | rel-RMSE 0.004078，cosine 0.999991708 |
| extreme-scores | 大幅Q/K、softmax underflow压力 | rel-RMSE 0.001593，cosine 0.999998884 |
| guard canaries | output/workspace前后各4 KiB越界写检测 | 全部保持`0xA5`，PASS |

所有case都无non-finite且重复执行bitwise deterministic；CPU contract继续覆盖不同
radius/chunk、单帧、极端`UINT32_MAX`饱和和workspace溢出。E17代码保留在显式
`H3_VDN_SAGE_ENABLE_E17_EXPERIMENT`宏后供研究复现，默认E16构建不编译该拒绝路径。

E21先做Q super-tile的最低开销版本：host按同一mask class构造最多32行的task，
64-thread workgroup中的两个wave各处理16行，彼此没有LDS通信或barrier。总wave数和
QK/PV工作量不变，唯一变量是相邻Q tile被放进同一workgroup同时访问相同K/V，
用于判断gfx1201 cache是否能自然复用；若无收益，不继续更重的显式LDS K/V缓存。

E21静态资源为208 VGPR、3328-byte group segment、0 scratch；production task数
从334减为167，总wave数不变。扩展后的T01和全部既有门禁通过，production hash
保持`d8fccefb0ea98938`。三轮结果：

| Run | Q quant | K quant | Attention | Profile total | GPU median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| E21-1 | 0.467 ms | 0.393 ms | 14.339 ms | 15.199 ms | 15.354 ms | 15.221 ms | 15.423 ms |
| E21-2 | 0.470 ms | 0.396 ms | 14.348 ms | 15.214 ms | 15.359 ms | 15.343 ms | 15.442 ms |
| E21-3 | 0.468 ms | 0.396 ms | 14.363 ms | 15.227 ms | 15.333 ms | 15.314 ms | 15.411 ms |

GPU median中位为**15.354 ms**，profile total中位为**15.214 ms**，相对E16
分别降低**4.81%**和**6.21%**，超过3%接受线；相对B5-E03分别降低8.74%和
9.96%。按profile total计算有效吞吐为**46.536 TOPS**、混合峰值占比
**18.257%**；attention-only中位14.348 ms，对应**19.359%**。这证明收益来自
同workgroup双wave的自然cache locality，无需显式LDS K/V缓存和barrier。E21接受，
并成为新的GPU5 promotion候选；E22继续测试64-row/4-wave是否能跨过15 ms M2。

E22 将同mask Q super-tile扩到64行/4 wave，production task数降到84；静态资源
206 VGPR、6656-byte group segment、0 scratch，全部T01/ISA门禁通过。首轮
attention 14.510 ms、profile total 15.378 ms、GPU median 15.383 ms，均略慢于
E21，未跨过15 ms M2。按快速判定回退，不执行三轮。16-row E16、32-row E21、
64-row E22的对照表明当前最优粒度是32-row/2-wave。

### 7.3 本轮收口状态

最终源码恢复并保留E21：32-row task、64-thread/2-wave workgroup、208 VGPR、
3328-byte group segment、0 scratch，workspace 73.043 MiB。收口复验再次通过
contract、完整T01、双WMMA ISA和determinism；production复验为attention
14.327 ms、profile total 15.183 ms、GPU median 15.343 ms，与E21三轮中位结论
一致。实验队列已无`RUNNING`项。

当前GPU5正式比较仍采用E21三轮中位：GPU median 15.354 ms、profile total
15.214 ms、46.536 TOPS、混合峰值占比18.257%。距离M2的15.0 ms还差0.214 ms；
下一阶段优先考虑不改变workspace的量化/attention调度重叠或更细的寄存器liveness，
而不是重新引入E17 barrier、E18控制流特化或E20全量V packing。GPU4 release
baseline仍为E03，待GPU4独占窗口做同卡promotion复验后再升级需求文档。

## 8. GPU4 promotion 与 M2 冲刺（本轮收口）

2026-09-04 开启后续会话。启动前通过 `rocm-smi` 检查物理 GPU4/GPU5：两张卡
均为 0% GPU use、0% VRAM allocated，GPU4 上次被完整 H3 任务污染的 63 ms 数据
继续作废。本会话先在 GPU4 对 E21 执行 correctness、ISA/resource 和三轮 benchmark
promotion 门禁；通过后再更新 release baseline。随后回到 GPU5，以 E21 为回退点
继续冲刺 M2（profile total <= 15.000 ms）。

| 项目 | 状态 | 当前结果 |
|---|---|---|
| P01 GPU4 独占检查 | PASS | GPU4/GPU5 均为 0% use、0% VRAM |
| P02 E21 GPU4 correctness/ISA | PASS | 完整 T01、guard、determinism、INT8/BF16 WMMA 均通过 |
| P03 E21 GPU4 三轮 benchmark | PASS | GPU median 15.598 ms；profile total 15.807 ms |
| P04 release baseline 升级 | PASS | requirements 已升级为 GPU4 E21 release candidate |
| E23 wave-level quant reduction | REJECT | profile total 15.154 ms，仅提升 0.39%，未跨过 M2 |
| E24 48-row Q super-tile | REJECT | profile total 17.677 ms，3-wave 调度明显回退 |
| E25 workgroup-shared task | RETAIN | 202 VGPR/52 B LDS；profile total 15.047 ms |
| E26 direct uniform task loads | ACCEPT | GPU5 profile total 14.974 ms，三轮均跨过 M2 |
| P05 E26 GPU4 promotion | DEFER | 多轮出现 20--26 ms 长尾，未取得连续三轮干净数据 |
| E27 row-at-a-time output normalize | RETAIN | GPU5 profile total 14.832 ms，三轮均低于 M2 |
| P06 最终默认构建与收口复验 | PASS | profile total 14.936 ms；GPU median 14.768 ms |

### 8.1 GPU4 E21 promotion 结果

| Run | Q quant | K quant | Attention | Profile total | GPU median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| P-E21-1 | 0.471 ms | 0.402 ms | 14.920 ms | 15.794 ms | 15.637 ms | 15.395 ms | 15.711 ms |
| P-E21-2 | 0.471 ms | 0.402 ms | 14.933 ms | 15.807 ms | 15.598 ms | 15.423 ms | 15.756 ms |
| P-E21-3 | 0.472 ms | 0.402 ms | 14.956 ms | 15.830 ms | 15.586 ms | 15.417 ms | 15.693 ms |

三轮 GPU median 中位为 **15.598 ms**，profile total 中位为 **15.807 ms**；
相对 GPU4 release E03 的 17.117/16.889 ms 分别降低 **8.87%/6.41%**。
三轮 production hash 均为 `d8fccefb0ea98938`，且计时前完整 GPU correctness
与 ISA 门禁通过。GPU4 这次绝对延迟比 GPU5 E21 慢约 0.6 ms，但轮间稳定，E21
相对 E03 的收益远大于设备状态差异；因此通过 promotion 性能门禁。

### 8.2 E23--E27：M2 冲刺结果

E21 的 GPU5 profile total 距离 M2 仅 0.214 ms，而 Q/K quant 合计约 0.864 ms。
现有 quant kernel 使用 256-entry LDS 做完整 block tree reduction，每次量化需要
8 轮 reduction barrier，加上初始化共 9 次同步。E23 改为每个 wave 先用 wave32
shuffle 求最大值，仅由 8 个 lane 写入 LDS，再由 wave0 汇总；预期把同步降至
2 次，保持量化定义、workspace 和 attention kernel 完全不变。先执行 CPU/GPU
correctness 与 hash 门禁，再在 GPU5 计时；若未达到 3% 独立接受线但直接跨过
15.000 ms M2，将按里程碑收益保留并用三轮稳定性确认。

静态与数值门禁已通过：两个 quant kernel 均为 9 VGPR、32-byte group segment、
0 scratch，attention 保持 208 VGPR/3328 B/0 scratch；所有定向 case 和 production
hash 不变。首轮为 Q 0.454 ms、K 0.385 ms、attention 14.314 ms、profile total
15.154 ms、GPU median 15.336 ms，尚未跨过 M2，继续补足三轮判断稳定收益。

E23 另两轮为 Q/K 0.454/0.387、0.454/0.386 ms，profile total 15.127/
15.305 ms，GPU median 15.304/15.376 ms。三轮 profile total 中位 **15.154 ms**、
GPU median 中位 **15.336 ms**，相对 E21 仅降低 0.39%/0.12%，没有跨过 M2，
也没有解锁 attention 结构；按既定 3% 接受规则回退。E24 随后测试 48-row task
与 96-thread/3-wave workgroup，补齐 E21 32-row 和 E22 64-row 之间的粒度。

E24 静态资源为 206 VGPR、4992-byte group segment、0 scratch，全部数值与 ISA
门禁通过；production task 数为 115。首轮 attention 16.829 ms、profile total
17.677 ms、GPU median 17.622 ms，明显慢于 E21，因此快速回退。E25 转向 task
元数据 liveness：当前 52-byte task 被每个 lane 私有复制后由编译器提升到 LDS，
恰好形成 52 * 64 = 3328 bytes group segment；候选改为整个 workgroup 共享一份
只读 task，以一次入口 barrier 换取 LDS 占用和重复元数据搬运的大幅降低。

E25 静态门禁得到 202 VGPR、52-byte group segment、0 scratch，全部正确性、
hash 与 WMMA ISA 检查通过。首轮 attention 14.166 ms、profile total 15.028 ms、
GPU median 15.127 ms，距离 M2 只差 0.028 ms；继续补足三轮稳定性数据。

后两轮 profile total 为 15.047/15.169 ms，GPU median 为 15.205/15.223 ms；
三轮中位分别为 **15.047 ms** 和 **15.205 ms**，相对 E21 改善 1.10%/0.97%。
虽然低于 3% 独立接受线且尚未稳定跨过 M2，但它将 VGPR 208 降到 202、group
segment 3328 B 降到 52 B，为后续 metadata 和 occupancy 优化直接解锁结构，故按
结构例外保留。E26 在此基础上改为直接读取 uniform task 字段，尝试继续去掉这
52 B LDS 与一次 workgroup barrier。

E26 静态资源进一步降为 **192 VGPR、0-byte group segment、0 scratch**，SGPR
为 38；correctness、hash 和双 WMMA ISA 均通过。首轮 Q/K 为 0.470/0.393 ms，
attention 14.110 ms、profile total **14.974 ms**、GPU median **14.788 ms**，首次
跨过 15.000 ms M2。按里程碑门禁继续补足三轮，不能用单轮越线代替稳定结论。

E26 三轮正式结果：

| Run | Q quant | K quant | Attention | Profile total | GPU median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| E26-1 | 0.470 ms | 0.393 ms | 14.110 ms | 14.974 ms | 14.788 ms | 14.775 ms | 14.820 ms |
| E26-2 | 0.472 ms | 0.396 ms | 14.038 ms | 14.906 ms | 14.769 ms | 14.733 ms | 14.879 ms |
| E26-3 | 0.473 ms | 0.397 ms | 14.121 ms | 14.990 ms | 14.783 ms | 14.720 ms | 14.833 ms |

profile total 中位为 **14.974 ms**，GPU median 中位为 **14.783 ms**；相对 E21
分别降低 **1.58%** 和 **3.72%**。按 profile total 计算有效吞吐为 **47.282
TOPS**、混合峰值占比 **18.550%**；attention-only 中位 14.110 ms，对应
**19.686%**。三轮 profile total 全部低于 15.000 ms，M2 正式达成。E26 接受为
GPU5 新候选，并进入 GPU4 promotion；production hash 保持 `d8fccefb0ea98938`。

GPU4 promotion 首批三轮中，P-E26-1 出现 measured max 26.933 ms，profile total
16.984 ms；该轮 quant 也同步放大，按污染/离群规则丢弃。紧接着两轮 profile total
为 15.267/15.211 ms，GPU median 为 15.726/15.784 ms，hash 与正确性保持通过。
轮后 GPU4 telemetry 仍为 0% use/0% VRAM，但系统中 GPU7 出现一条完整
`h3_vdn_forward` 任务；继续在 GPU4 补独占稳定轮次，不用首批混合状态升级基线。

补测 P-E26-4/5 仍分别出现 measured max 20.977/25.295 ms，profile total 为
15.711/15.574 ms；P-E26-6 恢复到 profile total 15.357 ms、GPU median 15.274 ms。
由于没有取得连续三轮无长尾结果，P05 暂缓，requirements 中 GPU4 正式 baseline
保持 E21，不把 GPU5 的 M2 结果直接外推为 GPU4 release 数据。E27 回到稳定的
GPU5 继续：将最终输出循环改为每个 row 只保留一个 denominator inverse，并立即
写完 8 个 depth fragment，尝试降低 192 VGPR 的尾段 liveness；除法与 shuffle
次数仍各为 8 次，数学顺序不变。

E27 correctness/hash/ISA 门禁通过，静态资源保持 192 VGPR、0 LDS、0 scratch；
编译后的 attention ISA 主体明显缩短。首轮 Q/K 0.467/0.391 ms、attention
13.881 ms、profile total **14.738 ms**、GPU median **14.717 ms**，把 M2 裕量
从 E26 的 0.026 ms 扩大到 0.262 ms；继续补足三轮。

E27 三轮正式结果：

| Run | Q quant | K quant | Attention | Profile total | GPU median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| E27-1 | 0.467 ms | 0.391 ms | 13.881 ms | 14.738 ms | 14.717 ms | 14.669 ms | 14.777 ms |
| E27-2 | 0.470 ms | 0.400 ms | 13.963 ms | 14.832 ms | 14.746 ms | 14.715 ms | 14.774 ms |
| E27-3 | 0.475 ms | 0.401 ms | 13.994 ms | 14.870 ms | 14.731 ms | 14.711 ms | 14.776 ms |

profile total 中位为 **14.832 ms**，GPU median 中位为 **14.731 ms**，attention
中位为 **13.963 ms**。相对 E26 再降低 0.95%/0.35%，相对 E21 累计降低
2.51%/4.06%。按 profile total 计算有效吞吐为 **47.735 TOPS**、混合峰值占比
**18.728%**；attention-only 为 **50.706 TOPS**、峰值占比 **19.893%**。
E27 单项低于 3% 接受线，但数学操作数不变、ISA 更短且三轮都把 M2 裕量扩大，
因此作为 E26 上的低风险结构优化保留。最终默认源码为 E27，进入 P06 收口复验。

### 8.3 本轮最终状态

使用空 `SAGE_AMDGPU_FLAGS` 强制重建默认产物后，CPU contract、GPU5 完整 T01、
output/workspace guard、determinism 和 INT8 QK/BF16 PV WMMA ISA 全部通过。最终
attention kernel 为 **192 VGPR、38 SGPR、0 LDS、0 scratch**；收口复验得到
Q/K 0.469/0.397 ms、attention 14.070 ms、profile total **14.936 ms**、GPU
median **14.768 ms**，production hash 为 `d8fccefb0ea98938`。

当前默认源码保留 E27，GPU5 正式三轮结论采用 profile total 14.832 ms、GPU
median 14.731 ms，M2 已达成。GPU4 正式 release baseline 仍为 E21；E26/E27
需要等 GPU4 没有 20--26 ms 系统长尾的连续独占窗口再做 promotion。实验队列
已无 `RUNNING` 项。

## 9. E27 GPU4 promotion 与 R2 集成会话（R2 已完成）

2026-09-04 按要求继续推进。启动时物理 GPU4 为 0% use、0% VRAM、约 14 W，
但物理 GPU7 正在运行一条完整 `h3_vdn_forward`（约 15.3 GiB VRAM、42% use）。
鉴于上一轮 GPU4 的 20--26 ms 长尾也出现在同类系统状态下，本次先连续采样 GPU4，
再执行 E27 correctness/ISA/resource 和三轮 benchmark；只有连续三轮无长尾才升级
release baseline。R2 集成随后在 `h3-vdn.c` 的实际接口和构建约束内推进。

| 项目 | 状态 | 当前结果 |
|---|---|---|
| P07 GPU4 连续独占检查 | PASS | 连续三次 0%/0%，25--26 C，14 W，PCIe x16 |
| P08 E27 GPU4 correctness/ISA | PASS | 完整 T01/guard/ISA；192 VGPR、0 LDS/scratch |
| P09 E27 GPU4 三轮 benchmark | DEFER | 三轮仍有 20.857--26.685 ms 长尾，数据作废 |
| R2-01 下游工作树与接口审计 | PASS | HEAD 已有旧实验路径；确认缺口为 E27 热核与 task cache |
| R2-02 模块接入与显式 dispatch | PASS | E27、167-task cache、显式 mode 与默认回退均已接入 |
| R2-03 单层/50 层验证 | PASS | 17x256 production、逐层误差与独立 profile 均通过 |
| R2-04 最终构建与回归 | PASS | all、contract、quant、WMMA、mask、VDN GPU ops 均通过 |

P09 每轮前 GPU4 telemetry 均为 0% use/0% VRAM、25--26 C、14 W，但三轮 10 次
measured run 的 max 分别为 26.685/26.611/20.857 ms；前两轮 Q quant 还异常放大
到 0.982/0.995 ms。该结果再次证明“设备瞬时空闲”不等于当前系统有干净的稳定
计时窗口，因此全部作废，不升级 E21 release baseline。正确性与资源门禁独立通过。
R2 不依赖这组性能 promotion，继续在下游仓库实施显式实验 mode 接入。

### 9.1 R2 下游审计与单算子接入

下游位于 `h3_workspace/h3-vdn.c`，分支 `vdn-h3-rocm`，HEAD `9855387` 已包含
`sage-i8-bf16` 显式 mode、约 73 MiB context workspace 和一条旧 gfx12 Sage
kernel；现有未提交修改主要涉及 LoRA profiling、权重缓存和 smoke test，均原样
保留。旧 Sage kernel 使用 256-thread block、两遍 QK 和逐 tile workgroup barrier，
GPU5 ABI benchmark 为 **95.680 ms**，正确但远慢于研究 workspace 的 E27。

R2 当前改动把 task builder 固定为最多 32-row，同一 production geometry 得到
167 tasks；新增独立 `h3_vdn_sage_gfx12.hip`，只由显式 `H3_VDN_SDPA=sage-i8-bf16`
dispatch，`auto` 仍选择精确 wave32。`h3_gpu` context 缓存 geometry、device task
offset/count 和完整 QI8/KI8/scales/tasks workspace，首次 geometry miss 上传 task，
后续 50 层复用。GPU5 合同、INT8 quant、WMMA score、mask leak 和 fused attention
均通过；接入后的首轮 production ABI benchmark 为 **14.911 ms**，相对旧路径
快 **6.42x**，cosine 0.999993586、relative RMSE 0.00361013、nonfinite=0。

补充三轮下游 ABI benchmark 为 14.890/14.949/15.019 ms，中位 **14.949 ms**；
输出 hash 均为 `b9a74fa3e1008c63`，cosine/relative RMSE 完全一致。相同进程中的
精确 wave32 baseline 为 409.746/410.733/410.295 ms，中位 410.295 ms；因此
R2 standalone 实际加速为 **27.45x**（该比例是相对精确实现，不等于整体模型
加速）。ISA/resource 复验为 192 VGPR、38 SGPR、0 LDS、0 scratch。

### 9.2 真实 H3 50 层传播验证（已完成）

第一轮使用 17 帧、`latent_h=latent_w=16`，实际 layout 为 sequence=1894、每帧
64 个 video token。wave32 与 Sage 各执行完整 50 层，并逐层比较 hidden：所有层
均无 non-finite；最大 relative RMSE 出现在第 22 层，为 1.8646%，对应 cosine
0.999826697；第 50 层为 0.8885% / 0.999960541。最终 video/audio 输出合并后的
relative RMSE 为 **0.7834%**、cosine **0.999969334**、invalid=0。

该形状下 wave32 forward 为 **12.733 s**，E27 Sage 为 **10.638 s**，整体降低
**16.45%**。这已经证明真实 50 层传播和 context task cache 可用，但它不是目标
256 token/frame：16x16 latent 经模型 patch 后只有 64 token/frame。下一轮改用
`latent_h=latent_w=32`，验证实际 production 256 token/frame 形状。

第二轮使用 `latent_h=latent_w=32`，实际 layout 为 **sequence=5158**、video rows
4352，即完整 17 帧 x 256 token/frame；video_start 随当前 prompt 为 806。逐层均无
non-finite，最差 relative RMSE 仍在第 22 层，为 **1.8525%**，cosine
**0.999828402**；第 50 层为 **0.7843% / 0.999969469**。最终 video/audio 输出
relative RMSE 为 **0.5970%**、cosine **0.999982179**、max_abs 0.114214、invalid=0。

production wave32 forward 为 **28.729 s**，E27 Sage 为 **18.777 s**，节省
**9.952 s（34.64%）**，整体 speedup **1.530x**，peak live memory 7.580 GiB。
双跑 profile 的 SDPA 15.498 s 是两种 mode 的合计，下一步用独立进程分别运行
wave32 与 Sage，取得可直接比较的 50 层 SDPA 累计时间后完成 R2-03 门禁。

独立 profile 结果：wave32 的 50 层 SDPA 累计 **14.865 s**，forward **28.404
s**；E27 Sage 的 SDPA 累计 **0.906 s**，forward **14.884 s**。因此真实 H3
SDPA 降低 **93.91%（16.41x）**，完整 forward 降低 **47.60%（1.908x）**，
每次 forward 节省 13.520 s。峰值 live memory 从 4.929 GiB 增至 4.998 GiB，
增量约 70.7 MiB，与预期 Sage workspace 一致。

profile 同时暴露一处计数缺口：Sage 分支提前返回导致 `attention` dispatch count
只显示非 Sage 的 2 次，而不是总计 52 次；SDPA event 时间本身正确。已在成功
dispatch 后补记 `mps_sdpa_dispatches`，最终回归将确认计数恢复为 52。

### 9.3 R2 收口状态

修正计数后再次独立运行 production Sage 50 层，profile 显示 `attention=52`、
SDPA **0.895 s**、forward **14.622 s**，输出 hash 与逐层双跑一致。相对独立
wave32 的 SDPA 14.865 s、forward 28.404 s，最终可复现结论为 SDPA 降低
**93.98%（16.61x）**，完整 forward 降低 **48.52%（1.943x）**，单次 50 层
节省 13.782 s；peak live memory 增量约 70.7 MiB。

最终已通过下游 `make all`、`git diff --check`、167-task CPU contract、Q/K INT8
bitwise quant、WMMA score、mask leak、融合 attention correctness、VDN QK/RoPE、
chunk/anchor/softmax gate 和 production 50 层传播。`auto` 默认仍走精确 wave32，
E27 仅由显式 `H3_VDN_SDPA=sage-i8-bf16` 启用。

本会话 R2 集成目标完成。GPU4 E27 promotion 仍因每轮前 0%/0% 时也出现
20.857--26.685 ms 系统长尾而暂缓，GPU4 正式 release baseline 保持 E21；等待
真正无整机干扰的窗口后只需重跑 P09，不影响已完成的 GPU5 R2 证据。

## 10. 完整 8-NFE、VAE/mux 与质量门禁（阶段收尾：部分通过）

2026-09-04 进入完整生成验证。性能与传播实验继续固定在物理 GPU5；物理 GPU4
只在整机无其他完整 H3 任务且连续采样稳定后补做 E27 promotion。启动审计时
GPU4 本身为 0% use/0% VRAM，但 GPU7 仍存在完整 H3 工作负载，因此不立即重复
P09，避免再次采集已知的系统级 20--26 ms 长尾。

| 项目 | 状态 | 当前结果 |
|---|---|---|
| N01 production 8-NFE wave32/E27 成对 latent | PASS | RelRMSE 0.2990%，cosine 0.999995530；231.674 -> 116.994 s |
| E2E-01 512x512、56 帧、8-NFE | PASS | seed0 两份 MP4 均通过；356.76 -> 230.88 s，1.545x |
| E2E-02 video/audio/mux 门禁 | PARTIAL | video/mux/sync PASS；audio 质量未过冻结线 |
| Q01 多 prompt/seed | PARTIAL | production seed 0/1/2 已覆盖；仍缺 2 组真实 prompt |
| P10 GPU4 E27 promotion | PASS | profile 14.892 ms、GPU median 15.121 ms；正式 baseline 升级 E27 |

测试工具同步增加 `VDN_E2E_SEED`，默认值仍为 0；这只改变测试输入噪声，不改变
生产 pipeline 或 scheduler。多 seed 结果不能替代不同 prompt/sequence 的正式门禁，
因此在补齐至少两份真实 prompt embedding 前，M6 不会标记完成。

### 10.1 Production 8-NFE latent 成对结果

GPU5 上使用同一进程、同一 prompt、同一确定性初始 latent，先运行精确 wave32，
恢复输入后再运行 E27。实际 layout 为 sequence=5158、video rows=4352（17x256），
8 个 NFE 的最终 video/audio 合并结果如下：

| 指标 | wave32 | E27 Sage | 变化 |
|---|---:|---:|---:|
| 8-NFE wall | 231.674 s | 116.994 s | -49.50%，1.980x |
| 最终 latent relative RMSE | 0 | 0.2990% | 低于 5% 研究门槛 |
| 最终 latent cosine | 1 | 0.999995530 | 高于 0.999 门槛 |
| 最终 latent max abs | 0 | 0.105916 | -- |
| non-finite | 0 | 0 | PASS |

E27 各 NFE wall 为 14.508/14.446/14.430/14.560/14.766/14.742/14.797/
14.745 s；对应 SDPA 为 0.725--0.732 s/NFE，未出现随扩散步数增加的异常放大。
最终 candidate video/audio hash 为 `483d55ede31e3a46`/
`48017587d9c3b1ba`。该结果完成 latent 门禁，VAE 后质量仍需由后续 E2E 成对产物判断。

### 10.2 VAE 后正式质量线（首组指标读取前冻结）

在读取第一组成对 MP4 的差分指标前，冻结以下门槛，后续 prompt/seed 不得事后放宽：

- 解码后 RGB 视频：全帧平均 PSNR >= 30 dB、SSIM >= 0.95；逐帧 P05 分别
  >= 25 dB 和 >= 0.90；
- 解码后音频：双声道合并 correlation >= 0.99、relative RMSE <= 10%，不得出现
  非有限值；
- 两个输出的帧数、尺寸、帧率、声道、采样率和样本数必须一致；各自音视频 stream
  duration 差必须 <= 1/24 秒，且 candidate 不得引入新的同步偏差；
- mux 文件必须可由 ffprobe/ffmpeg 完整解码；保留 BF16、candidate 和三处代表帧
  的并排检查材料。

这些是本项目 E27 BF16-PV 候选的首个正式 E2E 质量线。latent 仍沿用 operator
研究门槛 cosine >= 0.999、relative RMSE <= 5%、无 NaN/Inf。

### 10.3 Seed 0 完整 E2E 结果

同一 `example_0` prompt、seed 0、scheduler 和编码参数下分别生成并保留：

- `outputs/vdn-e2e-512-seed0-wave32.mp4`：2,315,918 bytes；
- `outputs/vdn-e2e-512-seed0-sage-e27.mp4`：2,317,114 bytes；
- `outputs/vdn-e2e-512-seed0-contact.png`：第 0/28/55 帧，顶行为 wave32、底行为 E27。

两份均为 56 帧 512x512 H.264、24 fps，AAC stereo 32 kHz；video/audio stream
duration 分别为 2.333333/2.325000 s，8.333 ms 差值完全一致，ffmpeg 可完整解码。

| 指标 | 结果 | 冻结门槛 | 判定 |
|---|---:|---:|---|
| Video PSNR average / P05 | 40.661 / 38.930 dB | 30 / 25 dB | PASS |
| Video SSIM average / P05 | 0.983074 / 0.978187 | 0.95 / 0.90 | PASS |
| Audio correlation | 0.981459 | >= 0.99 | **FAIL** |
| Audio relative RMSE | 19.232% | <= 10% | **FAIL** |
| Audio non-finite | 0 | 0 | PASS |
| A/V duration delta | 8.333 ms | <= 41.667 ms | PASS |

完整进程 wall 从 wave32 的 **356.76 s** 降到 E27 的 **230.88 s**，降低
**35.29%（1.545x）**；其中 8-NFE DiT 从 246.230 s 降到 119.637 s，降低
51.41%（2.058x）。video VAE 为 107.743/108.387 s，audio VAE 为
1.631/1.579 s，符合 decoder 与 SDPA mode 无关的预期。

结论：E27 已通过 production 8-NFE latent、视频、mux、同步和完整 E2E 性能门禁，
但音频没有通过首轮冻结质量线，因此保持 explicit experimental，不能 promotion
为 stable。下一候选优先评估仅对 text/audio global query 保留精确 wave32、video
query 使用 E27 的混合策略；在音频过线前不以多 seed 重复代替修复。

### 10.4 H01 global-query 精确混合候选（REJECTED）

新增显式 `sage-i8-bf16-hybrid`：先执行 E27，然后只对 video 区间之外的 global
query 用精确 wave32 覆盖输出。`auto` 和原 `sage-i8-bf16` 行为均不改变；suffix
non-video query 同样覆盖，保证不是只针对当前视频位于 sequence 尾部的样例特化。

GPU5 production sequence=5338 的单 NFE 成对初筛：wave32 forward 31.203 s，H01
为 18.131 s，降低 41.89%（1.721x）。video output relative RMSE/cosine 为
0.6176%/0.999980940；audio output 为 **3.5129%/0.999383403**，无 non-finite。
该结果显著改善 audio query 的单步传播且保留净收益，进入完整 8-NFE latent 门禁。

H01 完整 8-NFE 为 240.482 -> 144.571 s，降低 39.88%（1.663x）；最终 video
latent 为 0.2355%/0.999997230，但 audio latent 累积到 **10.8571%/0.994089481**。
虽然合并 relative RMSE 仅 0.6401%、cosine 0.999979515，audio 分项已略高于冻结的
10% 线，故不直接花费一次完整 VAE。H02 把同为 global query 的首尾 anchor frame
也切回精确 wave32，再按单 NFE -> 8-NFE -> VAE 的顺序逐级门禁。

H02 单 NFE 初筛未通过：audio relative RMSE/cosine 为
3.9847%/0.999206008，反而差于 H01 的 3.5129%/0.999383403；forward 也从
18.131 s 增至 21.159 s。该变化不是稳定的精度改善，立即回退 anchor 覆盖，
`sage-i8-bf16-hybrid` 恢复为 H01，仅精确覆盖 non-video global query。

H01 的 VAE 后 video 继续通过（PSNR 40.682 dB、P05 38.950 dB；SSIM
0.983140、P05 0.978280），但 audio correlation/relative RMSE 仅改善为
0.985745/16.874%，仍未达到 0.99/10%。完整 wall 为 256.42 s，相对 wave32
356.76 s 降低 28.13%（1.391x）。Q8/K16 scale-group 候选单 NFE audio 为
3.8323%/0.999268081，差于 H01，已回退且不保留额外 mode。

H03 改为 NFE 级精度调度：`H3_VDN_EXACT_EDGE_NFE=1` 时，显式 Sage mode 的首尾
NFE 使用 wave32，中间 6 步使用 H01；变量未设置时所有既有 mode 行为不变。
该候选理论 DiT wall 约 181 s，仍比全 wave32 快约 1.33x，先以完整 8-NFE latent
验证是否足以让 audio 回到冻结线内。

H03 实测候选 wall 为 169.827 s，相对同轮 wave32 241.499 s 提升 1.422x；video
latent 为 0.2132%/0.999997727，但 audio 为 **11.3370%/0.993553845**，没有改善
H01。H03 拒绝，调度环境变量与实现均回退，不进入完整 VAE。

H01/H02/H03 与 Q8/K16 候选最终都未通过冻结的 audio 门禁；为避免给下游增加一个
“更准但仍不合格”的长期模式，相关 hybrid/HQ/NFE 调度实现全部从最终源码移除，
只在本日志保留可复现实验结果。最终下游 mode 集合保持 `auto/scalar/wave32/`
`sage-i8-bf16` 以及尚未启用的 F16/FP8 保留名。

### 10.5 GPU4 E27 最终 promotion（PASS）

18:46 获得真正整机干净窗口：连续四次采样中物理 GPU4 均为 0% use、0% VRAM、
26 C、14 W，其他 GPU 也没有 H3 GPU workload。随后重新通过完整 GPU correctness、
T01、guard canary、determinism 和双 WMMA ISA 门禁，并连续执行三轮 10-iteration
production benchmark：

| Run | Q quant | K quant | Attention | Profile total | GPU median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| P-E27-clean-1 | 0.471 ms | 0.394 ms | 13.970 ms | 14.835 ms | 15.201 ms | 14.727 ms | 15.744 ms |
| P-E27-clean-2 | 0.471 ms | 0.399 ms | 14.041 ms | 14.911 ms | 15.121 ms | 14.754 ms | 15.791 ms |
| P-E27-clean-3 | 0.469 ms | 0.402 ms | 14.021 ms | 14.892 ms | 15.107 ms | 14.759 ms | 15.730 ms |

三轮中位为 profile total **14.892 ms**、GPU median **15.121 ms**、attention
**14.021 ms**；production hash 均为 `d8fccefb0ea98938`。本轮没有此前的 20--26 ms
长尾，max 全部 <= 15.791 ms。相对 GPU4 E21 release baseline 的 15.807/15.598 ms，
profile/GPU median 分别降低 **5.79%/3.06%**，因此 E27 正式通过 GPU4 promotion，
取代 E21 成为 research workspace 的 GPU4 release baseline。该结论只升级算子基线；
H3 stable 仍被完整 E2E audio 质量和多 prompt 资产门禁阻止。

### 10.6 多 seed 压力门禁（PARTIAL）

`test_vdn_forward_smoke` 新增可选 `VDN_SMOKE_RANDOM_SEED`；未设置时继续使用原有
sin/cos 确定性 fixture，设置后以与 E2E 相同的 H3 RNG 生成 video/audio 正态初始
latent。seed 1/2 使用 production sequence=5338、完整 8-NFE 成对比较；该测试只
用于确认误差稳健性，不把一个 prompt 的不同 seed 记作多 prompt 已完成。

seed 1/2 均完成 wave32/E27 同输入成对 8-NFE：

| Seed | wave32 wall | E27 wall | Combined RelRMSE/cosine | Video RelRMSE/cosine | Audio RelRMSE/cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 241.130 s | 117.960 s | 0.2136% / 0.999997720 | 0.1266% / 0.999999199 | 5.2778% / 0.998613470 |
| 2 | 241.386 s | 117.793 s | 0.2114% / 0.999997765 | 0.1269% / 0.999999195 | 5.0597% / 0.998719151 |

两组均无 non-finite，E27 latent wall 分别提升 2.044x/2.049x。随后各自完成 Sage
production VAE/mux：seed1/seed2 MP4 均为 56 帧 512x512、24 fps、stereo 32 kHz/
74400 samples，大小 2,316,145/2,319,546 bytes，进程 wall 231.79/231.35 s；latent
hash 与成对测试一致。并发双卡首轮曾被系统同时终止且无最终输出，该数据作废；
上述结论来自 GPU5 串行复测。

三 seed 结论不是“全部质量通过”：seed 1/2 的 production latent 分项通过，但
seed 0 的完整 VAE 后 audio 已明确失败。仓库媒体门禁脚本的独立复核也判定纯 E27
失败项为 audio cosine、relative RMSE 和 SI-SDR；H01 仍失败 cosine/relative RMSE。
因此 E27 保持 experimental，并且由于只有一组 `example_0` embedding，多 prompt
门禁继续 BLOCKED，而不是以 seed 数量替代 prompt 数量。

### 10.7 阶段收尾与下一入口

最终源码不保留 H01/H02/H03、Q8/K16 HQ 或 NFE 调度试验分支。收尾回归重新通过
`make all`、28,512,043 项 Sage mask/workspace contract、Q/K INT8 bitwise、WMMA
score 与融合 attention correctness；融合 attention relative RMSE 为 0.2181%、
cosine 为 0.999997835，mask mismatch 为 0。`VDN_E2E_SEED` 的非法输入也确认会在
GPU 与模型加载前以状态 2 快速失败。

本阶段最终状态为：**E27 operator/GPU4 promotion PASS，production 8-NFE、video、
VAE/mux、同步与端到端性能 PASS，seed 0 audio 质量 FAIL，多 prompt BLOCKED**。
因此 research baseline 已升级到 E27，但下游 `auto` 必须继续保持精确 wave32。
下一轮只从两个入口继续：先定位并降低 audio/global-query 的量化误差；获得至少两份
新的真实 prompt embedding 后，再按已冻结门槛补齐三 prompt 成对媒体门禁。

## 11. Audio/global-query 误差归因与修复（进行中）

2026-09-05 开始下一阶段。所有结果继续使用 GPU5、`example_0`、seed 0 和 production
latent geometry（17x32x32、audio latent 93、sequence 5338），并在同一进程内先跑
wave32 reference、恢复相同输入后再跑 candidate。

### 11.1 逐层 modality 误差与层级调度

`test_vdn_forward_smoke` 的逐层 observer 已拆分 text/audio/video 三段，避免 audio
被 4352 个 video rows 稀释。纯 E27 的 audio hidden relative RMSE 在 layer 1/10/20/
30 分别为 0.1860%/0.2724%/0.2985%/0.5323%，从 layer 31 后开始持续放大，layer 34
越过 1%，layer 50 达 3.2319%；最终 audio velocity 为 1.5398%。mask、non-finite
和首层没有异常突变，因此当前证据不支持“global-query mask 错误”。

A01 将后 20 层切回精确 wave32，只把最终 audio relative RMSE 改善到 1.1926%，
candidate/reference wall 为 26.879/29.308 s，收益仅 1.09x。精确尾层不能消除已经
进入 residual stream 的误差，故该方向快速拒绝，不进入 8-NFE。

### 11.2 Q/K scale 与 PV 精度归因

| Candidate | QK | PV | Video RelRMSE | Audio RelRMSE | Candidate/reference wall | 判定 |
|---|---|---|---:|---:|---:|---|
| A00 E27 | INT8 Q32/K64 | BF16 | 0.4964% | 1.5398% | observer run，不作正式性能值 | baseline |
| A02 row-scale | INT8 Q1/K1 | BF16 | 0.4787% | 1.4029% | 14.484/30.438 s | REJECT |
| A03 F16-PV | INT8 Q32/K64 | F16 | 0.4990% | 1.4210% | 14.209/28.872 s | REJECT |
| A04 exact-QK | BF16 scalar | BF16 | 0.4459% | 1.1209% | 16.814/29.240 s | diagnostic |
| A05 exact-QK + F16-PV | BF16 scalar | F16 | 0.4400% | **1.0162%** | 16.590/29.234 s | 进入 8-NFE |

Q1/K1 和单独 F16-PV 都只带来约 8--9% 的 audio 改善，不能证明分组 scale 或 BF16
PV 是单一主因。exact-QK 带来更明显改善，而与 F16-PV 组合后相对 A00 改善约 34%，
说明误差由 INT8 QK、低精度 PV/online softmax 和跨层放大共同构成。A05 当前仍是
归因实现：QK 使用标量 BF16 FMA，尚未替换为优化的 BF16 WMMA。下一门禁为完整
8-NFE audio latent；若不能显著压低误差，则不执行完整 VAE。

### 11.3 A05 完整 8-NFE 初筛（PASS）

A05 在同一进程完成 production 8-NFE wave32/candidate 成对运行：wall 为
235.774/133.833 s，提升 **1.761x**。最终 combined relative RMSE/cosine 为
0.1601%/0.999998719，video 为 0.1170%/0.999999316，audio 为
**3.3512%/0.999438625**，全部无 non-finite。reference video/audio hash 与 seed 0
正式 E2E 的 `e77bfd64f14b695c`/`bdde023376238608` 一致，确认输入与 scheduler 合同
相同。A05 通过 latent 初筛，进入完整 VAE/mux；是否接受仍只由冻结的解码后 audio
correlation >= 0.99、relative RMSE <= 10% 决定。

### 11.4 A05 完整媒体门禁（REJECTED）

A05 完整 E2E 的 DiT wall 为 133.833 s，相对同轮 wave32 的 235.774 s 提升
**1.761x**；产物 `outputs/vdn-e2e-512-seed0-sage-a05.mp4` 为 56 帧
512x512、24 fps、stereo 32 kHz/74400 samples，VAE、mux 和完整解码均通过。
与既有 seed 0 wave32 参考媒体成对比较：video PSNR/SSIM 为
42.085 dB/0.981612，继续通过；audio correlation 为 **0.987678**、relative
RMSE 为 **15.683%**，虽优于 E27 的 0.981459/19.232%，但仍同时未达到冻结的
0.99/10%。A05 因此拒绝 promotion，不能把单纯的 latent PASS 当作媒体 PASS。

### 11.5 后续归因候选（REJECTED）

在 A05 基础上继续做单 NFE production 初筛：

| Candidate | 变化 | Video RelRMSE | Audio RelRMSE | Candidate/reference wall | 判定 |
|---|---|---:|---:|---:|---|
| A06 | text/audio global query 用 wave32 覆盖 | 0.4352% | 1.1474% | 20.255/29.463 s | REJECT |
| A07 | 奇偶层交替精确 wave32 | 0.4407% | 1.0440% | 23.056/29.264 s | REJECT |
| A08 | A05 + FP16 high/FP16 residual 双 PV | 0.4331% | **0.9682%** | 16.679/29.284 s | diagnostic |
| A09 | BF16 rocWMMA QK + F16 PV | 0.4363% | 1.0774% | 16.032/29.174 s | REJECT |

A06 说明按 query modality 覆盖并不能隔离 residual stream 中已经产生的误差；A07
付出近一半精确层成本仍没有超过 A05。A08 的补偿 PV 仅比 A05 再改善约 4.7%，但
是本组唯一不明显牺牲吞吐的改善。A09 已用独立 16x16 tile oracle 验证 rocWMMA
QK 与 CPU BF16 dot bitwise 一致，因此其退化不是 fragment 映射错误；完整 H3
结果仍不如 A08，暂不进入 8-NFE。

### 11.6 A10 NFE 隔步纠偏（REJECTED）

A10 令奇数 NFE 使用 A08、偶数 NFE 使用精确 wave32，目标是在约一半精确成本下
周期性消除误差。GPU5 的 wave32 基线连续出现与候选无关的 Cholesky leading-minor
失败，故这些点全部作废；迁移至已单独通过纯 wave32 50 层基线的干净 GPU4 后重跑。

GPU4 的纯 wave32 reference 完整通过 8 NFE，A10 候选则在第 7/8 NFE、block
40--50 之间触发 `VDN video Cholesky solve` 失败（batch 20、leading minor 97）。
这表明隔步精确并没有把状态拉回稳定轨道，反而允许误差跨 NFE 累积到线性求解失稳。
A10 立即拒绝，不进入 VAE/mux。下一候选改为每个 NFE 内的末段层精确纠偏，并先做
单 NFE 快速失败测试；若质量/速度无实质改善，则停止调度类试验并清理实验分支。

### 11.7 层内精确纠偏（REJECTED）

A11 在每个 forward 的前 30 层使用 A08、后 20 层使用 wave32；单 NFE audio
relative RMSE 为 1.0935%，比纯 A08 的 0.9682% 更差，wall 为
22.162/29.694 s。A12 反转顺序，前 30 层 wave32、后 20 层 A08；audio relative
RMSE 仍为 0.9954%，wall 为 25.784/30.664 s，也没有超过纯 A08。两种方向均说明
误差不能用简单的前后层切换隔离，且切换边界会产生新的轨迹差异。层/NFE 调度类试验
至此停止；只对不改变执行轨迹的 A08 本体补一次完整 8-NFE latent 门禁。

### 11.8 A08 完整 8-NFE 与 A13 QK 加法树（REJECTED）

A08 在 GPU4 完成无 mode 切换的完整 8-NFE 成对测试，reference/candidate wall 为
238.558/136.597 s（**1.746x**）。video latent relative RMSE/cosine 为
0.1176%/0.999999309；audio 为 **3.6599%/0.999331601**，反而差于 A05 的
3.3512%/0.999438625。单 NFE 的小幅改善没有跨 scheduler 保持，故 A08 不进入
VAE/mux。

A13 把 A08 的标量 BF16 QK 从顺序 FMA 改成严格复现 wave32 的 32-lane reduction
加法树。单 NFE audio relative RMSE 为 1.0632%，且 wall 从 A08 的 16.679 s
增至 21.739 s；该归因同样拒绝。结合 A05--A13，剩余差异主要来自 16-key tile
online-softmax/PV 与逐 key wave32 的结合顺序。若继续追求 bitwise 轨迹，必须退回
逐 key 更新，等价于放弃当前 Sage 吞吐结构。

本轮没有候选同时通过性能与解码后 audio 门禁。所有 row-scale、F16-PV、exact-QK、
compensated-PV、BF16-WMMA、global-query 覆盖以及 layer/NFE 调度生产分支均已从源码
移除；默认 E27 与 `auto` 行为不变。只保留 `VDN_SMOKE_SAGE_LAYER_ERRORS=1` 的
text/audio/video 逐层诊断输出，供后续新内核做相同归因。多 prompt 仍因只有
`example_0.safetensors` 一份真实 embedding 而 BLOCKED。

### 11.9 清理后回归与阶段结论

清理后 `make BACKEND=hip -j16 all` 无 warning/error。GPU4 重新通过：

- 28,512,043 项 Sage mask/workspace contract；
- Q/K INT8 bitwise hash `eb5596d4f339262f`/`655e45aee655cfb8`；
- WMMA score max-abs 0；融合 attention relative RMSE 0.2181%、cosine
  0.999997835、mask mismatch 0；
- production sequence=5338 的 wave32/E27 50 层对照，无 non-finite，video/audio
  velocity relative RMSE 为 0.4964%/1.5398%，与归因前 A00 一致；reference output
  hash 仍为 `7b1f5534293946bd`/`c7272762443945fd`。

带逐层 observer 的 wall（29.314/19.834 s）包含每层整张 hidden snapshot/readback，
不得替代 10-iteration operator promotion 数值；正式 E27 性能基线仍采用 10.5 的
GPU4 clean-window 14.892 ms profile total / 15.121 ms GPU median。

阶段结论：E27 继续作为高性能 research baseline，但不能成为 H3 stable/`auto`；
seed 0 的完整媒体 audio 质量仍是硬阻塞。下一条值得投入的技术路线不再是现有 kernel
上的 scale/PV/调度微调，而是设计保留逐 key softmax 数值轨迹的块化 producer/
consumer，或为 H3 训练/校准可容忍 tile-softmax 的专用低精度路径。多 prompt 门禁需
先补充至少两份真实 prompt embedding 才能继续。

### 11.10 三 prompt 门禁复核（2026-09-07，FAIL/BLOCKED）

复核时 prompt 目录已新增两份真实 embedding，故覆盖 11.8/11.9 中“只有一份资产”的
旧状态：`example_0/1/2` 分别为 800/821/1299 tokens，三份均通过 safetensors
输入合同；example_1 value/tag hash 为 `0c9b4239c48ab800`/`bd89101b90dd4cf2`，
example_2 为 `9f7cb150d5e2a724`/`086032e0710a173c`。

| Prompt | wave32 8-NFE | E27 8-NFE | Video latent | Audio latent | 后续 |
|---|---|---|---:|---:|---|
| example_0 | PASS | PASS，2.058x | 已通过媒体门禁 | 媒体 19.232%/0.981459 | FAIL |
| example_1 | 第 2 NFE Cholesky fail | 未运行 | N/A | N/A | BLOCKED |
| example_2 | PASS，389.590 s | PASS，171.934 s（**2.266x**） | 0.1395%/0.999999027 | **5.3832%/0.998550036** | FAIL-fast |

example_1 的 wave32 失败先在 GPU4 出现于 batch 564/minor 77，再在空闲 GPU6 复现于
batch 605/minor 111；两次都发生在候选启动前，因此是该 prompt/seed 的 reference
稳定性阻塞，不能归因给 E27。example_2 的 reference 独立运行与成对运行均完整通过，
reference video/audio hash 稳定为 `2df0093c616ea512`/`0f36390bbc4f4770`；E27
candidate hash 为 `921f74ce5ba0e812`/`c6f51493a9ab1220`，无 non-finite。

example_2 audio 超过冻结的 5% latent 快速失败线，故按既定 staged gate 不继续花费
完整 VAE/mux；example_1 又没有合法 reference。多 prompt 资产门禁已从“缺资产”解除，
但质量结论仍为 **0/3 完整通过**：一个媒体 audio FAIL、一个 reference BLOCKED、一个
audio latent FAIL。E27 继续保持 explicit experimental，`auto` 继续使用 wave32。

## 12. Exact-order producer/consumer 与 Cholesky 稳定性

2026-09-07 启动下一阶段，两个工作流分开验收：attention 原型必须保留 wave32 的
逐 key online-softmax 更新顺序；`example_1` 必须先让纯 wave32 reference 稳定，
不能用 Sage 候选掩盖基线故障。

### 12.1 example_1 Cholesky 初步归因

系统矩阵由 `I + Σ sigmoid(beta) * k * k^T` 构成，理论上为正定矩阵。新增仅在
`H3_VDN_SOLVE_DIAGNOSTICS=1` 时启用的 factorization 前 readback：保存 rocSOLVER
改写前的矩阵，并可报告 non-finite、对角范围、最大非对称误差及 CPU double
Cholesky minor；默认路径不改变。

未同步的 wave32 reference 已分别在 GPU4/GPU6 的第 2 NFE 失败。开启诊断后，
GPU4 却完整通过 8 NFE，sequence=5359，video/audio hash 为
`e03a22d11112aebd`/`fafe04c794cf4e84`，且没有任何失败矩阵可报告。诊断 readback
引入的 stream synchronize 消除了原故障，当前证据转向 rocSOLVER 边界的时序问题，
而不是给理论 SPD 矩阵盲目增加 diagonal jitter。下一步以不做 D2H 的单独 pre-solve
synchronize 复测，确认最小修复并测量开销。

最小 `hipStreamSynchronize`（不做 D2H）随后完整通过 example_1 8 NFE，并复现上述
hash。example_0 production 单 forward 的进程 wall 对照为未同步 45.88 s、同步
40.80 s，输出 hash 完全相同；差异由热缓存/系统噪声主导，没有可测负收益。由于求解
本来就会在 POTRF 后同步读取 `device_info`，修复最终收敛为 `add_identity` 后、调用
rocSOLVER 前的固定 stream boundary，并保留诊断环境变量用于失败矩阵分析。

默认修复下，example_1 正式 wave32/E27 成对 8-NFE 完整通过：reference video/audio
hash 再次为 `e03a22d11112aebd`/`fafe04c794cf4e84`；E27 candidate hash 为
`6a6faf27b9237781`/`fd547dc5bcdb6901`。video latent relative RMSE/cosine 为
0.1358%/0.999999078，audio 为 **4.4874%/0.998992681**，无 non-finite，已低于
5% staged gate。该轮 wall 476.950/283.590 s（1.682x），绝对值受当时系统状态明显
放大，只保留同轮相对值。example_1 已从 BLOCKED 转为进入完整 VAE/mux 媒体门禁。

第一次完整媒体尝试在 GPU4 运行到第 2 NFE 后被外部进程/会话终止；进程没有报告
Cholesky、non-finite 或模型错误，最终也没有生成 MP4。该会话累计约 9055 s 后消失，
远超正常推理时间，因此本点按环境异常丢弃，不能记作数值失败或媒体门禁结果。待获得
稳定 GPU 窗口后仍需重新完成 example_1 的 VAE/mux 与解码后音视频成对比较。

随后在干净 GPU6 窗口完成正式重跑。wave32 reference 与 E27 candidate 均通过完整
8-NFE、video/audio VAE 和 mux，分别生成 2,315,423/2,311,098-byte、56 帧
512x512、24 fps、stereo 32 kHz/74400-sample MP4。reference/candidate latent hash
分别为 `e03a22d11112aebd`/`fafe04c794cf4e84` 与
`6a6faf27b9237781`/`fd547dc5bcdb6901`，再次复现成对 latent 测试。开启 profile 的
DiT wall 为 256.676/123.530 s（**2.078x**）；video VAE wall 为 110.377/109.492 s。

解码后 video PSNR mean/min 为 41.357/38.480 dB，SSIM mean/min 为
0.977344/0.963726，temporal cosine/energy ratio 为 0.992819/1.000165，全部 PASS。
audio correlation 为 **0.983433**、relative RMSE 为 **18.1844%**、SI-SDR 为
**14.688 dB**，同时未达到 0.99、15%、15 dB 的媒体脚本门槛，更未达到此前冻结的
10% RMSE 严格线。故 example_1 的 reference BLOCKED 已解除、Cholesky 修复通过真实
E2E 验证，但 E27 对该 prompt 的最终结论仍为 **audio FAIL**。质量报告保存在
`outputs/vdn-e2e-512-prompt1-sage-e27.quality.json`。

### 12.2 Exact-order 块化 producer/consumer（REJECTED）

在空闲 GPU6（无 KFD workload）实现并逐级测试了三个保持数值轨迹的原型。所有原型
都保留 reference 的 D128 QK 加法树、逐 key `expf` online-softmax，以及 PV FMA
更新顺序；批处理只改变系数如何通过 LDS 发布给 consumer。production geometry
输出 hash 始终为 `3d65eea81de34693`，与精确 wave32 **bitwise 一致**。

稳态对照为 wave32 **412.908 ms**、E27 **15.458 ms**。这里的 standalone wave32
绝对时间远高于 E27 是预期结果：它逐 key 执行完整 BF16 QK/PV；两者只用于同轮相对
比较，不能把 wave32 数值误写成 E27 promotion 基线。

| 原型 | 结构 | Batch | 平均时间 | 相对 wave32 | 判定 |
|---|---|---:|---:|---:|---|
| E28 | 3 wave，整批 QK 后 barrier/整批 PV | 32/64/128/256 | 1458.894/1453.258/1438.686/1439.006 ms | 最佳 3.48x 慢 | REJECT |
| E29 | 1 producer + 2 consumer，双 LDS buffer 异步握手 | 32/64/128/256 | 938.088/939.608/945.209/1191.311 ms | 最佳 2.27x 慢 | REJECT |
| E30 | 1 producer + 1 consumer，双 LDS buffer | 4/8/16/32 | 730.448/719.945/705.675/696.230 ms | 最佳 1.69x 慢 | REJECT |
| E30 | 同上 | 64/128/256 | 694.162/780.542/1193.207 ms | 最佳 1.68x 慢 | REJECT |

E30 batch=32 的独立重复为 693.310 ms，说明 32/64 已在约 0.69 s 平台区；继续增大
batch 会因流水粒度和 LDS 增长而退化，继续缩小则被握手开销反噬。静态 code-object
元数据也排除了 scratch/VGPR 爆炸：E30 batch=32 为 23 VGPR、44 SGPR、1052 B LDS、
0 scratch；精确 wave32 为 34 VGPR、34 SGPR、0 LDS、0 scratch。也就是说拆分确实把
VGPR 从 34 降到 23，但多 wave 调度、LDS coefficient 流量、producer/consumer 握手和
失去单 wave 内 QK/PV 紧耦合的成本更大。

因此该 exact-order producer/consumer 路线在 gfx1201 上快速失败：最佳 E30 仍比
精确 reference 慢约 68%，比 E27 慢约 45 倍，不进入 50 层/8-NFE。所有 E28--E30
环境变量和内核分支在记录结果后从生产源码清理。若未来继续追求 bitwise 轨迹，需要
不同的融合/持久化执行模型，而不是在同一 workgroup 内继续增加 wave specialization；
当前 stable 仍保持 wave32，E27 仍保持显式 experimental。

清理后重新编译 `h3_vdn_sdpa_bench`、`h3_vdn_forward_smoke_tests` 和
`h3_vdn_solve_tests` 无 warning/error；GPU6 solve 回归 PASS。production geometry
默认 wave32 为 411.495 ms/hash `3d65eea81de34693`，E27 10-iteration 平均为
15.161 ms/hash `e928bdad19ecb388`，确认实验原型没有污染正式路径。综合三 prompt：
example_0 解码后 audio FAIL，example_1 解码后 audio FAIL，example_2 latent audio
快速失败，E27 的三 prompt promotion 仍为 **0/3**。

solve 单测同时改为在同一 command group 内执行 `frame_stats -> solve`，连续重复 64 次，
避免原先分两次 submit 把 producer/rocSOLVER 边界问题隐藏掉；GPU4 与 GPU6 均通过
该压力回归及 CPU inverse 数值核对。

### 12.3 E31 逐 key softmax state + tile PV（REJECTED）

E28--E30 保留了完整 BF16 QK/PV 轨迹，但没有利用 Sage 的 WMMA 并行。E31 因此在
E27 的 16Q x 16K wave 映射内逐 key 更新 `running_max/running_sum`，同时顺序组合
每个 key 的 `(old_scale,new_scale)` 为 tile affine coefficient；QK 继续使用 INT8
WMMA，PV 继续使用 BF16 WMMA。该变体保留逐 key softmax state，只把最终 PV affine
composition 留在 tile 粒度，是比完整 exact-order 更贴近高吞吐结构的诊断。

静态资源为 192 VGPR、64 SGPR、0 LDS、0 scratch；E27 对照为 192 VGPR、约 40 SGPR、
0 LDS/scratch，两类 WMMA ISA 均保留。GPU correctness 无 non-finite、mask mismatch=0，
但 production 10-iteration operator 从 E27 的 **15.548 ms** 退化到
**25.546 ms**（+64.3%），relative RMSE 只从 0.361013% 改善到 **0.361007%**，
cosine 只从 0.999993586 变为 0.999993587。收益远小于测量与 BF16 输出分辨率，性能
则明显失败，因此不进入单 NFE，实验开关和模板实例已从源码清理。

该结果进一步缩小了根因：只恢复逐 key 的 softmax state 舍入轨迹几乎不改变 operator
误差，剩余 audio 风险主要来自 INT8 QK 分组量化和 BF16/WMMA PV accumulation，不能
靠继续调整 softmax batch 大小解决。下一轮应先做 per-head/per-modality 误差敏感度
测量；若误差分散而不是集中在少量 head，则停止 inference-only hybrid 搜索，转向
H3 专用校准/训练或保持 wave32 stable。

### 12.4 E32 per-head 精确覆盖扫描（REJECTED）

为验证误差是否集中在少量 attention head，新增临时 exact-head overlay：E27 先计算
全部 56 heads，再仅用 wave32 覆盖指定连续 head 段。覆盖 56/56 heads 时 production
operator hash 精确恢复为 `3d65eea81de34693`，证明 range 地址与覆盖语义正确。随后在
GPU6、`example_0` production 单 NFE 上逐个扫描七组 8 heads；每组都在同进程与
wave32 reference 比较。

| 精确 head | Audio RelRMSE | Video RelRMSE | Candidate wall |
|---|---:|---:|---:|
| 0--7 | 5.4972% | 0.6360% | 16.460 s |
| 8--15 | 4.8795% | 0.6128% | 16.213 s |
| 16--23 | 5.1334% | 0.6314% | 16.187 s |
| 24--31 | 5.0681% | 0.5874% | 16.925 s |
| 32--39 | 5.5286% | 0.6069% | 16.279 s |
| 40--47 | 4.9093% | 0.6253% | 16.222 s |
| 48--55 | 5.0492% | 0.5807% | 16.575 s |

七组均无 non-finite，但没有一组优于同 fixture 的纯 E27 audio 约 1.54%；局部换成
精确 head 反而统一放大到 4.88--5.53%。由此推断 E27 的 head-wise 误差在 output
projection 中存在明显抵消，精确/近似混合会破坏该抵消，而不是存在少数可低成本回退
的“坏 head”。不继续组合 4/16-head 搜索，临时环境变量和 kernel range 接线已清理。

至此 inference-only 的 softmax batch、producer/consumer、layer/NFE/modality/head
fallback 均没有得到同时满足速度和 audio 质量的候选。后续若没有 H3 专用校准/训练
数据，应停止扩大混合规则搜索：保留 E27 作为约 18.7% 混合峰值利用率的显式研究模式，
stable/`auto` 继续使用 wave32。

### 12.5 下游 P9 重复路线复核与 smooth-K（REJECTED）

2026-09-08 在准备将本仓库作为 `h3-vdn.c` 唯一 SageAttention 子仓库时，审计发现
下游尚有一组未提交 P9 试验。全部正式样本均由物理 GPU 4/BDF `e3:00.0` 守护执行，
exit 0、concurrency guard 为空且输出 finite。完整下游 diff 已保存为 H3 Git blob
`a2ed5e0cc9a8fb056149b28abbe36c53ac9ca6cc`。

- Q8/K16 把 E27 的 Q32/K64 scale group 收紧到 8/16。S=5837/H=56/D=128 五组
  crossed operator 中位为 wave32 `0.501677 s`、候选 `0.018547 s`，但 prompt 2
  production 8-NFE audio latent relative RMSE/cosine 为
  **10.639524%/0.994340064**，同时违反 5%/0.999 门禁；video
  `0.221297%/0.999997551` 通过。该结果强化 10.4 节已有 Q8/K16 REJECT，不新增长期
  mode。
- reference-aligned smooth-K 按 `[head,dimension]` 计算真实 sequence 的 F32 K mean，
  K 减均值后仍做 K64 量化。CPU/GPU mean、INT8 bytes 和 scale oracle 逐项一致；五组
  含税 operator 中位为 `0.501287→0.019663 s`。但 prompt 2 单 NFE/50 层的 combined
  relative RMSE 为 **1.441189%**，audio 为 **5.227620%/0.998633146**，均差于 E27
  和 Q8/K16，因此按 staged gate 停止 8-NFE。数学上不改变合法 key 集 softmax 的常量
  shift 没有转化成 H3 的实际精度收益。
- 只让前 17 层使用 E27、18--50 层使用 exact wave32 时，forward
  `33.821389→33.920405 s`，combined relative RMSE/cosine 仍为
  `1.1424441%/0.999934765`，audio `4.148523%/0.999141051`。后 33 层 exact 既未消除
  已传播误差，也失去性能收益，与 A01/A11/A12 的结论一致。

因此 Q8/K16、smooth-K 和新增 layer/NFE 调度只作为失败证据保留，全部从 H3 runtime
清理。后续 Sage kernel 研究在本仓库完成并先通过 registry/operator 门禁，H3 只接入
固定提交并执行模型级质量验证。
