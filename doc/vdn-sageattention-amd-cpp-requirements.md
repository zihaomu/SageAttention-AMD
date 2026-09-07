# VDN SageAttention AMD GPU 原生 C++/HIP 实现需求规格

> 状态：Research MVP + R2 integrated / M6 audio failed
> 更新日期：2026-09-07
> 当前实现仓库：`zihaomu/SageAttention-AMD`；首个下游集成：`h3-vdn.c`，默认分支 `vdn-h3-rocm`
> 首要目标设备：AMD Radeon gfx1201，wave32
> 设备约束：正式 promotion 只使用物理 GPU 4；隔离的 GPU 5/6 可用于研究与稳定性复测
> 文档目的：明确 SageAttention 在 VDN 自定义注意力上的原生 C++/HIP 实现边界、接口、数值门禁和性能验收条件，供后续内核研究与实现使用。

已验证加速栈、失败路线和后续投入边界的简表见
[`h3-vdn-sageattention-acceleration-summary.md`](h3-vdn-sageattention-acceleration-summary.md)。

## 1. 目标结论

本项目需要的不是把 PyTorch 版 SageAttention 直接接入 h3，而是实现一条不依赖 PyTorch、Triton 或 Python 运行时的原生 C++17/HIP 路径：

- 输入保持 VDN 当前的 BF16 Q/K/V 和 NHD 内存布局；
- QK 使用对称 INT8 量化和 gfx12 WMMA；
- PV 至少研究 BF16、FP16 和 OCP FP8 E4M3 三条候选路径；
- 支持 VDN 的 `window + chunk + bidirectional anchors + global non-video tokens` 自定义掩码；
- 不物化完整的 `S × S` attention 矩阵；
- 量化、掩码、softmax、PV 和 BF16 输出在一个可流式执行的 attention pipeline 中完成；
- 保留现有 scalar oracle 与精确 wave32 BF16 实现，Sage 路径必须显式选择，不能静默成为默认路径；
- 正式 release/promotion 性能结论只在物理 GPU 4 上给出；其他空闲同型卡上的结果
  必须标为 research/diagnostic，不能改写 GPU4 baseline。

SageAttention 是近似数值优化，不能满足“BF16 输出哈希完全一致”。因此它属于独立的高性能实验轨道；稳定默认轨道继续由现有精确 wave32 BF16 SDPA 承担。

规范词：本文中的“必须（MUST）”“应该（SHOULD）”“可以（MAY）”分别表示强制需求、推荐需求和可选研究项。

### 1.1 落地计划改造结论

原计划的方向不变，但实施门禁改成以下四段，避免同时调试量化、VDN mask、QK/PV 两种 WMMA 和 H3 生命周期：

1. **R0 独立合同层**：在当前 research workspace 固化 geometry、interval task、workspace、RNE INT8 量化合同和 CPU 测试。
2. **R1 单一 MVP**：只实现 `sage-i8-bf16`，直接生成 gfx12 INT8 QK WMMA 与 BF16 PV WMMA；使用缩小但同构的 H3 17-frame case 与 production shape benchmark 验证。
3. **R2 H3 接入**：把 raw-pointer launch 封装到现有 `h3_gpu_tensor *` C ABI 后面，在 `h3_gpu` 中缓存 workspace 和 geometry metadata，完成同进程、同 Q/K/V 的精确 wave32 对比。
4. **R3 候选扩张**：只有 R2 的真实 50 层仍有净收益且误差达标，才投入 FP16/FP8 PV、smooth-K/V 和 tile heuristic 搜索。

截至 2026-09-07，R0/R1 research MVP 已落地，E27 已接入下游 `h3-vdn.c` 并通过
GPU5 production 50 层与完整 8-NFE/R2 验证；随后在真正干净窗口完成 GPU4 E27
promotion，GPU4 release baseline 已由 E21 升级为 E27。完整 512x512、56 帧 E2E
的视频、mux、同步和性能门禁通过，但纯 E27 与混合候选均未通过冻结的音频质量线。
`example_0/1/2` 三份真实 prompt embedding 已齐；结果分别为解码后 audio FAIL、
解码后 audio FAIL、audio latent fail-fast，即 **0/3**，因此不能把 Sage mode 标成
H3 stable。`example_1` 的精确 reference Cholesky 间歇故障已通过固定 pre-solve
stream boundary 修复，并由完整 8-NFE/E2E 复测确认。

## 2. 研究基线与引用边界

| 项目 | 固定版本/状态 | 对本实现的意义 |
|---|---|---|
| [SageAttention 主仓库](https://github.com/thu-ml/SageAttention) | `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` | API、算法和许可基线；官方稳定说明主要覆盖 CUDA GPU |
| [gfx12 原生 HIP PR #368](https://github.com/thu-ml/SageAttention/pull/368) | Open，head `66f5e64c9e36084c863a4480e570069245e58f90` | gfx1201 的 INT8-WMMA QK、FP8/FP16 PV、量化和 lane layout 参考 |
| [ROCm Triton PR #381](https://github.com/thu-ml/SageAttention/pull/381) | Open，head `6aa2622f0fdad0b3cbccbfc30d6f5954f8c29020` | ROCm 算法行为和性能参考，不作为 C++ 运行时依赖 |
| [SageAttention 论文](https://arxiv.org/abs/2410.02367) | arXiv | INT8 QK、平滑与量化依据 |
| [SageAttention2 论文](https://arxiv.org/abs/2411.10958) | arXiv | 更低精度和精度补偿策略依据 |
| [HIP FP8 文档](https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.5/reference/fp8_numbers.html) | AMD 官方文档 | gfx1200/gfx1201 使用 OCP FP8，不能假定 FNUZ |
| [Clang AMDGPU builtins](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html) | LLVM 官方文档 | gfx12 wave32 WMMA builtin 的权威接口定义 |
| [rocWMMA examples](https://github.com/ROCm/rocm-examples) | AMD 官方示例 | i8 GEMM、tile 和数据搬运的实现参考 |

必须注意：两个 ROCm 相关 PR 截至本文日期仍处于 Open 状态，不能视为上游稳定承诺。PR 中对 Wan 或标准 causal/non-causal attention 的加速数据不能直接外推到 VDN 自定义稀疏掩码。

如复用 PR #368 的代码而不是独立实现，必须保留 Apache-2.0 许可证、原始版权说明和变更说明，并完成依赖与来源审计。

## 3. 范围和非目标

### 3.1 第一阶段范围

- 原生 C++17/HIP 编译与调用。
- 首要优化架构为 `gfx1201`、wave32、head dimension 128。
- batch 固定为 1，但 heads 和 sequence 必须由运行时参数传入。
- 支持 VDN 当前 BF16 输入和 BF16 输出。
- 支持 VDN 当前 production 掩码，不退化为错误的 dense、causal 或普通 sliding window attention。
- 端到端计时必须包含 Q/K/V 预处理、量化、attention 和输出转换。
- 保留精确实现作为回退与质量参考。

### 3.2 第一阶段非目标

- 不引入 PyTorch、ATen、Triton、CUDA 或 NVCC 依赖。
- 不在第一阶段承诺所有 AMD 架构；非 gfx12 设备必须安全回退。
- 不修改 VDN 权重格式，不把 attention INT8/FP8 与模型权重量化混为一项工作。
- 不实现多 GPU 分片。
- 不把 SageAttention 默认开启。
- 不以外部仓库的标准 attention benchmark 代替 h3-vdn.c 的真实 50 层和完整 E2E 验收。

## 4. VDN 当前工作负载契约

### 4.1 Tensor 契约

相邻 `h3-vdn.c` 当前公开调用使用 `h3_gpu_tensor *`，必须保持不变：

```c
int h3_gpu_vdn_window_sdpa_bf16(
    h3_gpu *gpu,
    h3_gpu_tensor *output,
    const h3_gpu_tensor *query,
    const h3_gpu_tensor *key,
    const h3_gpu_tensor *value,
    uint32_t sequence,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t video_start,
    uint32_t frames,
    uint32_t tokens_per_frame,
    uint32_t radius,
    uint32_t chunk,
    int anchor_both,
    float scale);
```

raw pointer + `hipStream_t` 只属于第 6 节的内部 HIP 模块接口，不替换公共 C ABI。

输入和输出必须满足：

- Q/K/V 为连续 BF16；
- 逻辑布局为 `[sequence][heads][head_dim]`，即 NHD；
- Q/K 已完成 norm 和 RoPE；
- V 是 value projection 的输出；
- 输出布局、元素数量和输入一致；
- 首个 Sage 版本只接受 `head_dim == 128`，其他维度必须回退或明确报 unsupported。

### 4.2 当前 production shape

示例 0 的 production attention shape 为：

| 参数 | 值 |
|---|---:|
| `sequence` | 5338 |
| `heads` | 56 |
| `head_dim` | 128 |
| `video_start` | 986 |
| `frames` | 17 |
| `tokens_per_frame` | 256 |
| 视频 token 数 | 4352 |
| `video_end` | 5338 |
| `radius` | 1 |
| `chunk` | 5 |
| `anchor_both` | true |
| `scale` | `1 / sqrt(128)` |

实现不能硬编码 `sequence=5338` 或 `video_start=986`。现有 prompt 已出现 800、821 和 1299 等不同文本长度，因此 sequence 与 video 起点必须动态处理。

关键非对齐条件：

- `sequence=5338` 不是 64 的整数倍，向上 pad 为 5376 时有 38 个无效 row；
- `video_start=986` 对 64 取模为 26，对 16 取模为 10；
- 视频边界会切穿 Q/K tile；
- `tokens_per_frame=256` 对 64 对齐，但这不能消除视频起点的非对齐。

任何只在对齐 synthetic shape 上正确的实现均不合格。

### 4.3 VDN mask 的精确定义

设：

```text
video_end = video_start + frames * tokens_per_frame
query_frame(q) = (q - video_start) / tokens_per_frame
query_chunk(q) = query_frame(q) / chunk
```

允许访问规则必须与现有 scalar oracle 一致：

1. query 不在视频区间内时，可访问全部 key；
2. key 不在视频区间内时，所有 query 均可访问；
3. 当 `anchor_both=true` 时，第一帧和最后一帧中的 query 为 global query；
4. 内部视频 query 可以访问：
   - 全部非视频 key；
   - 第一帧 anchor；
   - query 所在 chunk 前后 `radius` 个 chunk 的完整窗口；
   - 最后一帧 anchor；
5. 相交或相邻区间必须先合并，不能重复累计 softmax。

production 的 17 帧、chunk 5、radius 1 可理解为：

| query 所在 chunk | 普通窗口 | 额外 anchor |
|---|---|---|
| 0，帧 0–4 | 帧 0–9 | 最后一帧 16 |
| 1，帧 5–9 | 帧 0–14 | 最后一帧 16 |
| 2，帧 10–14 | 帧 5–16 | 第一帧 0 |
| 3，帧 15–16 | 帧 10–16 | 第一帧 0；最后一帧 query 本身为 global |

通用实现还必须支持 `video_end < sequence`，即视频后方仍有非视频 token，不能根据当前样例中视频恰好位于尾部而省略 suffix。

## 5. 对外行为与模式选择

### 5.1 兼容性

- 必须保持 `h3_gpu_vdn_window_sdpa_bf16()` 的 C ABI 和现有调用点可用。
- 必须保留现有 `scalar` oracle 和 `wave32` 精确路径。
- Sage 路径是 approximate mode，必须由用户显式选择。
- `auto` 在稳定阶段仍应选择精确 wave32 路径，不得静默选择 Sage。

建议的模式值：

```text
auto
scalar
wave32
sage-i8-bf16
sage-i8-f16
sage-i8-fp8-e4m3
```

第一阶段可以通过 `H3_VDN_SDPA` 环境变量完成选择；稳定后应增加等价 CLI 参数，并在启动日志中打印最终 dispatch 结果、架构、量化模式和回退原因。

### 5.2 失败与回退

- 显式选择 Sage 且设备/shape 不支持时，必须返回清晰错误，不能悄悄运行别的算法。
- `auto` 模式可以在 unsupported 或 workspace 分配失败时回退精确 wave32，并记录一次原因。
- kernel launch、非法参数或运行时错误不得被当作可忽略回退条件。
- 禁止修改 Q/K/V 输入。
- 禁止每个 DiT block 重复 `hipMalloc`/`hipFree`。

## 6. 原生 C++/HIP 接口需求

### 6.1 建议文件边界

```text
sageattetion_workspace/
  include/h3_vdn_sage.hpp           # 可移植的内部 raw-pointer API
  src/h3_vdn_sage.cpp               # 参数、workspace 和 interval tasks
  src/h3_vdn_sage_gfx12.hip         # gfx12 quant/QK/softmax/PV kernels
  tests/test_contract.cpp            # CPU contract
  tests/test_gpu.cpp                 # GPU4 correctness 和 H3 定向 case
  tests/bench_h3_vdn.cpp             # production geometry benchmark

h3-vdn.c/                            # R2 下游接入
  h3_gpu_hip.cpp                     # 公共 C ABI dispatch/fallback
  h3_vdn_sage_*.{hpp,cpp,hip}        # 从 research workspace 提升的模块
  tests/test_vdn_sage.cpp            # 与现有 scalar/wave32 同进程比较
```

实际文件名可以调整，但算法内核不应继续无限扩张现有通用 HIP 文件。

### 6.2 内部 launch API

内部接口必须是 raw pointer + POD parameters + `hipStream_t`，不得暴露 Torch tensor。建议形态：

```cpp
enum class h3_vdn_sage_pv_mode : uint32_t {
    bf16,
    fp16,
    fp8_e4m3,
};

struct h3_vdn_sage_geometry {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t video_start;
    uint32_t frames;
    uint32_t tokens_per_frame;
    uint32_t radius;
    uint32_t chunk;
    bool anchor_both;
};

struct h3_vdn_sage_params {
    const void *query_bf16;
    const void *key_bf16;
    const void *value_bf16;
    void *output_bf16;
    h3_vdn_sage_geometry geometry;
    float scale;
    h3_vdn_sage_pv_mode pv_mode;
    void *workspace;
    size_t workspace_bytes;
    hipStream_t stream;
};

size_t h3_vdn_sage_workspace_size(const h3_vdn_sage_geometry &geometry,
                                  h3_vdn_sage_pv_mode mode);

hipError_t h3_vdn_sage_prepare_workspace(/* geometry + workspace + stream */);
hipError_t h3_vdn_sage_launch_prepared(const h3_vdn_sage_params &params);
hipError_t h3_vdn_sage_launch(const h3_vdn_sage_params &params);
```

具体要求：

- 所有 size 计算必须检查乘法和加法溢出；
- C++ 异常不能跨越 C ABI；
- `workspace_size()` 必须是纯计算，不隐式分配内存；
- 所有 kernel 在调用方 stream 上有序执行，不使用全局隐式同步；
- workspace 必须可由 `h3_gpu` context 复用；
- geometry 对应的 task metadata 应缓存并跨 50 层、8 NFE 复用，shape 改变时才重建；
- `h3_gpu` 需要保存 `gcnArchName`/架构能力，不应只保存 warp size。

FP8 tensor 第一阶段可以在内部使用 `uint8_t` packed storage；除非通用 tensor API 确实需要持有 FP8，否则不应为这一条内核过早扩张公共 dtype 枚举。

research API 将冷路径拆成 `prepare_workspace()`，缓存命中后调用 `launch_prepared()`；便利接口 `launch()` 会先 prepare 再 launch，不应在 H3 的 50 层热路径中使用。

## 7. 建议的数据流

```text
BF16 Q/K/V
   │
   ├─ Q：按 32-row group 做 signed INT8 量化 ─┐
   ├─ K：可选 smooth-K，按 64-row group 做 signed INT8 量化 ─┤
   └─ V：BF16 / FP16 / smooth-V + OCP FP8 E4M3 ───────────────┤
                                                               ▼
     geometry task list → I8 WMMA QK → VDN mask → online softmax → WMMA PV
                                                               │
                                                               ▼
                                                          BF16 output
```

Q/K/V 预处理每次 attention 调用各执行一次。禁止针对每个允许区间重复量化 K 或 V。

### 7.1 Geometry task list

不应创建 `sequence × sequence` dense mask。建议将一个 mask class 的 query row 拆成多个 Q task，每个 task 包含已排序、已合并的 key interval：

```cpp
struct h3_vdn_key_interval {
    uint32_t begin;  // inclusive
    uint32_t end;    // exclusive
};

struct h3_vdn_q_task {
    uint32_t q_begin;
    uint32_t q_count;
    uint32_t interval_count;
    h3_vdn_key_interval allowed[5];
};
```

一个 task 内的 query 必须共享完全相同的 mask。task 边界不得跨越：

- 非视频与视频 query 边界；
- 第一/最后 anchor frame query 边界；
- 不同 chunk mask class；
- padded row 边界。

允许区间至多由以下片段构成，并应在 host 端提前合并：视频前非视频区、第一帧、chunk window、最后一帧、视频后非视频区。

### 7.2 量化定义

首版应与参考实现对齐：

- Q：每 32 row、每 head、每 D group 计算对称 scale；
- K：每 64 row、每 head、每 D group 计算对称 scale；
- signed INT8 目标范围为 `[-127, 127]`；
- 使用 round-to-nearest-even，并做显式饱和；
- `amax == 0` 时输出全 0 且使用有限、非零 scale，禁止 NaN/Inf；
- padded Q/K 值写 0，但 padded key 必须同时被 mask 为 `-inf`，不能仅依赖数值为 0；
- scale 和 mean 使用 FP32；
- D128 必须完整覆盖 8 个 K=16 的 WMMA depth tile。

smooth-K 和 smooth-V 必须作为可独立开启的实验开关：

- smooth-K 的均值只统计真实 sequence，不含 padded row；
- smooth-V 的输出 correction 必须在 FP32 累计完成；
- 开启平滑前后必须分别记录误差和耗时，不能只凭上游默认值决定。

### 7.3 QK：INT8 WMMA

gfx12 wave32 核心必须使用等价于以下官方 builtin 的矩阵指令：

```text
__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12
```

需求：

- Q/K 为 signed INT8，builtin 的 signedness/flags 必须被单元测试覆盖；
- I32 累加完成后再乘 Q scale、K scale 和 `1/sqrt(D)`；
- 反汇编必须确认生成实际 `v_wmma_i32_16x16x16_iu8` 指令，而非编译器展开的 scalar dot；
- lane 到 fragment 的映射必须由固定测试向量验证，不能仅通过随机误差测试推断正确。

可以先用 rocWMMA 快速验证 tile 设计，再为关键路径封装 direct builtin；最终选择由生成 ISA、寄存器、LDS 和真实性能决定。

### 7.4 VDN mask 与边界 tile

- 完全不相交的 K tile 必须直接跳过，不加载 K/V；
- 部分相交的 tile 必须逐 element predicate；
- masked score 必须在 row max 之前写为负无穷或等价安全值；
- padded key 永远 masked；padded query 不得写越界输出；
- interval 之间 online-softmax state 必须正确合并；
- 必须有“被屏蔽 key 使用极大正值”的泄漏测试，确保 masked value 不影响 max、sum 或 output。

### 7.5 Online softmax

- row max `m` 和 row sum `l` 使用 FP32；
- 跨 tile 更新必须使用数值稳定的 online softmax 合并公式；
- 可以评估 `exp2` 与 `exp`，但每种实现均需单独经过误差门禁；
- 禁止把完整 attention probability 写回 HBM；
- task 的多个 interval 必须共享同一组 `m/l/output accumulator`；
- 每个合法 query 至少有一个允许 key；若参数导致空集合，必须在 host 校验阶段报错。

### 7.6 PV 候选路径

按以下顺序实现和评估：

1. `sage-i8-bf16`：softmax probability 转 BF16，V 保持 BF16，BF16 WMMA + FP32 accumulator；
2. `sage-i8-f16`：probability/V 使用 FP16，FP16 WMMA + FP32 accumulator；
3. `sage-i8-fp8-e4m3`：probability/V 使用 OCP FP8 E4M3，FP8 WMMA + FP32 accumulator。

对应 gfx12 官方 builtin 包括：

```text
__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
```

FP8 路径要求：

- gfx1201 使用 OCP E4M3，转换使用 `__HIP_E4M3`/`__HIP_SATFINITE` 的等价语义；
- 不得使用或假定 gfx12 支持 FNUZ 格式；
- scale、softmax offset、饱和范围和 correction 必须写入算法说明与测试；
- packed byte 顺序和 WMMA fragment lane layout 必须独立验证；
- FP8 不是预设赢家，若量化开销或误差不合格，BF16/FP16 PV 可以成为最终路径。

### 7.7 Tile 和调度搜索

必须至少评估以下候选空间，而不是把外部 PR 的 launch heuristic 原样硬编码：

- Q tile：16、32、64 row；
- K/V tile：64、128 row；
- 每 block wave 数量；
- LDS single buffer 与 double buffer；
- quant kernel 独立 launch 与适度融合；
- global query 和 window query 的独立 dispatch 参数；
- query task 的 persistent/static scheduling。

每个候选必须记录：kernel time、有效 TFLOP/s 或 TOPS、HBM bytes、VGPR/SGPR、LDS、occupancy、private segment/spill。出现 spill 的配置默认淘汰，除非真实 E2E 数据证明它仍然更快。

## 8. Workspace 与资源预算

以 production 示例为例：

```text
elements = 5338 * 56 * 128 = 38,262,784
单个 BF16 tensor  = 76,525,568 bytes ≈ 72.98 MiB
单个 I8/FP8 tensor = 38,262,784 bytes ≈ 36.49 MiB
```

pad 到 5376 row 后：

```text
单个 padded BF16 tensor  ≈ 73.50 MiB
单个 padded I8/FP8 tensor ≈ 36.75 MiB
QI8 + KI8 + VFP8          ≈ 110.25 MiB
```

此外还需 scale、mean、task metadata 和必要的 accumulator scratch。要求：

- workspace 必须按实际 mode 精确计算，不能无条件分配最大方案；
- 推荐初始总预算不超过 160 MiB；超出时必须说明原因和收益；
- workspace 在 `h3_gpu` 生命周期内复用；
- 禁止在 50 个 block 中各自常驻一份；
- 禁止使用进程级、无归属、非线程安全的 global pool；
- 同一 `h3_gpu` context 的 stream 顺序必须安全；若未来支持并发 stream，需要为每个并发调用提供独立 workspace。

量化 scale 的最低空间量级很小，例如示例中 Q 32-row group 和 K 64-row group 的 FP32 scale 分别约为 37,408 bytes 与 18,816 bytes；因此不能为了简化索引而创建巨大的 per-element scale。

## 9. 架构、编译和 ISA 要求

- 使用当前工程的 C++17 + `hipcc` 构建链。
- gfx12 代码必须有编译期和运行时双重 guard；其他 `HIP_ARCHS` 构建目标不能因 gfx12 builtin 无条件出现而编译失败。
- 首版禁止全局开启 fast-math。后续 fast-math 只能作为独立候选，并重新完成全部数值门禁。
- 允许使用 `/opt/rocm/include/rocwmma` 的 header-only 能力，不应新增不必要的动态链接依赖。
- build 和运行日志必须记录 ROCm、HIP、Clang 版本、GPU arch 与 git commit。
- 必须使用 `llvm-objdump` 或等价工具确认目标 code object 中存在期望的 WMMA ISA。
- 必须检查 kernel metadata，记录寄存器、LDS、scratch/private segment；关键 kernel 不允许未解释的 scratch spill。
- 对 gfx1200/gfx1201 的共用代码必须基于实际编译验证，不能只根据宏名称推断二进制兼容。

## 10. 正确性与质量门禁

### 10.1 CPU/标量合同测试

必须添加可快速运行的测试，覆盖：

- mask interval 构造与逐元素 scalar mask 完全一致；
- Q/K INT8 round-to-nearest-even、饱和、全零 group；
- FP8 E4M3 转换边界、NaN/Inf 策略与 packed layout；
- online softmax 跨多个不连续 interval 的合并；
- 非 16/32/64 对齐 sequence；
- `video_start` 切穿 tile；
- 有视频后缀 non-video tokens；
- `frames < chunk`、最后 chunk 不完整、radius 覆盖全部视频；
- `anchor_both` true/false；
- 非法参数与整数溢出。

### 10.2 GPU operator 测试

每一种 Sage mode 必须与现有 scalar BF16 oracle 比较，并报告：

- max absolute error；
- RMSE 与 relative RMSE；
- cosine similarity；
- 每 row 误差的 P50/P95/P99/max；
- NaN/Inf 数量；
- 至少两次运行的输出 hash，验证同一环境下确定性。

建议的研究期初始门槛：operator cosine similarity `>= 0.999`、relative RMSE `<= 5%`、无 NaN/Inf。正式门槛必须在查看完整 E2E 候选结果前冻结；不能为了让某个实现通过而事后放宽。

### 10.3 VDN 传播测试

通过单次 operator 不等于可用于扩散模型。必须依次完成：

1. 一个 DiT block 的输出对比；
2. 50 层、单 NFE 的逐层误差曲线；
3. 完整 8-NFE latent 对比；
4. VAE 后逐帧视频质量对比；
5. 音频输出质量和 A/V 同步检查；
6. 最终 mux 文件可播放性检查。

精确 wave32 reference 必须先独立通过相同的完整 8-NFE。VDN frame-statistics
producer 与 rocSOLVER Cholesky consumer 必须在同一 command group 下做连续压力回归；
reference 的求解失败不能归因给 Sage candidate，也不能用 diagonal jitter 隐藏。

最终 E2E 必须使用相同 prompt embeddings、seed、输入条件、scheduler 和编码参数。Sage 输出 hash 预计不同，因此质量门禁应使用 latent 误差、逐帧指标和人工检查，不应要求与 BF16 MP4 hash 相同。

至少使用 3 组代表性 prompt，覆盖不同 `video_start/sequence`，并保存：

- BF16 reference 输出；
- candidate 输出；
- 差分统计；
- 并排或交替帧检查材料；
- 音频频谱/误差和同步结果。

## 11. 性能门禁

当前已知精确 wave32 基线：

| 范围 | 基线 |
|---|---:|
| production shape standalone SDPA | 0.415342 s |
| 50 层真实 SDPA 累计 | 16.372 s / NFE |
| 单 NFE 总时间 | 约 31–33 s |

所有 Sage 性能必须把量化和平滑预处理计入端到端 operator time，不能只报告核心 QK/PV kernel。

建议研究门槛：

- standalone 含预处理时间 `<= 0.353 s`，即相对 0.415342 s 至少快 15%；
- 或 50 层真实 SDPA 累计 `<= 14.735 s`，即至少快 10%；
- 进入 stable candidate 前，两项都应满足，并证明完整 production E2E 有净收益；
- 若只在长 sequence synthetic case 快、真实 VDN mask 变慢，则不得合入默认可发布配置。

计时必须分段记录：

```text
task metadata（首次/缓存命中分开）
smooth-K / smooth-V
Q quant
K quant
V conversion/quant
QK + mask + softmax
PV
output conversion/correction
operator total
```

测试要求：

- GPU4 正式 promotion 时只设置物理 GPU 4 可见，并在日志中记录 PCI/BDF 或稳定
  设备标识；GPU5/6 研究点必须标明物理卡号、占用状态，且不得替代 GPU4 结论；
- 先 warm-up，再至少 5 次 measured run，报告 median、min、max；
- 同时记录时钟、温度、功耗、显存使用和是否 throttling；
- standalone、单 NFE、50 层和完整 8-NFE 必须使用相同 mode 名称与 build commit；
- 上游 PR 的 1.2×、1.4× 等数据只作方向参考，不属于本项目验收证据。

### 11.1 历史 GPU4 release candidate（E21）

2026-09-04 在物理 GPU4 独占窗口完成 E21 promotion。production shape 使用
167 个 32-row task、64-thread/2-wave workgroup，workspace 为 73.043 MiB；
attention kernel 静态资源为 208 VGPR、3328-byte group segment、0 scratch。
完整 GPU correctness、H3 定向 case、guard canary、determinism 与两类 WMMA ISA
门禁均通过。

| 指标 | GPU4 E03 旧基线 | GPU4 E21 release candidate | 变化 |
|---|---:|---:|---:|
| GPU event median | 17.117 ms | 15.598 ms | -8.87% |
| Profile total | 16.889 ms | 15.807 ms | -6.41% |
| Attention | 15.990 ms | 14.933 ms | -6.61% |
| 有效吞吐（按 profile total） | 41.921 TOPS | 44.790 TOPS | +6.84% |
| 混合峰值占比 | 16.447% | 17.573% | +1.126 pp |

E21 三轮 profile total 为 15.794/15.807/15.830 ms，GPU event median 为
15.637/15.598/15.586 ms，production hash 均为 `d8fccefb0ea98938`。因此 E21
曾取代 E03 成为 GPU4 baseline；当前已由 11.4 节的 E27 干净窗口结果取代。

### 11.2 GPU5 M2 research candidate（E27）

后续独立研究在物理 GPU5 上将 task metadata 改为 uniform 直接读取，并按 row
完成输出归一化和写回。E27 的 attention kernel 为 192 VGPR、38 SGPR、0 LDS、
0 scratch；三轮 profile total 为 14.738/14.832/14.870 ms，中位 **14.832 ms**，
GPU event 中位 **14.731 ms**，达到 standalone M2（<=15.000 ms）。对应有效吞吐
为 47.735 TOPS、混合峰值占比 18.728%。

E27 已通过 research workspace 的完整正确性、定向 H3 case、越界 canary、
determinism 和 ISA 门禁。早期 GPU4 窗口出现 20--26 ms 系统长尾而暂缓；后续
真正干净窗口的正式 promotion 见 11.4 节。该 standalone 结果本身不构成 R2，
真实 H3 证据见下一节。

### 11.3 R2 真实 H3 50 层结果

E27 已通过独立 gfx12 HIP 模块接入 `h3-vdn.c`，由显式
`H3_VDN_SDPA=sage-i8-bf16` 启用；`auto` 保持精确 wave32。`h3_gpu` context 缓存
QI8/KI8/scales workspace 和 geometry 对应的 32-row task metadata，production
geometry 只在首次 miss 上传，后续 50 层复用。

GPU5 上的 17 帧 x 256 token/frame 实际 layout 为 sequence=5158、video_start=806。
逐层双跑结果全部有限；最差 relative RMSE 为 1.8525%（第 22 层），对应 cosine
0.999828402；第 50 层为 0.7843% / 0.999969469。最终 video/audio 输出 relative
RMSE 为 0.5970%、cosine 0.999982179、invalid=0。

| 指标 | 精确 wave32 | E27 Sage | 变化 |
|---|---:|---:|---:|
| 50 层累计 SDPA | 14.865 s | 0.895 s | -93.98%，16.61x |
| 50 层 forward | 28.404 s | 14.622 s | -48.52%，1.943x |
| Peak live memory | 4.929 GiB | 4.998 GiB | +70.7 MiB |
| Attention dispatch count | 52 | 52 | profile 计数一致 |

这些结果完成 R2 的单 forward/50 层门禁；后续完整 8-NFE、VAE/mux 与多 seed
结果见 11.5 节。由于 audio 和多 prompt 仍未通过，mode 继续保持 explicit
experimental，不进入 `auto`。

### 11.4 当前 GPU4 release baseline（E27）

在连续四次 0% use/0% VRAM、26 C、14 W 且整机无其他 H3 GPU workload 的窗口，
E27 重新通过 correctness/guard/determinism/ISA 后完成三轮 10-iteration benchmark。
profile total 为 14.835/14.911/14.892 ms，中位 **14.892 ms**；GPU event median
为 15.201/15.121/15.107 ms，中位 **15.121 ms**；三轮 max 均 <=15.791 ms，hash
均为 `d8fccefb0ea98938`。相对 E21 的 profile/GPU median 改善 5.79%/3.06%，故 E27
正式成为 GPU4 release baseline。该 promotion 只证明算子性能与正确性，不覆盖
下述仍未通过的 E2E audio 与多 prompt stable 门禁。

### 11.5 M6 production E2E 与多 seed 结果（部分通过）

GPU5 上完成相同 `example_0` prompt、seed 0 的 wave32/E27 production 成对生成：
56 帧 512x512、24 fps，stereo 32 kHz/74400 samples。完整进程 wall 为
356.76/230.88 s，E27 降低 **35.29%（1.545x）**；其中 8-NFE DiT 为
246.230/119.637 s，降低 51.41%（2.058x）。两份 MP4 均可完整解码，A/V duration
差均为 8.333 ms。

视频通过冻结门槛：逐帧 PSNR mean/min 为 42.046/39.940 dB，SSIM mean/min 为
0.981421/0.975019，temporal delta cosine 为 0.993825。音频无 non-finite、无新增
clipping/silence 问题，但 cosine 0.981459、relative RMSE 19.232%、SI-SDR
14.186 dB，未通过冻结的 cosine >= 0.99、relative RMSE <= 10% 主门槛；同时也未
通过媒体门禁脚本较宽的 0.99/15%/15 dB 复核门槛。精确覆盖 non-video global
query 的混合候选把音频改善到 0.985745/16.874%/15.357 dB，仍未通过 cosine 和
relative RMSE，故相关候选已回退。

另以 seed 1/2 完成 production wave32/E27 成对 8-NFE latent：combined relative
RMSE 为 0.2136%/0.2114%，audio 分项为 5.2778%/5.0597%，全部无 non-finite；两组
E27 也分别完成完整 VAE/mux，wall 为 231.79/231.35 s。它们证明实现可稳定运行，
但不能覆盖 seed 0 的音频失败。

三份真实 prompt 已全部完成 staged gate：`example_0` 解码后 audio 为
0.981459/19.232%（correlation/relative RMSE）；`example_1` 在修复 reference
Cholesky 边界后完成两份 MP4，video PASS，但 audio 为
0.983433/18.1844%、SI-SDR 14.688 dB；`example_2` audio latent relative RMSE
5.3832%，按快速失败线不进入 VAE。M6 当前是有完整资产的质量 FAIL，而不是缺资产
BLOCKED；`auto` 保持 wave32。

## 12. Profiling 与可观测性

现有 `H3_HIP_PROFILE_SDPA` 类别应继续保留，并为 Sage 增加子阶段统计。至少提供：

- mode 与 dispatch reason；
- geometry cache hit/miss；
- workspace bytes 与峰值；
- Q/K/V quant bytes；
- 跳过的 full-masked K tile 数；
- partial boundary tile 数；
- 每个 kernel 的 HIP event time；
- 累计到每 NFE 和完整生成的时间。

默认非 profiling 模式不应为每层打印大量日志。推荐首行打印配置，结束时打印累计摘要；详细逐 kernel 数据由显式 profiling 开关开启。

## 13. 实现里程碑

### M0：合同与脚手架

- [x] 固化 mask CPU oracle 和 interval task builder。
- [x] 增加 runtime arch capability 与显式 mode dispatch。
- [x] 实现 workspace size/layout，完成 overflow 与边界测试。
- [x] 确保非 gfx12 build 和现有精确路径不回归。

### M1：量化原语

- [x] Q 32-row signed INT8 quant。
- [x] K 64-row signed INT8 quant。
- [ ] FP16/BF16/FP8 V 预处理候选。
- [x] CPU bit-level rounding 与 packed layout 测试。
- [x] GPU4 上记录量化时间与带宽。

### M2：I8 QK + VDN mask + online softmax

- [x] D128 的 gfx12 I8 WMMA QK。
- [x] global/window/anchor task 调度。
- [x] full tile skip 和 partial tile predicate。
- [x] 多 interval online-softmax 合并。
- [x] ISA、spill、occupancy 验证。

### M3：BF16/FP16 PV

- [x] BF16 PV WMMA + FP32 accumulator。
- [ ] FP16 PV 候选。
- [x] standalone correctness/performance 表。
- [x] 当前只保留 BF16 mode；未启用的 F16 名称返回明确错误。

### M4：FP8 PV

- [ ] OCP E4M3 conversion、scale 与 offset。
- [ ] FP8 WMMA + FP32 accumulator。
- [ ] smooth-V correction。
- [ ] 与 BF16/FP16 PV 做质量—性能 Pareto 对比。

### M5：真实模型传播

- [x] 一个 block。
- [x] 50 层单 NFE，逐层误差曲线。
- [x] frame-statistics/Cholesky 同 command-group 64 轮压力回归（GPU4/GPU6）。
- [ ] 完整模型连续多轮资源泄漏检查。
- [x] 保留最优 candidate，淘汰没有净收益的复杂分支。

### M6：完整 E2E

- [x] 512×512、56 帧、8-NFE + VAE + mux（GPU5/GPU6，seed 0/1/2）。
- [x] 同一真实 prompt 的 seed 0/1/2 压力合同。
- [x] 至少 3 组不同真实 prompt 合同（800/821/1299 tokens）。
- [x] 视频、同步和可播放性门禁。
- [ ] 音频质量门禁（三 prompt 为 0/3；详见 11.5）。
- [x] 完整 E2E 性能与 BF16 reference 对比。
- [x] 当前决定保持 explicit experimental，不进入 `auto`。

## 14. Definition of Done

只有同时满足以下条件，C++ SageAttention 路径才算实现完成：

- [ ] 原生 C++17/HIP，无 PyTorch/Triton/CUDA 运行时依赖。
- [ ] gfx1201 实际生成 I8 和目标 PV WMMA ISA。
- [ ] VDN 自定义 mask 与 scalar oracle 全部边界测试一致。
- [ ] 不创建 dense `S × S` attention/probability buffer。
- [ ] production 非对齐 shape 无 OOB、无 NaN/Inf、无资源泄漏。
- [ ] 显式 approximate opt-in；精确 wave32/scalar 路径仍可用。
- [ ] 量化在计时范围内，真实 50 层和完整 E2E 均有可复现净收益。
- [ ] 完成预先冻结的 operator、latent、视频和音频质量门禁。
- [ ] 只在物理 GPU 4 完成验收，并保留设备与环境证据。
- [ ] README、优化计划和测试命令同步更新。
- [ ] 若复用上游代码，许可证和版权归属完整。

## 15. 研究优先级与建议起步顺序

建议按以下顺序研究，尽早排除不适合 VDN 的方案：

1. 先完成 mask interval/task builder 和 CPU 合同测试；这是 VDN 与上游标准 SageAttention 最大的语义差异。
2. 用 D128 production shape 完成 Q/K INT8 quant + I8 WMMA QK，不急于加入 FP8 PV。
3. 首先尝试 BF16 PV，以较小的额外误差验证“INT8 QK 是否足以产生净收益”。
4. 再比较 FP16 PV，确认转换成本、寄存器和真实精度。
5. 只有当 PV 明确成为瓶颈时，再投入 OCP FP8 E4M3、softmax offset 和 smooth-V。
6. standalone 达标后立即进入真实 50 层；不要在 synthetic benchmark 上长期微调。
7. 任何候选一旦在 50 层误差或真实耗时上失败，记录结果后停止扩大实现复杂度。

## 16. 尚需实验回答的问题

- VDN 的非连续 key interval 会让多大比例的 16×16/64×64 tile 成为 partial tile？
- task list 应以 mask class、query frame还是固定 Q tile 为调度单位？
- K/V 的全 sequence 预量化能否被计算完全摊销？
- smooth-K 对 VDN 已做 norm+RoPE 的 K 是否仍有正收益？
- BF16 PV 是否已经足够快，避免 FP8 引入额外误差和转换开销？
- FP8 PV 的最佳 scale/offset 是否与上游标准 dense attention 相同？
- global text/audio query 与 window video query 是否需要两套 tile heuristic？
- 是否应把 quant 与 QK、V conversion 与 PV 融合，还是独立 kernel 更利于 occupancy？
- production 的 `video_start % 64 != 0` 对 LDS copy 和 partial tile 成本有多大影响？
- Sage 路径对 8 NFE 的误差是否稳定累积，还是在个别 layer/step 突增？

这些问题必须用 h3-vdn.c 的真实 workload 回答，不能只从上游实现推断。

## 17. 第一轮实验建议

第一轮最小可行实验只实现 `sage-i8-bf16`：

```text
shape: S=5338, H=56, D=128
mask: video_start=986, frames=17, tokens/frame=256, radius=1, chunk=5,
      anchor_both=true
Q: signed INT8, 32-row scale
K: signed INT8, 64-row scale
PV: BF16 WMMA, FP32 accumulator
output: BF16
device: physical GPU 4 only
```

第一轮必须产出一张统一结果表：

| Candidate | Q/K quant ms | QK+softmax ms | PV ms | Total ms | Cosine | RelRMSE | WMMA ISA | Spill |
|---|---:|---:|---:|---:|---:|---:|---|---|
| exact wave32 baseline | n/a | 待测 | 待测 | 415.342 | 1.0 | 0 | n/a | 待查 |
| sage-i8-bf16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 必须 | 禁止 |

只有这条路径达到正确性门槛并接近或超过 standalone 性能目标，才继续投入 FP16/FP8 PV。这样可以把“自定义 mask + INT8 QK”的核心不确定性与“FP8 PV”的额外复杂度分开验证。

### 17.1 当前 research MVP 结果

后续性能试验、污染样本、峰值效率口径和接受/回退记录统一维护在
[`h3-vdn-sageattention-optimization.md`](h3-vdn-sageattention-optimization.md)；
本节只回填已经复现并接受的稳定结论。

环境：物理 GPU4，Radeon AI PRO R9700，`gfx1201`，wave32，ROCm/HIP 7.2.3；命令通过 `ROCR_VISIBLE_DEVICES=4 HIP_VISIBLE_DEVICES=0` 隔离设备。

| Candidate | Q quant ms | K quant ms | QK+softmax+BF16 PV ms | Total ms | Cosine | RelRMSE | WMMA ISA | Spill |
|---|---:|---:|---:|---:|---:|---:|---|---|
| exact wave32 baseline（既有记录） | n/a | n/a | 待同进程复测 | 415.342 | 1.0 | 0 | n/a | 待查 |
| sage-i8-bf16 research MVP | 0.469 | 0.398 | 16.024 | 16.889 | 0.999970562 | 0.7678% | I8 QK + BF16 PV 均确认 | 0-byte private segment |

补充结果：

- production geometry：`S=5338,H=56,D=128,video_start=986`，334 个 Q task，workspace 73.051 MiB，metadata cold path 0.027–0.028 ms；GPU4 独占时三轮 5-iteration run 的 median 为 17.080–17.143 ms，观测 min 16.967 ms、max 17.857 ms。按有效 VDN 工作量计算为 41.921 TOPS，即 INT8/BF16 混合 matrix 理论峰值的 16.447%。
- cosine/RelRMSE 来自 `S=83` 非对齐、带 suffix 的 GPU-vs-CPU case；同构 17-frame H3 case 的 cosine 为 0.999996639、RelRMSE 为 0.2593%。
- `S=83` case 的 RMSE 为 0.0000592265；row RMSE P50/P95/P99/max 为 0.0000545288/0.0000879817/0.000110548/0.000110711。
- 均匀 score 的 masked-mean case max absolute error 为 0.000136711；masked K/V 泄漏与重复运行确定性 case 通过。
- attention kernel metadata：wave32，driver-dependent report 215–216 VGPR、59 SGPR、1664-byte group segment、0-byte private segment；无 scratch spill。

这些数据证明 R1 值得进入 H3 接入，但尚未证明 50 层和 8-NFE 的质量或真实 E2E 加速。尤其是 standalone 的 synthetic Q/K/V 与既有 wave32 基线不是同一进程内的成对计时，R2 必须重新测量，不能直接把 standalone 比值宣称为 H3 加速。
