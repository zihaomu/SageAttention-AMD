# SageAttention AMD

[![Metadata](https://github.com/zihaomu/SageAttention-AMD/actions/workflows/metadata.yml/badge.svg)](https://github.com/zihaomu/SageAttention-AMD/actions/workflows/metadata.yml)

Native C++17/HIP SageAttention kernels and sparse-attention planning for AMD
GPUs. The project has no PyTorch, ATen, Triton, CUDA, or Python runtime
dependency.

> **Project status:** experimental. The latest release is
> [`v0.1.0-experimental`](https://github.com/zihaomu/SageAttention-AMD/releases/tag/v0.1.0-experimental).
> `main` is `0.2.0-dev` and adds a model-independent interval-plan API while
> retaining the first H3 VDN specialization and compatibility API.

## What is implemented

The current E27 specialization supports:

- caller-provided ordered-interval plans independent of model geometry;
- BF16 NHD input/output, batch 1, dynamic sequence/head count, `D=128`;
- symmetric signed INT8 Q/K quantization with 32-row Q and 64-row K groups;
- gfx12 wave32 `v_wmma_i32_16x16x16_iu8` QK;
- sparse attention represented as ordered key intervals per Q super-tile;
- the H3 VDN `window + chunk + bidirectional anchors + global tokens` planner;
- streaming FP32 online softmax without an `S x S` score or mask buffer;
- gfx12 BF16 WMMA PV with FP32 accumulation;
- caller-owned HIP stream and reusable workspace;
- CPU mask/quantization contracts, GPU correctness, OOB canaries,
  determinism, ISA checks, and a production-shape benchmark.

FP16 and FP8 E4M3 enum values are reserved but return
`hipErrorNotSupported`. Non-gfx12 devices, non-wave32 execution, and head
dimensions other than 128 are rejected explicitly.

## Generic plan and the H3 VDN adapter

The repository name describes the intended library scope. The first verified
consumer is H3 VDN, whose attention mask is not plain dense or causal
attention. Keeping that workload as the initial specialization preserves the
actual benchmark and quality evidence instead of claiming unsupported
generality.

The generic API accepts raw device pointers, a HIP stream, an operation
descriptor, caller-owned workspace, and an ordered-interval plan. H3 geometry
is converted into that plan by the compatibility adapter. The E27 hot kernel
remains specialized and receives the same task layout as the v0.1 release.

## Capability and evidence model

The project tracks support as an explicit tuple of GPU architecture, wave size,
ROCm/toolchain, layout, dtype, batch, head dimension, mask plan, and numerical
mode. A measurement on one tuple is never treated as evidence for another.

- [`registry/platforms/`](registry/platforms/) records validated machine
  classes and software stacks.
- [`registry/kernels/`](registry/kernels/) records the exact dispatch domain
  and numerical contract of every specialization.
- [`registry/workloads/`](registry/workloads/) records stable benchmark shapes
  and mask semantics.
- [`benchmarks/results/`](benchmarks/results/) joins those IDs to an immutable
  measured session and source commit.

Run `make metadata-check` to validate references and compatibility. The current
validated tuple is R9700/gfx1201 wave32 on ROCm 7.2.3 with the experimental E27
ordered-interval `D=128` specialization and H3 VDN workload. Registry presence
does not by itself mean a kernel is a stable default; status and downstream
quality are recorded separately.

## Build

Validated environment:

- ROCm 7.2.3;
- AMD Radeon AI PRO R9700;
- `gfx1201`, wave32;
- x86_64 Linux.

Build the library, tests, and benchmark:

```sh
make -j
```

The output library is:

```text
build/libsageattention_amd.a
```

Run host-side contracts:

```sh
make contract-test
```

GPU commands intentionally require an explicit physical device index:

```sh
H3_PHYSICAL_GPU=4 make gpu-test
make isa
H3_PHYSICAL_GPU=4 make bench
```

The recipes expose only the selected HSA agent with `ROCR_VISIBLE_DEVICES`,
then address it as HIP device 0. Use a confirmed idle card for performance
measurements.

### Machine onboarding and optimization campaigns

The dependency-free development CLI implements the first multi-architecture
onboarding loop. Generated plans, platform drafts, research results, and
campaign reports stay below ignored `build/sagectl/`:

```sh
make tool-test metadata-check
./tools/sagectl doctor --gpu 4 --require-idle --force
./tools/sagectl plan --workload ordered-interval-dense-s35-h1-d128
./tools/sagectl baseline --gpu 4 \
  --platform r9700-gfx1201-rocm7.2.3-ubuntu24.04 \
  --workload ordered-interval-dense-s35-h1-d128
./tools/sagectl search --gpu 4 \
  --campaign gfx1201-ordered-interval-d128-smoke
./tools/sagectl report \
  --campaign gfx1201-ordered-interval-d128-smoke
```

Replace physical GPU `4` with a device that has been checked as idle. The
`baseline` command also accepts a doctor-generated platform record through
`--platform-draft build/sagectl/platforms/<platform-id>.json`; on a new GFX
target it emits a matching exact-kernel draft for review. Architecture objects
are isolated under `build/sagectl/architectures/<gfx>/`.

The current registered campaign validates comparisons between existing kernels;
compile-time candidate generation remains future work. See the
[multi-architecture optimization playbook](doc/multi-architecture-optimization-playbook.md)
for evidence and promotion boundaries.

## API lifecycle

New consumers should use the experimental generic API in
[`include/sage_attention.hpp`](include/sage_attention.hpp):

1. Construct a `descriptor` and canonical `interval_plan`.
2. Call `validate_interval_plan()` and `query_support()`.
3. Call `workspace_size()` and allocate caller-owned device workspace.
4. On a plan cache miss, call `prepare_workspace()`.
5. Reuse that metadata with `launch_prepared()`.
6. Use `launch_profiled()` only for synchronous diagnostics.

The detailed contract is in
[the generic interval-plan API guide](doc/generic-interval-plan-api.md).

The source-compatible v0.1 H3 adapter remains in
[`include/h3_vdn_sage.hpp`](include/h3_vdn_sage.hpp). Existing callers retain
the same lifecycle:

1. Call `h3_vdn_sage_workspace_size()` for a geometry and mode.
2. Allocate the returned number of device bytes in the caller context.
3. On a geometry cache miss, call `h3_vdn_sage_prepare_workspace()`.
4. Reuse the prepared metadata with `h3_vdn_sage_launch_prepared()`.
5. Use `h3_vdn_sage_launch_profiled()` only for synchronous diagnostics.

`h3_vdn_sage_launch()` remains a cold-path convenience function. It builds a
generic interval plan and prepares metadata for every call, so it should not be
used in a multi-layer hot loop.

All size computations are overflow checked. Explicit dispatch surfaces an
unsupported architecture, mode, shape, or insufficient workspace rather than
silently choosing a different algorithm.

## Performance evidence

The production benchmark geometry is:

```text
S=5338, H=56, D=128
video_start=986, frames=17, tokens_per_frame=256
radius=1, chunk=5, anchor_both=true
```

All E27 operator measurements include Q/K quantization. The clean physical
GPU4 promotion result was:

| Metric | E27 result |
|---|---:|
| Profile total | **14.892 ms** |
| GPU event median | **15.121 ms** |
| Effective throughput | about **47.6 TOPS** |
| Effective mixed-peak utilization | about **18.7%** |

In a clean GPU6 comparison, exact wave32 measured 414.786 ms and E27 measured
15.419 ms, a **26.9x** operator speedup. In the H3 integration, 50-layer SDPA
fell from 14.865 s to 0.895 s (**16.61x**) and the complete 50-layer forward
fell from 28.404 s to 14.622 s (**1.943x**).

These measurements are workload-specific. Cross-device absolute values are
not mixed into one baseline; GPU4 is the formal promotion device, while GPU5
and GPU6 results are research or stability evidence.

## Quality status

E27 is fast and passes operator correctness, real 50-layer propagation,
8-NFE execution, decoded-video, VAE, mux, and synchronization checks. It is
**not a stable model default** because the frozen H3 audio gate passed 0 of 3
real prompts:

| Prompt | Performance | Video | Audio result |
|---|---:|---|---|
| `example_0` | DiT **2.058x** | PASS | correlation 0.981459, RelRMSE 19.232% — FAIL |
| `example_1` | DiT **2.078x** | PASS | correlation 0.983433, RelRMSE 18.1844% — FAIL |
| `example_2` | 8-NFE **2.266x** | latent PASS | audio latent RelRMSE 5.3832% — FAIL-fast |

The H3 integration therefore keeps exact BF16 wave32 as `auto` and exposes
E27 only through explicit `H3_VDN_SDPA=sage-i8-bf16` selection. Operator speed
must not be presented as model-level acceptance.

## Repository layout

```text
include/   experimental public C++/HIP API
src/       generic dispatch/workspace, H3 planner, and gfx12 E27 kernel
tests/     CPU contracts, GPU correctness/canaries, benchmark
doc/       requirements, optimization ledger, decisions, H3 integration
registry/  platform, kernel, and workload capability records
benchmarks/append-only structured performance evidence
tools/     development-time metadata validation
```

Generated objects, profiler databases, and media outputs are intentionally
excluded from version control.

## Generalization roadmap

The public project can become model-independent without turning the E27 hot
kernel into a branch-heavy generic kernel:

1. Stabilize the new generic interval-task plan API independent of VDN geometry.
2. Keep the current H3 VDN geometry-to-interval conversion as an integration
   adapter and reference sparse-mask planner.
3. Dispatch to specialized kernels by architecture, head dimension, input
   layout, quantization mode, and PV mode.
4. Add dense, causal, sliding-window, and caller-provided block/interval plans.
5. Add new head dimensions and AMD architectures only with dedicated
   correctness, ISA, and performance evidence.
6. Keep model-level quality policy in each consumer repository.

Near-term work should focus on the generic plan boundary and new validated
specializations. Previously rejected producer/consumer, softmax batching,
layout packing, and runtime exact/approximate hybrid experiments are recorded
in the optimization ledger and should not be repeated without new evidence.
The longer-term sequence is maintained in [`ROADMAP.md`](ROADMAP.md).

## Documentation

- [Acceleration decisions](doc/h3-vdn-sageattention-acceleration-summary.md)
- [Optimization ledger](doc/h3-vdn-sageattention-optimization.md)
- [Requirements and gates](doc/vdn-sageattention-amd-cpp-requirements.md)
- [H3 VDN integration](doc/h3-vdn-integration.md)
- [Architecture and dispatch boundaries](doc/architecture.md)
- [Benchmark protocol](doc/benchmarking.md)
- [Multi-architecture optimization playbook](doc/multi-architecture-optimization-playbook.md)
- [Support and compatibility policy](doc/support-policy.md)
- [Generic interval-plan API](doc/generic-interval-plan-api.md)
- [Changelog](CHANGELOG.md)

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for adding a platform, kernel shape,
workload, or benchmark result. Issues include dedicated templates for kernel
requests and reproducible benchmark contributions. Repository-level and scoped
[`AGENTS.md`](AGENTS.md) files give coding agents the same evidence, GPU-safety,
and promotion rules used by human contributors.

## Upstream provenance

The gfx12 WMMA fragment-packing approach was informed by the Apache-2.0 work
in [`thu-ml/SageAttention` PR #368](https://github.com/thu-ml/SageAttention/pull/368)
at commit `66f5e64c9e36084c863a4480e570069245e58f90`.

This implementation removes Torch/ATen integration, uses raw HIP pointers and
caller-owned workspace/stream, adds interval-mask planning, and implements the
BF16 PV path. Attribution is retained in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
