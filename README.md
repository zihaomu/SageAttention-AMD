# SageAttention AMD

Native C++17/HIP SageAttention kernels and sparse-attention planning for AMD
GPUs. The project has no PyTorch, ATen, Triton, CUDA, or Python runtime
dependency.

> **Project status:** experimental. Version `0.1.0-experimental` ships one
> production-shaped specialization for the H3 VDN interval-mask workload on
> AMD gfx12. The repository is intended to grow into a model-independent AMD
> SageAttention library without weakening its specialized kernels.

## What is implemented

The current E27 specialization supports:

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

## Why the first specialization mentions H3 VDN

The repository name describes the intended library scope. The first verified
consumer is H3 VDN, whose attention mask is not plain dense or causal
attention. Keeping that workload as the initial specialization preserves the
actual benchmark and quality evidence instead of claiming unsupported
generality.

The implementation is already independent of H3 tensor types: the public API
accepts raw device pointers, a HIP stream, geometry, and caller-owned
workspace. A future generic interval-plan API will separate model-specific
planning from the kernel dispatcher; the current VDN planner will remain one
adapter/profile. See [Generalization roadmap](#generalization-roadmap).

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

## API lifecycle

The experimental v0.1 API is declared in
[`include/h3_vdn_sage.hpp`](include/h3_vdn_sage.hpp). Its `h3_vdn_` prefix
identifies the first mask planner, not a dependency on the H3 runtime.

1. Call `h3_vdn_sage_workspace_size()` for a geometry and mode.
2. Allocate the returned number of device bytes in the caller context.
3. On a geometry cache miss, call `h3_vdn_sage_prepare_workspace()`.
4. Reuse the prepared metadata with `h3_vdn_sage_launch_prepared()`.
5. Use `h3_vdn_sage_launch_profiled()` only for synchronous diagnostics.

`h3_vdn_sage_launch()` is a cold-path convenience function that prepares task
metadata for every call. It should not be used in a multi-layer hot loop.

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
src/       workspace planner, quantization, and gfx12 E27 kernel
tests/     CPU contracts, GPU correctness/canaries, benchmark
doc/       requirements, optimization ledger, decisions, H3 integration
```

Generated objects, profiler databases, and media outputs are intentionally
excluded from version control.

## Generalization roadmap

The public project can become model-independent without turning the E27 hot
kernel into a branch-heavy generic kernel:

1. Introduce a generic interval-task plan API independent of VDN geometry.
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

## Documentation

- [Acceleration decisions](doc/h3-vdn-sageattention-acceleration-summary.md)
- [Optimization ledger](doc/h3-vdn-sageattention-optimization.md)
- [Requirements and gates](doc/vdn-sageattention-amd-cpp-requirements.md)
- [H3 VDN integration](doc/h3-vdn-integration.md)

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
