# Roadmap

This roadmap states direction, not current support. The registry and validated
benchmark records are the source of truth.

## Foundation

- Stabilize platform, kernel, workload, and benchmark metadata contracts.
- Extract a generic interval-task plan API from the first H3 VDN adapter.
- Add an explicit dispatcher keyed by architecture, wave size, layout, data
  type, head dimension, mask-plan class, and numerical mode.
- Keep experimental and supported status visible in API and registry data.

## Shape coverage

- Add dedicated evidence for common head dimensions such as 64, 128, and 256.
- Cover dense, causal, sliding-window, and caller-provided interval/block plans.
- Extend layout and batch coverage without adding conversion work to hot paths.
- Evaluate BF16, FP16, INT8, and newer low-precision modes separately.

## AMD architecture coverage

- Keep the current gfx12 wave32 path as its own specialization.
- Add gfx11/RDNA and CDNA-family backends only after ISA and numerical audits.
- Maintain per-platform ROCm compatibility records instead of one global
  “AMD supported” label.
- Automate compile checks where public runners are available and retain
  hardware-in-the-loop promotion for performance and correctness.

## Distribution and integration

- Add CMake package configuration and semantic API/version policy.
- Provide stable planner/dispatcher interfaces for downstream consumers.
- Publish reproducible benchmark summaries generated from registry data.
- Add consumer-owned model-quality gates without embedding model policy in the
  kernel library.
