# Changelog

## Unreleased — 0.2.0-dev

- Add the model-independent `sageattention` C++ interval-plan API.
- Add explicit shape/options/kernel descriptors and current-device support
  queries.
- Route the source-compatible H3 VDN API through the generic E27 dispatcher.
- Add CPU plan-contract tests, arbitrary interval GPU correctness, and
  bit-for-bit generic/H3 compatibility checks.
- Add `sagectl` machine probes, registry-driven interval plans and benchmarks,
  an exact portable development baseline, and research-result drafts.
- Add validated optimization campaigns with static gates, short-run ranking,
  paired confirmation, and local reports.
- Resolve E27 host and device capability checks through an internal
  specialization registration table without changing the E27 or H3 APIs.

## 0.1.0-experimental — 2026-09-08

- Publish the first gfx1201/wave32 E27 specialization.
- Add H3 VDN mask planning, reusable workspace preparation, and profiled launch
  APIs.
- Add CPU/GPU correctness, guard-canary, determinism, ISA, and production-shape
  benchmark coverage.
- Establish platform, kernel, workload, and append-only benchmark registries.
