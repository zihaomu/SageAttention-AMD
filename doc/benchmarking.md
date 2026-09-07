# Benchmark protocol

The goal is reproducible evidence across AMD platforms and kernel shapes, not a
single leaderboard number.

## Before measurement

1. Use a committed source revision and record whether the tree is dirty.
2. Identify the platform, kernel, and workload registry entries.
3. Inspect GPU utilization and memory use. Select a physical device explicitly.
4. Confirm clocks, power mode, thermals, and competing processes are stable
   when the available tooling exposes them.
5. Run correctness, boundary, guard-canary, determinism, and ISA checks.

## Measurement

- Warm up before recording samples.
- Use GPU events for operator timing and synchronized wall time for end-to-end
  timing; never compare the two as if they were the same scope.
- Record median, minimum, maximum, iteration count, and important sub-times.
- State whether quantization, planning, metadata upload, allocation, or tensor
  conversion is included.
- Keep exact and approximate numerical modes separate.
- Re-run a baseline in the same session for a speedup claim whenever possible.

Current benchmark recipes require `H3_PHYSICAL_GPU` and isolate the selected
HSA agent. The numeric device index belongs in the result session metadata, not
in a stable platform ID.

## Cross-platform reporting

Absolute latency from different platforms may be reported side by side when
each environment is complete. A ratio across different GPUs, ROCm versions,
clocks, or workload definitions must be labelled as a platform comparison, not
a kernel speedup.

## Evidence levels

- `research`: useful directional data; may use fewer repetitions or an
  experimental source state.
- `candidate`: clean commit and full operator gates, awaiting repeatability or
  downstream validation.
- `promotion`: clean commit, idle device, repeatable measurement, full operator
  gates, and all required downstream quality gates.

An operator may have promotion-grade performance evidence while its kernel
remains experimental because a consumer quality gate failed. Record both facts.

## Result storage

Store one JSON record per session under `benchmarks/results/`. Result records
are append-only and validated by `make metadata-check`. If a record is wrong,
add a replacement with `supersedes` and explain the correction.
