# Benchmark-evidence agent instructions

Benchmark records are evidence, not hand-edited performance documentation.

- Never invent, extrapolate, average across machines, or copy a number from a
  prose document into a result file unless its original command and commit are
  known.
- Add one JSON file per measurement session. Existing records are immutable;
  add a superseding record when correcting metadata.
- Promotion evidence requires a clean source commit, an explicitly selected
  idle GPU, warm-up iterations, repeated measurements, and passing correctness.
- Store operator and end-to-end measurements as different scopes. Include
  quantization or metadata time only when the metric says that it does.
- Use stable platform, kernel, and workload IDs from `registry/`; do not encode
  a hostname or a transient HIP device index into those IDs.
- Run `make metadata-check` after adding or changing a record.
