# Device-code agent instructions

These rules apply to `src/` in addition to the repository instructions.

- Preserve the public API contract in `include/`; do not silently change tensor
  layout, ownership, synchronization, workspace, or stream semantics.
- Keep architecture dispatch outside hot loops. Add a separate specialization
  when architecture, wave size, head dimension, tile geometry, or numerical
  method changes materially.
- Reject unsupported inputs explicitly. Do not add an unmeasured fallback and
  describe it as accelerated support.
- Preserve FP32 online-softmax state and guard all fully masked, partial-tile,
  extreme-score, and zero-work paths.
- Quantization changes must document scale scope, rounding, clamping, zero
  handling, and accumulator range.
- Never tune from wall time alone. Inspect generated ISA and report occupancy,
  resource pressure, launch count, and operator sub-times where available.
- A device-code change is incomplete until targeted GPU correctness, guard
  canaries, determinism, `make isa`, and the affected production workload pass.
- Record failed or neutral optimization experiments in the optimization ledger
  before removing the experimental code.
