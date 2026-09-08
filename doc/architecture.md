# Architecture

SageAttention AMD separates portable policy from architecture- and
shape-specific execution. This lets the project become general without
flattening fast kernels into one heavily branched implementation.

## Layers

1. **Consumer adapter** converts model semantics into a supported plan. H3 VDN
   is the first adapter, not the library boundary.
2. **Plan representation** describes ordered key intervals for query-row tiles
   independently of model tensor types. Dense, causal, and sliding-window
   planners can all produce this representation; a block-plan representation
   remains future work.
3. **Dispatcher** selects a specialization from explicit runtime properties:
   GPU architecture, wave size, layout, data types, head dimension, mask-plan
   class, and numerical mode.
4. **Kernel specialization** owns tile geometry, quantization, softmax, and PV
   implementation for a narrow dispatch domain.
5. **Evidence registry** links a platform, kernel, workload, source commit,
   correctness outcome, and measurements.

```text
consumer semantics
       |
       v
adapter/planner --> generic plan --> dispatcher
                                      |   |   |
                                      v   v   v
                                  specialized kernels
                                      |
                                      v
                         correctness + benchmark evidence
```

## Compatibility model

Support is a tuple, not a single GPU label:

```text
(architecture, wave size, ROCm, layout, dtype, batch,
 head dimension, mask plan, numerical mode)
```

Each kernel registry entry declares a subset of this space. The dispatcher
must fail clearly outside it. A new GPU or ROCm version requires validation,
even when it shares the same GFX architecture.

## Current interval-plan contract

The v0.2 development API represents a plan as canonical `q_task` records. Each
task covers consecutive query rows with one shared set of ordered,
non-overlapping key intervals. The current E27 dispatcher accepts at most 32
query rows and five intervals per task. H3 VDN is now a planner/adapter that
produces these records; the device kernel has no H3 geometry dependency.

## Numerical policy

Quantization scope, rounding, clamping, accumulator width, softmax precision,
and fully masked-row behavior are part of the kernel identity. Changing one may
produce a new numerical mode or specialization and requires new correctness and
downstream quality evidence.

## Ownership boundaries

The library owns raw-pointer/stream/workspace contracts, planning primitives,
dispatch, kernels, and operator tests. Consumers own tensor-framework adapters,
context lifetime, fallback/default policy, and model-level quality gates.
