# Architecture

SageAttention AMD separates portable policy from architecture- and
shape-specific execution. This lets the project become general without
flattening fast kernels into one heavily branched implementation.

## Layers

1. **Consumer adapter** converts model semantics into a supported plan. H3 VDN
   is the first adapter, not the library boundary.
2. **Plan representation** describes dense, causal, windowed, block, or ordered
   interval work independently of model tensor types.
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

## Numerical policy

Quantization scope, rounding, clamping, accumulator width, softmax precision,
and fully masked-row behavior are part of the kernel identity. Changing one may
produce a new numerical mode or specialization and requires new correctness and
downstream quality evidence.

## Ownership boundaries

The library owns raw-pointer/stream/workspace contracts, planning primitives,
dispatch, kernels, and operator tests. Consumers own tensor-framework adapters,
context lifetime, fallback/default policy, and model-level quality gates.
