# H3 VDN integration

H3 VDN is the first integration consumer of SageAttention AMD. The standalone
repository owns the interval planner, workspace contract, quantization, gfx12
kernel, operator tests, and benchmark. `h3-vdn.c` owns model tensors, context
lifetime, dispatch policy, profiling, 50-layer propagation, and decoded-media
quality gates.

## Current status

The E27 specialization is integrated in the `h3-vdn.c` branch
`vdn-h3-rocm`. It is selected explicitly with:

```sh
H3_VDN_SDPA=sage-i8-bf16
```

Unset/`auto` remains exact BF16 wave32 because E27 has not passed the H3 audio
quality gate. The integration must not silently select Sage or silently fall
back when an explicit Sage request is unsupported.

## Adapter boundary

The standalone library accepts raw device pointers and a caller-owned HIP
stream/workspace. The H3 adapter is responsible for:

1. validating BF16 NHD tensors and non-overlapping output/value storage;
2. converting the H3 geometry and `anchor_both` representation;
3. checking gfx12, wave32, and `D=128` before explicit dispatch;
4. allocating and reusing workspace in the owning `h3_gpu` context;
5. preparing immutable task metadata only on geometry cache misses;
6. surfacing unsupported mode, architecture, shape, and allocation failures;
7. collecting per-NFE operator and end-to-end profile data.

The standalone project must not depend on `h3_gpu_tensor`, model weights,
environment-variable parsing, VAE/mux code, or rocSOLVER state.

## Context lifetime

H3 should retain context-owned state equivalent to:

```text
architecture and wave size
Sage workspace pointer and allocated bytes
prepared geometry/mode key
metadata-ready flag
```

The mutable Q/K quantization buffers belong to an in-flight attention call.
The current H3 execution is serial on one stream, so one context workspace can
be reused. If concurrent streams are introduced, each in-flight call needs
independent mutable workspace; immutable task metadata may be shared only with
explicit lifetime guarantees.

## Update workflow

Future integration updates should use one pinned SageAttention-AMD commit or
release rather than manually editing two copies:

1. develop and validate a candidate in this repository;
2. pass CPU contract, GPU correctness, canary, determinism, ISA, and
   production-shape benchmark gates;
3. tag or record the exact candidate commit;
4. update the vendored source or subtree in `h3-vdn.c` in one isolated change;
5. run same-QKV wave32/Sage comparison, one NFE, 50 layers, 8 NFE, and the
   three-prompt decoded-media gate;
6. update the downstream pin only after recording the complete result.

Source vendoring or `git subtree` is preferred over a build-time network
dependency. A submodule can be used if the downstream project standardizes on
that workflow, but H3 release builds should remain offline and reproducible.

## Required downstream gates

- exact wave32 and Sage run in the same executable with identical inputs;
- physical GPU and occupancy/concurrency state are recorded;
- production geometry is exercised without being hardcoded into dispatch;
- output includes max absolute error, RMSE, relative RMSE, cosine,
  non-finite count, and stable hashes;
- one block and 50-layer propagation pass;
- 8-NFE video and audio latent gates pass;
- three real prompts pass decoded video, audio, container, and A/V sync gates;
- exact wave32 remains available regardless of Sage acceptance.

Current E27 passes the performance, operator, propagation, video, VAE, mux,
and synchronization gates but fails audio for all three staged prompts. It is
therefore an explicit research integration, not a stable/default path.
