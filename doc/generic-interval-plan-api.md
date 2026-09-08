# Generic interval-plan API

Status: experimental on `main`, targeting version 0.2.

The public header [`include/sage_attention.hpp`](../include/sage_attention.hpp)
separates model mask construction from kernel dispatch. Consumers describe a
tensor operation and provide canonical query tasks whose rows share an ordered
set of allowed key intervals.

## Data model

`sageattention::tensor_shape` has independent query/KV sequence lengths and
head counts so the public boundary does not assume self-attention or equal head
counts. The current E27 dispatcher accepts only:

- batch 1 and NHD layout;
- equal Q/KV sequence lengths and head counts;
- BF16 input/output and head dimension 128;
- signed symmetric INT8 QK plus BF16 PV;
- gfx12 wave32 execution;
- 1–32 query rows and 1–5 key intervals per task.

Unsupported combinations return `hipErrorNotSupported`; malformed descriptors
or plans return `hipErrorInvalidValue`.

An `interval_plan` is a host view of `q_task` records. A canonical plan must:

1. cover every query row exactly once in increasing order;
2. keep each task within the query sequence;
3. contain at least one non-empty interval per task;
4. keep intervals ordered, non-overlapping, and within the KV sequence;
5. match `descriptor::task_count`.

Adjacent intervals are valid. Fully masked query tasks are not supported by the
current numerical contract.

## Lifecycle

```cpp
using namespace sageattention;

descriptor operation = {/* shape */, /* options */, task_count};
interval_plan plan = {tasks.data(), tasks.size()};

if (validate_interval_plan(operation, plan) != hipSuccess)
    return;
if (query_support(operation) != hipSuccess)
    return;

const std::size_t bytes = workspace_size(operation);
// Allocate bytes on the selected HIP device.

prepare_workspace(operation, plan, workspace, bytes, stream);
launch_prepared(params);
```

`prepare_workspace()` copies immutable task metadata into caller-owned device
workspace. Callers may release the host plan after the call returns. Reuse the
same prepared metadata for hot launches with the same descriptor and plan.
`launch()` is the convenience cold path, while `launch_profiled()` is a
synchronous diagnostic path.

The library does not allocate device memory, change the current HIP device, or
own the caller stream.

## Dispatch

`kernel_id::automatic_select` chooses a specialization from the descriptor and
current device. `kernel_id::e27_gfx12_d128` requests the current specialization
explicitly. Both currently resolve to the same E27 path after capability
validation; keeping the selector explicit allows later kernels to coexist.

## H3 compatibility

[`include/h3_vdn_sage.hpp`](../include/h3_vdn_sage.hpp) is retained unchanged at
the declaration and data-layout level. Its planner converts VDN geometry into
generic tasks, prepares the generic workspace, and forwards hot/profiled
launches to the generic dispatcher. Existing H3 callers do not need source
changes.

Compatibility tests verify:

- the v0.1 public header declarations remain unchanged;
- H3 and generic workspace sizes match;
- generic and H3 key-allowance decisions match;
- prepared generic and H3 GPU results are bit-for-bit identical;
- the E27 device instruction stream is unchanged.
