#ifndef SAGE_ATTENTION_HPP
#define SAGE_ATTENTION_HPP

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace sageattention {

inline constexpr std::uint32_t max_intervals_per_task = 5;

enum class layout : std::uint32_t {
    nhd = 0,
};

enum class data_type : std::uint32_t {
    bf16 = 0,
    fp16 = 1,
    fp8_e4m3 = 2,
};

enum class qk_mode : std::uint32_t {
    symmetric_i8 = 0,
};

enum class pv_mode : std::uint32_t {
    bf16 = 0,
    fp16 = 1,
    fp8_e4m3 = 2,
};

enum class kernel_id : std::uint32_t {
    automatic_select = 0,
    e27_gfx12_d128 = 1,
};

/* Logical tensor shape. NHD means [batch, sequence, heads, head_dimension].
 * The current E27 specialization requires batch 1, self-attention, equal Q/KV
 * head counts, and D=128. Separate fields preserve the generic API boundary. */
struct tensor_shape {
    std::uint32_t batch;
    std::uint32_t query_sequence;
    std::uint32_t key_value_sequence;
    std::uint32_t query_heads;
    std::uint32_t key_value_heads;
    std::uint32_t head_dimension;
    layout tensor_layout;
};

struct kernel_options {
    data_type input_type;
    data_type output_type;
    qk_mode qk;
    pv_mode pv;
    kernel_id kernel;
};

struct key_interval {
    std::uint32_t begin;
    std::uint32_t end;
};

/* All rows in one task share the same ordered, non-overlapping set of allowed
 * key intervals. The current E27 specialization accepts 1..32 query rows and
 * 1..max_intervals_per_task intervals per task. */
struct q_task {
    std::uint32_t q_begin;
    std::uint32_t q_count;
    std::uint32_t interval_count;
    key_interval allowed[max_intervals_per_task];
};

/* Host view consumed by prepare_workspace() or launch(). The task storage only
 * needs to remain valid for the duration of the call. Prepared launches read
 * the device copy in caller-owned workspace and do not retain this pointer. */
struct interval_plan {
    const q_task *tasks;
    std::size_t task_count;
};

struct descriptor {
    tensor_shape shape;
    kernel_options options;
    std::size_t task_count;
};

struct params {
    const void *query;
    const void *key;
    const void *value;
    void *output;
    descriptor operation;
    float scale;
    void *workspace;
    std::size_t workspace_bytes;
    hipStream_t stream;
};

struct profile {
    float q_quant_ms;
    float k_quant_ms;
    float attention_ms;
    float total_ms;
};

/* Host-only validation. Unsupported but structurally valid E27 combinations
 * return hipErrorNotSupported; malformed values return hipErrorInvalidValue. */
hipError_t validate_descriptor(const descriptor &operation);

/* Validates exact Q-row coverage, task geometry, interval order, and bounds. */
hipError_t validate_interval_plan(
    const descriptor &operation,
    const interval_plan &plan);

/* Host-only helper. Returns false for out-of-range indices or malformed views. */
bool key_allowed(
    const descriptor &operation,
    const interval_plan &plan,
    std::uint32_t query,
    std::uint32_t key);

std::int8_t quantize_symmetric_i8(float value, float scale);

/* Checks the current HIP device in addition to the host descriptor. */
hipError_t query_support(const descriptor &operation);

/* Returns zero for invalid/unsupported descriptors or size overflow. */
std::size_t workspace_size(const descriptor &operation);

/* Validates and uploads immutable interval metadata into caller-owned
 * workspace. The copy is ordered on stream. Call this on a plan cache miss. */
hipError_t prepare_workspace(
    const descriptor &operation,
    const interval_plan &plan,
    void *workspace,
    std::size_t workspace_bytes,
    hipStream_t stream);

/* Asynchronous hot path. Workspace must contain metadata prepared from an
 * interval plan whose task_count matches params.operation.task_count. */
hipError_t launch_prepared(const params &operation);

/* Synchronous diagnostic path with per-stage GPU event timing. */
hipError_t launch_profiled(
    const params &operation,
    profile *timing);

/* Convenience cold path: prepare interval metadata, then launch. */
hipError_t launch(
    const params &operation,
    const interval_plan &plan);

}  // namespace sageattention

#endif
