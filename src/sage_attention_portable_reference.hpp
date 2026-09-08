#ifndef SAGE_ATTENTION_PORTABLE_REFERENCE_HPP
#define SAGE_ATTENTION_PORTABLE_REFERENCE_HPP

#include "sage_attention.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace sageattention::reference {

/* Development-time exact BF16 reference. It is intentionally separate from
 * the public dispatcher and is never selected as an optimized default. */
hipError_t query_portable_exact_support(
    std::uint32_t query_sequence,
    std::uint32_t query_heads,
    std::uint32_t head_dimension);

hipError_t launch_portable_exact(
    const void *query_bf16,
    const void *key_bf16,
    const void *value_bf16,
    void *output_bf16,
    const q_task *device_tasks,
    std::size_t task_count,
    std::uint32_t sequence,
    std::uint32_t heads,
    std::uint32_t head_dimension,
    float scale,
    hipStream_t stream);

}  // namespace sageattention::reference

#endif
