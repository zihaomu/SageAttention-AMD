#include "sage_attention_portable_reference.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr std::uint32_t kThreads = 256;

__global__ __launch_bounds__(kThreads) void portable_exact_interval_kernel(
    const hip_bfloat16 *__restrict__ query,
    const hip_bfloat16 *__restrict__ key,
    const hip_bfloat16 *__restrict__ value,
    hip_bfloat16 *__restrict__ output,
    const sageattention::q_task *__restrict__ tasks,
    std::uint32_t task_count,
    std::uint32_t sequence,
    std::uint32_t heads,
    std::uint32_t head_dimension,
    float attention_scale) {
    const std::uint32_t query_row = blockIdx.x;
    const std::uint32_t head = blockIdx.y;
    const std::uint32_t thread = threadIdx.x;
    if (query_row >= sequence || head >= heads || thread >= kThreads) return;

    const sageattention::q_task *task = nullptr;
    for (std::uint32_t index = 0; index < task_count; ++index) {
        const sageattention::q_task *candidate = tasks + index;
        if (query_row >= candidate->q_begin &&
            query_row - candidate->q_begin < candidate->q_count) {
            task = candidate;
            break;
        }
    }
    if (!task) return;

    __shared__ float reduction[kThreads];
    __shared__ float rescale;
    __shared__ float weight;
    __shared__ float denominator;
    float output_accumulator = 0.0f;
    float running_max = -INFINITY;
    if (thread == 0) denominator = 0.0f;
    __syncthreads();

    const std::size_t query_base =
        (static_cast<std::size_t>(query_row) * heads + head) * head_dimension;
    for (std::uint32_t interval_index = 0;
         interval_index < task->interval_count; ++interval_index) {
        const sageattention::key_interval interval =
            task->allowed[interval_index];
        for (std::uint32_t key_row = interval.begin; key_row < interval.end;
             ++key_row) {
            const std::size_t key_base =
                (static_cast<std::size_t>(key_row) * heads + head) *
                head_dimension;
            float partial = 0.0f;
            for (std::uint32_t dimension = thread;
                 dimension < head_dimension; dimension += kThreads) {
                partial += static_cast<float>(query[query_base + dimension]) *
                           static_cast<float>(key[key_base + dimension]);
            }
            reduction[thread] = partial;
            __syncthreads();
            for (std::uint32_t stride = kThreads / 2; stride; stride >>= 1) {
                if (thread < stride)
                    reduction[thread] += reduction[thread + stride];
                __syncthreads();
            }
            if (thread == 0) {
                const float score = reduction[0] * attention_scale;
                const float next_max = fmaxf(running_max, score);
                rescale = expf(running_max - next_max);
                weight = expf(score - next_max);
                denominator = denominator * rescale + weight;
                running_max = next_max;
            }
            __syncthreads();
            if (thread < head_dimension) {
                output_accumulator =
                    output_accumulator * rescale +
                    weight * static_cast<float>(value[key_base + thread]);
            }
            __syncthreads();
        }
    }
    if (thread < head_dimension) {
        output[query_base + thread] =
            hip_bfloat16(output_accumulator / denominator);
    }
}

}  // namespace

namespace sageattention::reference {

hipError_t query_portable_exact_support(std::uint32_t query_sequence,
                                        std::uint32_t query_heads,
                                        std::uint32_t head_dimension) {
    if (!query_sequence || !query_heads || !head_dimension ||
        head_dimension > kThreads) {
        return hipErrorNotSupported;
    }
    int device = 0;
    hipError_t error = hipGetDevice(&device);
    if (error != hipSuccess) return error;
    hipDeviceProp_t properties = {};
    error = hipGetDeviceProperties(&properties, device);
    if (error != hipSuccess) return error;
    if (properties.maxThreadsPerBlock < static_cast<int>(kThreads) ||
        query_sequence > static_cast<std::uint32_t>(properties.maxGridSize[0]) ||
        query_heads > static_cast<std::uint32_t>(properties.maxGridSize[1])) {
        return hipErrorNotSupported;
    }
    return hipSuccess;
}

hipError_t launch_portable_exact(const void *query_bf16,
                                 const void *key_bf16,
                                 const void *value_bf16,
                                 void *output_bf16,
                                 const q_task *device_tasks,
                                 std::size_t task_count,
                                 std::uint32_t sequence,
                                 std::uint32_t heads,
                                 std::uint32_t head_dimension,
                                 float scale,
                                 hipStream_t stream) {
    if (!query_bf16 || !key_bf16 || !value_bf16 || !output_bf16 ||
        !device_tasks || !task_count || task_count > UINT32_MAX ||
        !std::isfinite(scale) || scale <= 0.0f) {
        return hipErrorInvalidValue;
    }
    const hipError_t support =
        query_portable_exact_support(sequence, heads, head_dimension);
    if (support != hipSuccess) return support;
    const dim3 grid(sequence, heads);
    hipLaunchKernelGGL(
        portable_exact_interval_kernel, grid, dim3(kThreads), 0, stream,
        static_cast<const hip_bfloat16 *>(query_bf16),
        static_cast<const hip_bfloat16 *>(key_bf16),
        static_cast<const hip_bfloat16 *>(value_bf16),
        static_cast<hip_bfloat16 *>(output_bf16), device_tasks,
        static_cast<std::uint32_t>(task_count), sequence, heads,
        head_dimension, scale);
    return hipGetLastError();
}

}  // namespace sageattention::reference
