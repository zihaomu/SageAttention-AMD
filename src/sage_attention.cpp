#include "sage_attention.hpp"
#include "sage_attention_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr std::size_t kWorkspaceAlignment = 256;

bool checked_add(std::size_t a, std::size_t b, std::size_t *result) {
    if (a > std::numeric_limits<std::size_t>::max() - b) return false;
    *result = a + b;
    return true;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t *result) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) return false;
    *result = a * b;
    return true;
}

bool checked_align(std::size_t value, std::size_t alignment,
                   std::size_t *result) {
    std::size_t widened = 0;
    if (!checked_add(value, alignment - 1, &widened)) return false;
    *result = widened & ~(alignment - 1);
    return true;
}

}  // namespace

namespace sageattention {

hipError_t validate_descriptor(const descriptor &operation) {
    const tensor_shape &shape = operation.shape;
    if (!shape.batch || !shape.query_sequence || !shape.key_value_sequence ||
        !shape.query_heads || !shape.key_value_heads ||
        !shape.head_dimension || !operation.task_count ||
        operation.task_count > UINT32_MAX) {
        return hipErrorInvalidValue;
    }
    if (shape.tensor_layout != layout::nhd || shape.batch != 1 ||
        shape.query_sequence != shape.key_value_sequence ||
        shape.query_heads != shape.key_value_heads ||
        shape.head_dimension != 128 ||
        operation.options.input_type != data_type::bf16 ||
        operation.options.output_type != data_type::bf16 ||
        operation.options.qk != qk_mode::symmetric_i8 ||
        operation.options.pv != pv_mode::bf16 ||
        (operation.options.kernel != kernel_id::automatic_select &&
         operation.options.kernel != kernel_id::e27_gfx12_d128)) {
        return hipErrorNotSupported;
    }
    return hipSuccess;
}

hipError_t validate_interval_plan(const descriptor &operation,
                                  const interval_plan &plan) {
    const hipError_t descriptor_error = validate_descriptor(operation);
    if (descriptor_error != hipSuccess) return descriptor_error;
    if (!plan.tasks || plan.task_count != operation.task_count) {
        return hipErrorInvalidValue;
    }

    std::uint32_t next_query = 0;
    for (std::size_t task_index = 0; task_index < plan.task_count;
         ++task_index) {
        const q_task &task = plan.tasks[task_index];
        if (task.q_begin != next_query || !task.q_count ||
            task.q_count > 32 ||
            task.q_count > operation.shape.query_sequence - next_query ||
            !task.interval_count ||
            task.interval_count > max_intervals_per_task) {
            return hipErrorInvalidValue;
        }
        std::uint32_t previous_end = 0;
        for (std::uint32_t interval_index = 0;
             interval_index < task.interval_count; ++interval_index) {
            const key_interval &interval = task.allowed[interval_index];
            if (interval.begin >= interval.end ||
                interval.end > operation.shape.key_value_sequence ||
                (interval_index && interval.begin < previous_end)) {
                return hipErrorInvalidValue;
            }
            previous_end = interval.end;
        }
        next_query += task.q_count;
    }
    return next_query == operation.shape.query_sequence
               ? hipSuccess
               : hipErrorInvalidValue;
}

bool key_allowed(const descriptor &operation, const interval_plan &plan,
                 std::uint32_t query, std::uint32_t key) {
    if (!plan.tasks || plan.task_count != operation.task_count ||
        query >= operation.shape.query_sequence ||
        key >= operation.shape.key_value_sequence) {
        return false;
    }
    for (std::size_t task_index = 0; task_index < plan.task_count;
         ++task_index) {
        const q_task &task = plan.tasks[task_index];
        const std::uint64_t task_end =
            static_cast<std::uint64_t>(task.q_begin) + task.q_count;
        if (query < task.q_begin || query >= task_end) continue;
        if (task.interval_count > max_intervals_per_task) return false;
        for (std::uint32_t interval_index = 0;
             interval_index < task.interval_count; ++interval_index) {
            const key_interval &interval = task.allowed[interval_index];
            if (key >= interval.begin && key < interval.end) return true;
        }
        return false;
    }
    return false;
}

std::int8_t quantize_symmetric_i8(float value, float scale) {
    if (!(scale > 0.0f) || !std::isfinite(scale) || !std::isfinite(value))
        return 0;
    const float clipped =
        std::max(-127.0f, std::min(127.0f, value / scale));
    return static_cast<std::int8_t>(std::nearbyint(clipped));
}

bool make_workspace_layout(const descriptor &operation,
                           workspace_layout *layout_result) {
    if (!layout_result || validate_descriptor(operation) != hipSuccess)
        return false;

    workspace_layout result = {};
    result.q_groups =
        (static_cast<std::size_t>(operation.shape.query_sequence) + 31) / 32;
    result.k_groups =
        (static_cast<std::size_t>(operation.shape.key_value_sequence) + 63) /
        64;
    result.task_count = operation.task_count;

    std::size_t q_elements = 0;
    std::size_t k_elements = 0;
    if (!checked_mul(operation.shape.batch, operation.shape.query_sequence,
                     &q_elements) ||
        !checked_mul(q_elements, operation.shape.query_heads, &q_elements) ||
        !checked_mul(q_elements, operation.shape.head_dimension,
                     &q_elements) ||
        !checked_mul(operation.shape.batch,
                     operation.shape.key_value_sequence, &k_elements) ||
        !checked_mul(k_elements, operation.shape.key_value_heads,
                     &k_elements) ||
        !checked_mul(k_elements, operation.shape.head_dimension,
                     &k_elements)) {
        return false;
    }

    std::size_t cursor = 0;
    result.q_i8_offset = cursor;
    if (!checked_add(cursor, q_elements, &cursor) ||
        !checked_align(cursor, kWorkspaceAlignment, &cursor)) return false;
    result.k_i8_offset = cursor;
    if (!checked_add(cursor, k_elements, &cursor) ||
        !checked_align(cursor, kWorkspaceAlignment, &cursor)) return false;

    std::size_t scale_elements = 0;
    std::size_t scale_bytes = 0;
    result.q_scale_offset = cursor;
    if (!checked_mul(operation.shape.query_heads, result.q_groups,
                     &scale_elements) ||
        !checked_mul(scale_elements, sizeof(float), &scale_bytes) ||
        !checked_add(cursor, scale_bytes, &cursor) ||
        !checked_align(cursor, kWorkspaceAlignment, &cursor)) return false;
    result.k_scale_offset = cursor;
    if (!checked_mul(operation.shape.key_value_heads, result.k_groups,
                     &scale_elements) ||
        !checked_mul(scale_elements, sizeof(float), &scale_bytes) ||
        !checked_add(cursor, scale_bytes, &cursor) ||
        !checked_align(cursor, kWorkspaceAlignment, &cursor)) return false;

    std::size_t task_bytes = 0;
    result.tasks_offset = cursor;
    if (!checked_mul(result.task_count, sizeof(q_task), &task_bytes) ||
        !checked_add(cursor, task_bytes, &cursor) ||
        !checked_align(cursor, kWorkspaceAlignment, &cursor)) return false;
    result.bytes = cursor;
    *layout_result = result;
    return true;
}

std::size_t workspace_size(const descriptor &operation) {
    workspace_layout layout_result = {};
    return make_workspace_layout(operation, &layout_result)
               ? layout_result.bytes
               : 0;
}

}  // namespace sageattention
