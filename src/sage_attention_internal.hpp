#ifndef SAGE_ATTENTION_INTERNAL_HPP
#define SAGE_ATTENTION_INTERNAL_HPP

#include "sage_attention.hpp"

#include <cstddef>

namespace sageattention {

struct workspace_layout {
    std::size_t q_i8_offset;
    std::size_t k_i8_offset;
    std::size_t q_scale_offset;
    std::size_t k_scale_offset;
    std::size_t tasks_offset;
    std::size_t q_groups;
    std::size_t k_groups;
    std::size_t task_count;
    std::size_t bytes;
};

struct kernel_registration {
    kernel_id id;
    const char *name;
    const char *architecture_prefixes[2];
    std::uint32_t wavefront_size;
    layout tensor_layout;
    data_type input_type;
    data_type output_type;
    qk_mode qk;
    pv_mode pv;
    std::uint32_t batch;
    std::uint32_t head_dimension;
    std::uint32_t max_query_rows_per_task;
    std::uint32_t max_intervals;
};

const kernel_registration *resolve_kernel_registration(
    const descriptor &operation);

bool device_matches_registration(
    const kernel_registration &registration,
    const hipDeviceProp_t &properties);

bool make_workspace_layout(
    const descriptor &operation,
    workspace_layout *result);

}  // namespace sageattention

#endif
