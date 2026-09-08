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

bool make_workspace_layout(
    const descriptor &operation,
    workspace_layout *result);

}  // namespace sageattention

#endif
