#ifndef H3_VDN_SAGE_INTERNAL_HPP
#define H3_VDN_SAGE_INTERNAL_HPP

#include "h3_vdn_sage.hpp"

#include <cstddef>

struct h3_vdn_sage_workspace_layout {
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

bool h3_vdn_sage_make_workspace_layout(
    const h3_vdn_sage_geometry &geometry,
    h3_vdn_sage_pv_mode mode,
    h3_vdn_sage_workspace_layout *layout);

#endif
