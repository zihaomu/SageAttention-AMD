#include "h3_vdn_sage.hpp"
#include "h3_vdn_sage_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr std::size_t kWorkspaceAlignment = 256;
constexpr std::uint32_t kQTileRows = 32;

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
    std::size_t widened;
    if (!checked_add(value, alignment - 1, &widened)) return false;
    *result = widened & ~(alignment - 1);
    return true;
}

std::uint64_t video_end_wide(const h3_vdn_sage_geometry &geometry) {
    return static_cast<std::uint64_t>(geometry.video_start) +
           static_cast<std::uint64_t>(geometry.frames) *
               geometry.tokens_per_frame;
}

std::uint32_t query_frame(const h3_vdn_sage_geometry &geometry,
                          std::uint32_t query) {
    return (query - geometry.video_start) / geometry.tokens_per_frame;
}

std::uint32_t append_interval(h3_vdn_key_interval *intervals,
                              std::uint32_t count,
                              std::uint32_t begin,
                              std::uint32_t end) {
    if (begin >= end) return count;
    intervals[count++] = {begin, end};
    return count;
}

std::uint32_t intervals_for_query(
    const h3_vdn_sage_geometry &geometry,
    std::uint32_t query,
    h3_vdn_key_interval *result) {
    const std::uint32_t video_end =
        static_cast<std::uint32_t>(video_end_wide(geometry));
    if (query < geometry.video_start || query >= video_end) {
        result[0] = {0, geometry.sequence};
        return 1;
    }

    const std::uint32_t frame = query_frame(geometry, query);
    if (geometry.anchor_both &&
        (frame == 0 || frame + 1 == geometry.frames)) {
        result[0] = {0, geometry.sequence};
        return 1;
    }

    h3_vdn_key_interval candidates[5];
    std::uint32_t count = 0;
    count = append_interval(candidates, count, 0, geometry.video_start);
    if (geometry.anchor_both) {
        count = append_interval(candidates, count, geometry.video_start,
                                geometry.video_start +
                                    geometry.tokens_per_frame);
    }

    std::uint32_t lower;
    std::uint32_t upper;
    if (geometry.chunk) {
        const std::uint32_t q_chunk = frame / geometry.chunk;
        const std::uint32_t lower_chunk =
            q_chunk > geometry.radius ? q_chunk - geometry.radius : 0;
        const std::uint64_t upper_chunks =
            static_cast<std::uint64_t>(q_chunk) + geometry.radius + 1;
        const std::uint64_t chunks_to_cover_frames =
            (static_cast<std::uint64_t>(geometry.frames) +
             geometry.chunk - 1) /
            geometry.chunk;
        lower = lower_chunk * geometry.chunk;
        upper = upper_chunks >= chunks_to_cover_frames
                    ? geometry.frames - 1
                    : static_cast<std::uint32_t>(
                          upper_chunks * geometry.chunk - 1);
    } else {
        lower = frame > geometry.radius ? frame - geometry.radius : 0;
        const std::uint64_t upper_wide =
            static_cast<std::uint64_t>(frame) + geometry.radius;
        upper = upper_wide >= geometry.frames
                    ? geometry.frames - 1
                    : static_cast<std::uint32_t>(upper_wide);
    }
    count = append_interval(
        candidates, count,
        geometry.video_start + lower * geometry.tokens_per_frame,
        geometry.video_start + (upper + 1) * geometry.tokens_per_frame);

    if (geometry.anchor_both) {
        count = append_interval(candidates, count,
                                video_end - geometry.tokens_per_frame,
                                video_end);
    }
    count = append_interval(candidates, count, video_end, geometry.sequence);

    std::sort(candidates, candidates + count,
              [](const h3_vdn_key_interval &a,
                 const h3_vdn_key_interval &b) {
                  return a.begin < b.begin ||
                         (a.begin == b.begin && a.end < b.end);
              });
    std::uint32_t merged = 0;
    for (std::uint32_t index = 0; index < count; ++index) {
        if (merged && candidates[index].begin <= result[merged - 1].end) {
            result[merged - 1].end =
                std::max(result[merged - 1].end, candidates[index].end);
        } else {
            result[merged++] = candidates[index];
        }
    }
    return merged;
}

template <typename Emit>
std::size_t visit_tasks(const h3_vdn_sage_geometry &geometry, Emit emit) {
    std::size_t count = 0;
    const std::uint32_t video_end =
        static_cast<std::uint32_t>(video_end_wide(geometry));
    std::uint32_t row = 0;
    while (row < geometry.sequence) {
        h3_vdn_key_interval mask[5] = {};
        const std::uint32_t mask_count =
            intervals_for_query(geometry, row, mask);
        std::uint32_t class_end;
        if (row < geometry.video_start) {
            class_end = geometry.video_start;
        } else if (row >= video_end) {
            class_end = geometry.sequence;
        } else {
            const std::uint32_t frame = query_frame(geometry, row);
            std::uint32_t end_frame;
            if (geometry.anchor_both && frame == 0) {
                end_frame = 1;
            } else if (geometry.anchor_both && frame + 1 == geometry.frames) {
                end_frame = geometry.frames;
            } else if (geometry.chunk) {
                const std::uint64_t candidate =
                    (static_cast<std::uint64_t>(frame / geometry.chunk) + 1) *
                    geometry.chunk;
                end_frame = candidate >= geometry.frames
                                ? geometry.frames
                                : static_cast<std::uint32_t>(candidate);
                if (geometry.anchor_both && end_frame == geometry.frames)
                    --end_frame;
            } else {
                end_frame = frame + 1;
            }
            class_end = geometry.video_start +
                        end_frame * geometry.tokens_per_frame;
        }
        if (class_end <= row || class_end > geometry.sequence)
            class_end = row + 1;
        while (row < class_end) {
            h3_vdn_q_task task = {};
            task.q_begin = row;
            task.q_count = std::min(kQTileRows, class_end - row);
            task.interval_count = mask_count;
            std::copy(mask, mask + mask_count, task.allowed);
            emit(count, task);
            ++count;
            row += task.q_count;
        }
    }
    return count;
}

}  // namespace

hipError_t h3_vdn_sage_validate_geometry(
    const h3_vdn_sage_geometry &geometry) {
    if (!geometry.sequence || !geometry.heads || geometry.head_dim != 128 ||
        !geometry.frames || !geometry.tokens_per_frame) {
        return hipErrorInvalidValue;
    }
    const std::uint64_t end = video_end_wide(geometry);
    if (geometry.video_start > geometry.sequence ||
        end > geometry.sequence || end > UINT32_MAX) {
        return hipErrorInvalidValue;
    }
    return hipSuccess;
}

bool h3_vdn_sage_key_allowed(const h3_vdn_sage_geometry &geometry,
                             std::uint32_t query,
                             std::uint32_t key) {
    if (h3_vdn_sage_validate_geometry(geometry) != hipSuccess ||
        query >= geometry.sequence || key >= geometry.sequence) {
        return false;
    }
    h3_vdn_key_interval intervals[5] = {};
    const std::uint32_t count =
        intervals_for_query(geometry, query, intervals);
    for (std::uint32_t index = 0; index < count; ++index) {
        if (key >= intervals[index].begin && key < intervals[index].end)
            return true;
    }
    return false;
}

std::size_t h3_vdn_sage_task_count(
    const h3_vdn_sage_geometry &geometry) {
    if (h3_vdn_sage_validate_geometry(geometry) != hipSuccess) return 0;
    return visit_tasks(geometry,
                       [](std::size_t, const h3_vdn_q_task &) {});
}

hipError_t h3_vdn_sage_build_tasks(
    const h3_vdn_sage_geometry &geometry,
    h3_vdn_q_task *tasks,
    std::size_t task_capacity,
    std::size_t *task_count) {
    if (!task_count || h3_vdn_sage_validate_geometry(geometry) != hipSuccess)
        return hipErrorInvalidValue;
    const std::size_t required = h3_vdn_sage_task_count(geometry);
    *task_count = required;
    if (!tasks || task_capacity < required) return hipErrorInvalidValue;
    visit_tasks(geometry,
                [tasks](std::size_t index, const h3_vdn_q_task &task) {
                    tasks[index] = task;
                });
    return hipSuccess;
}

std::int8_t h3_vdn_sage_quantize_symmetric_i8(float value, float scale) {
    if (!(scale > 0.0f) || !std::isfinite(scale) || !std::isfinite(value))
        return 0;
    const float clipped =
        std::max(-127.0f, std::min(127.0f, value / scale));
    return static_cast<std::int8_t>(std::nearbyint(clipped));
}

bool h3_vdn_sage_make_workspace_layout(
    const h3_vdn_sage_geometry &geometry,
    h3_vdn_sage_pv_mode mode,
    h3_vdn_sage_workspace_layout *layout) {
    if (!layout || mode != h3_vdn_sage_pv_mode::bf16 ||
        h3_vdn_sage_validate_geometry(geometry) != hipSuccess) {
        return false;
    }
    h3_vdn_sage_workspace_layout result = {};
    result.q_groups = (static_cast<std::size_t>(geometry.sequence) + 31) / 32;
    result.k_groups = (static_cast<std::size_t>(geometry.sequence) + 63) / 64;

    std::size_t elements;
    if (!checked_mul(geometry.sequence, geometry.heads, &elements) ||
        !checked_mul(elements, geometry.head_dim, &elements)) return false;
    result.task_count = h3_vdn_sage_task_count(geometry);

    std::size_t cursor = 0;
    result.q_i8_offset = cursor;
    if (!checked_add(cursor, elements, &cursor) ||
        !checked_align(cursor, kWorkspaceAlignment, &cursor)) return false;
    result.k_i8_offset = cursor;
    if (!checked_add(cursor, elements, &cursor) ||
        !checked_align(cursor, kWorkspaceAlignment, &cursor)) return false;

    std::size_t scale_elements;
    std::size_t scale_bytes;
    result.q_scale_offset = cursor;
    if (!checked_mul(geometry.heads, result.q_groups, &scale_elements) ||
        !checked_mul(scale_elements, sizeof(float), &scale_bytes) ||
        !checked_add(cursor, scale_bytes, &cursor) ||
        !checked_align(cursor, kWorkspaceAlignment, &cursor)) return false;
    result.k_scale_offset = cursor;
    if (!checked_mul(geometry.heads, result.k_groups, &scale_elements) ||
        !checked_mul(scale_elements, sizeof(float), &scale_bytes) ||
        !checked_add(cursor, scale_bytes, &cursor) ||
        !checked_align(cursor, kWorkspaceAlignment, &cursor)) return false;

    std::size_t task_bytes;
    result.tasks_offset = cursor;
    if (!checked_mul(result.task_count, sizeof(h3_vdn_q_task), &task_bytes) ||
        !checked_add(cursor, task_bytes, &cursor) ||
        !checked_align(cursor, kWorkspaceAlignment, &cursor)) return false;
    result.bytes = cursor;
    *layout = result;
    return true;
}

std::size_t h3_vdn_sage_workspace_size(
    const h3_vdn_sage_geometry &geometry,
    h3_vdn_sage_pv_mode mode) {
    h3_vdn_sage_workspace_layout layout = {};
    return h3_vdn_sage_make_workspace_layout(geometry, mode, &layout)
               ? layout.bytes
               : 0;
}
