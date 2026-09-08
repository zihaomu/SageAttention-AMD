#include "h3_vdn_sage.hpp"
#include "sage_attention.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr std::uint32_t kQTileRows = 32;

bool checked_mul(std::size_t a, std::size_t b, std::size_t *result) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) return false;
    *result = a * b;
    return true;
}

std::size_t sized_task_count(const h3_vdn_sage_geometry &geometry) {
    std::size_t elements = 0;
    if (!checked_mul(geometry.sequence, geometry.heads, &elements) ||
        !checked_mul(elements, geometry.head_dim, &elements)) return 0;
    return h3_vdn_sage_task_count(geometry);
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

sageattention::descriptor make_generic_descriptor(
    const h3_vdn_sage_geometry &geometry,
    h3_vdn_sage_pv_mode mode,
    std::size_t task_count) {
    return {
        {1, geometry.sequence, geometry.sequence, geometry.heads,
         geometry.heads, geometry.head_dim, sageattention::layout::nhd},
        {sageattention::data_type::bf16, sageattention::data_type::bf16,
         sageattention::qk_mode::symmetric_i8,
         static_cast<sageattention::pv_mode>(
             static_cast<std::uint32_t>(mode)),
         sageattention::kernel_id::e27_gfx12_d128},
        task_count};
}

sageattention::q_task make_generic_task(const h3_vdn_q_task &source) {
    sageattention::q_task result = {};
    result.q_begin = source.q_begin;
    result.q_count = source.q_count;
    result.interval_count = source.interval_count;
    for (std::uint32_t index = 0; index < source.interval_count; ++index) {
        result.allowed[index] = {source.allowed[index].begin,
                                 source.allowed[index].end};
    }
    return result;
}

sageattention::params make_generic_params(
    const h3_vdn_sage_params &source) {
    return {source.query_bf16,
            source.key_bf16,
            source.value_bf16,
            source.output_bf16,
            make_generic_descriptor(
                source.geometry, source.pv_mode,
                sized_task_count(source.geometry)),
            source.scale,
            source.workspace,
            source.workspace_bytes,
            source.stream};
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
    return sageattention::quantize_symmetric_i8(value, scale);
}

std::size_t h3_vdn_sage_workspace_size(
    const h3_vdn_sage_geometry &geometry,
    h3_vdn_sage_pv_mode mode) {
    if (h3_vdn_sage_validate_geometry(geometry) != hipSuccess) return 0;
    return sageattention::workspace_size(
        make_generic_descriptor(geometry, mode, sized_task_count(geometry)));
}

hipError_t h3_vdn_sage_prepare_workspace(
    const h3_vdn_sage_geometry &geometry,
    h3_vdn_sage_pv_mode mode,
    void *workspace,
    std::size_t workspace_bytes,
    hipStream_t stream) {
    if (h3_vdn_sage_validate_geometry(geometry) != hipSuccess)
        return hipErrorInvalidValue;
    const std::size_t task_count = sized_task_count(geometry);
    const sageattention::descriptor operation =
        make_generic_descriptor(geometry, mode, task_count);
    const hipError_t descriptor_error =
        sageattention::validate_descriptor(operation);
    if (descriptor_error != hipSuccess) return descriptor_error;

    try {
        std::vector<h3_vdn_q_task> h3_tasks(operation.task_count);
        std::size_t built_count = 0;
        hipError_t error = h3_vdn_sage_build_tasks(
            geometry, h3_tasks.data(), h3_tasks.size(), &built_count);
        if (error != hipSuccess || built_count != operation.task_count)
            return hipErrorInvalidValue;
        std::vector<sageattention::q_task> generic_tasks;
        generic_tasks.reserve(h3_tasks.size());
        for (const h3_vdn_q_task &task : h3_tasks)
            generic_tasks.push_back(make_generic_task(task));
        const sageattention::interval_plan plan = {
            generic_tasks.data(), generic_tasks.size()};
        return sageattention::prepare_workspace(
            operation, plan, workspace, workspace_bytes, stream);
    } catch (...) {
        return hipErrorOutOfMemory;
    }
}

hipError_t h3_vdn_sage_launch_prepared(const h3_vdn_sage_params &params) {
    if (h3_vdn_sage_validate_geometry(params.geometry) != hipSuccess)
        return hipErrorInvalidValue;
    return sageattention::launch_prepared(make_generic_params(params));
}

hipError_t h3_vdn_sage_launch_profiled(
    const h3_vdn_sage_params &params,
    h3_vdn_sage_profile *profile) {
    if (!profile) return hipErrorInvalidValue;
    sageattention::profile generic_profile = {};
    const hipError_t error = sageattention::launch_profiled(
        make_generic_params(params), &generic_profile);
    if (error == hipSuccess) {
        profile->q_quant_ms = generic_profile.q_quant_ms;
        profile->k_quant_ms = generic_profile.k_quant_ms;
        profile->attention_ms = generic_profile.attention_ms;
        profile->total_ms = generic_profile.total_ms;
    } else {
        *profile = {};
    }
    return error;
}

hipError_t h3_vdn_sage_launch(const h3_vdn_sage_params &params) {
    const hipError_t prepare_error = h3_vdn_sage_prepare_workspace(
        params.geometry, params.pv_mode, params.workspace,
        params.workspace_bytes, params.stream);
    if (prepare_error != hipSuccess) return prepare_error;
    return h3_vdn_sage_launch_prepared(params);
}
