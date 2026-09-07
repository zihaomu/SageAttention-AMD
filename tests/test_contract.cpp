#include "h3_vdn_sage.hpp"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                         #condition);                                        \
            return false;                                                    \
        }                                                                    \
    } while (0)

namespace {

bool scalar_allowed(const h3_vdn_sage_geometry &g, std::uint32_t query,
                    std::uint32_t key) {
    const std::uint32_t video_end =
        g.video_start + g.frames * g.tokens_per_frame;
    if (query < g.video_start || query >= video_end || key < g.video_start ||
        key >= video_end)
        return true;
    const std::uint32_t qf =
        (query - g.video_start) / g.tokens_per_frame;
    const std::uint32_t kf =
        (key - g.video_start) / g.tokens_per_frame;
    if (g.anchor_both &&
        (qf == 0 || qf + 1 == g.frames || kf == 0 || kf + 1 == g.frames))
        return true;
    std::uint32_t lower;
    std::uint32_t upper;
    if (g.chunk) {
        const std::uint32_t q_chunk = qf / g.chunk;
        lower = q_chunk > g.radius ? (q_chunk - g.radius) * g.chunk : 0;
        const std::uint64_t upper_chunks =
            static_cast<std::uint64_t>(q_chunk) + g.radius + 1;
        const std::uint64_t chunks_to_cover_frames =
            (static_cast<std::uint64_t>(g.frames) + g.chunk - 1) / g.chunk;
        upper = upper_chunks >= chunks_to_cover_frames
                    ? g.frames - 1
                    : static_cast<std::uint32_t>(upper_chunks * g.chunk - 1);
    } else {
        lower = qf > g.radius ? qf - g.radius : 0;
        const std::uint64_t end =
            static_cast<std::uint64_t>(qf) + g.radius;
        upper = end >= g.frames ? g.frames - 1
                                : static_cast<std::uint32_t>(end);
    }
    return kf >= lower && kf <= upper;
}

bool check_geometry(const h3_vdn_sage_geometry &g) {
    CHECK(h3_vdn_sage_validate_geometry(g) == hipSuccess);
    const std::size_t count = h3_vdn_sage_task_count(g);
    CHECK(count > 0);
    std::vector<h3_vdn_q_task> tasks(count);
    std::size_t built = 0;
    CHECK(h3_vdn_sage_build_tasks(g, tasks.data(), tasks.size(), &built) ==
          hipSuccess);
    CHECK(built == count);

    std::uint32_t next_query = 0;
    for (const h3_vdn_q_task &task : tasks) {
        CHECK(task.q_begin == next_query);
        CHECK(task.q_count >= 1 && task.q_count <= 32);
        CHECK(task.q_begin + task.q_count <= g.sequence);
        CHECK(task.interval_count >= 1 && task.interval_count <= 5);
        for (std::uint32_t index = 0; index < task.interval_count; ++index) {
            CHECK(task.allowed[index].begin < task.allowed[index].end);
            CHECK(task.allowed[index].end <= g.sequence);
            if (index)
                CHECK(task.allowed[index - 1].end < task.allowed[index].begin);
        }
        for (std::uint32_t local = 0; local < task.q_count; ++local) {
            const std::uint32_t query = task.q_begin + local;
            for (std::uint32_t key = 0; key < g.sequence; ++key) {
                bool interval_allowed = false;
                for (std::uint32_t index = 0; index < task.interval_count;
                     ++index) {
                    interval_allowed |=
                        key >= task.allowed[index].begin &&
                        key < task.allowed[index].end;
                }
                CHECK(interval_allowed == scalar_allowed(g, query, key));
                CHECK(interval_allowed ==
                      h3_vdn_sage_key_allowed(g, query, key));
            }
        }
        next_query += task.q_count;
    }
    CHECK(next_query == g.sequence);
    return true;
}

bool test_masks_and_tasks() {
    const h3_vdn_sage_geometry cases[] = {
        /* Misaligned S/video boundaries plus non-video suffix. */
        {83, 2, 128, 11, 6, 9, 1, 2, true},
        /* frames < chunk, incomplete tile, anchors disabled. */
        {37, 1, 128, 7, 3, 6, 4, 8, false},
        /* Per-frame window mode (chunk == 0). */
        {65, 3, 128, 3, 7, 7, 2, 0, true},
        /* Single-frame video: first/last anchor are the same interval. */
        {31, 1, 128, 5, 1, 13, 0, 5, true},
        /* Extreme radius/chunk must saturate without uint64 multiplication. */
        {31, 1, 128, 3, 5, 5, UINT32_MAX, UINT32_MAX, true},
    };
    for (const auto &geometry : cases) CHECK(check_geometry(geometry));

    /* Production H3 geometry is checked without an O(S^2) scalar sweep. */
    const h3_vdn_sage_geometry production = {
        5338, 56, 128, 986, 17, 256, 1, 5, true};
    CHECK(h3_vdn_sage_validate_geometry(production) == hipSuccess);
    const std::size_t count = h3_vdn_sage_task_count(production);
    CHECK(count >= (production.sequence + 31) / 32);
    std::vector<h3_vdn_q_task> tasks(count);
    std::size_t built = 0;
    CHECK(h3_vdn_sage_build_tasks(production, tasks.data(), tasks.size(),
                                  &built) == hipSuccess);
    CHECK(built == count);
    CHECK(tasks.front().q_begin == 0);
    CHECK(tasks.back().q_begin + tasks.back().q_count == production.sequence);
    return true;
}

bool test_quantization_contract() {
    CHECK(std::fesetround(FE_TONEAREST) == 0);
    CHECK(h3_vdn_sage_quantize_symmetric_i8(0.5f, 1.0f) == 0);
    CHECK(h3_vdn_sage_quantize_symmetric_i8(1.5f, 1.0f) == 2);
    CHECK(h3_vdn_sage_quantize_symmetric_i8(2.5f, 1.0f) == 2);
    CHECK(h3_vdn_sage_quantize_symmetric_i8(-1.5f, 1.0f) == -2);
    CHECK(h3_vdn_sage_quantize_symmetric_i8(999.0f, 1.0f) == 127);
    CHECK(h3_vdn_sage_quantize_symmetric_i8(-999.0f, 1.0f) == -127);
    CHECK(h3_vdn_sage_quantize_symmetric_i8(3.0f, 0.0f) == 0);
    CHECK(h3_vdn_sage_quantize_symmetric_i8(
              std::numeric_limits<float>::infinity(), 1.0f) == 0);
    return true;
}

bool test_validation_and_workspace() {
    const h3_vdn_sage_geometry production = {
        5338, 56, 128, 986, 17, 256, 1, 5, true};
    const std::size_t bytes = h3_vdn_sage_workspace_size(
        production, h3_vdn_sage_pv_mode::bf16);
    CHECK(bytes > 70ull * 1024 * 1024);
    CHECK(bytes < 80ull * 1024 * 1024);
    CHECK(h3_vdn_sage_workspace_size(production,
                                     h3_vdn_sage_pv_mode::fp16) == 0);

    h3_vdn_sage_geometry invalid = production;
    invalid.head_dim = 64;
    CHECK(h3_vdn_sage_validate_geometry(invalid) == hipErrorInvalidValue);
    invalid = production;
    invalid.video_start = 1000;
    CHECK(h3_vdn_sage_validate_geometry(invalid) == hipErrorInvalidValue);
    invalid = production;
    invalid.tokens_per_frame = 0;
    CHECK(h3_vdn_sage_validate_geometry(invalid) == hipErrorInvalidValue);

    const h3_vdn_sage_geometry overflow = {
        UINT32_MAX, UINT32_MAX, 128, 0, 1, 1, 0, 1, false};
    CHECK(h3_vdn_sage_validate_geometry(overflow) == hipSuccess);
    CHECK(h3_vdn_sage_workspace_size(overflow,
                                     h3_vdn_sage_pv_mode::bf16) == 0);
    return true;
}

}  // namespace

int main() {
    if (!test_masks_and_tasks() || !test_quantization_contract() ||
        !test_validation_and_workspace())
        return EXIT_FAILURE;
    std::puts("VDN mask/task, INT8 rounding, validation and workspace passed");
    return EXIT_SUCCESS;
}
