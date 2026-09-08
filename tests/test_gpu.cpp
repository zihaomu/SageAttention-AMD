#include "h3_vdn_sage.hpp"
#include "sage_attention.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define HIP_CHECK(expression)                                                \
    do {                                                                     \
        const hipError_t status_ = (expression);                             \
        if (status_ != hipSuccess) {                                         \
            std::fprintf(stderr, "HIP FAIL %s:%d: %s: %s\n", __FILE__,      \
                         __LINE__, #expression, hipGetErrorString(status_));  \
            return false;                                                    \
        }                                                                    \
    } while (0)

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                         #condition);                                        \
            return false;                                                    \
        }                                                                    \
    } while (0)

namespace {

struct device_buffer {
    void *data = nullptr;
    ~device_buffer() {
        const hipError_t ignored = hipFree(data);
        (void)ignored;
    }
    bool allocate(std::size_t bytes) {
        return hipMalloc(&data, bytes) == hipSuccess;
    }
};

std::uint64_t hash_bf16(const std::vector<hip_bfloat16> &values) {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (const hip_bfloat16 value : values) {
        hash ^= value.data;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

sageattention::descriptor generic_descriptor(
    const h3_vdn_sage_geometry &geometry,
    std::size_t task_count) {
    return {{1, geometry.sequence, geometry.sequence, geometry.heads,
             geometry.heads, geometry.head_dim, sageattention::layout::nhd},
            {sageattention::data_type::bf16,
             sageattention::data_type::bf16,
             sageattention::qk_mode::symmetric_i8,
             sageattention::pv_mode::bf16,
             sageattention::kernel_id::automatic_select},
            task_count};
}

std::vector<sageattention::q_task> build_generic_tasks(
    const h3_vdn_sage_geometry &geometry) {
    std::vector<h3_vdn_q_task> h3_tasks(
        h3_vdn_sage_task_count(geometry));
    std::size_t built = 0;
    if (h3_vdn_sage_build_tasks(geometry, h3_tasks.data(), h3_tasks.size(),
                                &built) != hipSuccess ||
        built != h3_tasks.size()) {
        return {};
    }
    std::vector<sageattention::q_task> result(h3_tasks.size());
    for (std::size_t task_index = 0; task_index < h3_tasks.size();
         ++task_index) {
        result[task_index].q_begin = h3_tasks[task_index].q_begin;
        result[task_index].q_count = h3_tasks[task_index].q_count;
        result[task_index].interval_count =
            h3_tasks[task_index].interval_count;
        for (std::uint32_t interval_index = 0;
             interval_index < h3_tasks[task_index].interval_count;
             ++interval_index) {
            result[task_index].allowed[interval_index] = {
                h3_tasks[task_index].allowed[interval_index].begin,
                h3_tasks[task_index].allowed[interval_index].end};
        }
    }
    return result;
}

std::vector<float> cpu_attention(
    const h3_vdn_sage_geometry &g,
    const std::vector<hip_bfloat16> &query,
    const std::vector<hip_bfloat16> &key,
    const std::vector<hip_bfloat16> &value,
    float scale) {
    const std::size_t elements =
        static_cast<std::size_t>(g.sequence) * g.heads * g.head_dim;
    std::vector<float> result(elements, 0.0f);
    std::vector<float> scores(g.sequence);
    for (std::uint32_t q = 0; q < g.sequence; ++q) {
        for (std::uint32_t h = 0; h < g.heads; ++h) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::uint32_t k = 0; k < g.sequence; ++k) {
                if (!h3_vdn_sage_key_allowed(g, q, k)) {
                    scores[k] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float dot = 0.0f;
                const std::size_t qb =
                    (static_cast<std::size_t>(q) * g.heads + h) * g.head_dim;
                const std::size_t kb =
                    (static_cast<std::size_t>(k) * g.heads + h) * g.head_dim;
                for (std::uint32_t d = 0; d < g.head_dim; ++d)
                    dot = std::fma(static_cast<float>(query[qb + d]),
                                   static_cast<float>(key[kb + d]), dot);
                scores[k] = dot * scale;
                maximum = std::max(maximum, scores[k]);
            }
            float denominator = 0.0f;
            for (std::uint32_t k = 0; k < g.sequence; ++k)
                if (std::isfinite(scores[k]))
                    denominator += std::exp(scores[k] - maximum);
            for (std::uint32_t d = 0; d < g.head_dim; ++d) {
                float output = 0.0f;
                for (std::uint32_t k = 0; k < g.sequence; ++k) {
                    if (!std::isfinite(scores[k])) continue;
                    const std::size_t vi =
                        (static_cast<std::size_t>(k) * g.heads + h) *
                            g.head_dim +
                        d;
                    output = std::fma(
                        std::exp(scores[k] - maximum) / denominator,
                        static_cast<float>(value[vi]), output);
                }
                result[(static_cast<std::size_t>(q) * g.heads + h) *
                           g.head_dim +
                       d] = output;
            }
        }
    }
    return result;
}

bool run_operator(const h3_vdn_sage_geometry &g,
                  const std::vector<hip_bfloat16> &query,
                  const std::vector<hip_bfloat16> &key,
                  const std::vector<hip_bfloat16> &value,
                  std::vector<hip_bfloat16> *output) {
    const std::size_t tensor_bytes = query.size() * sizeof(query[0]);
    const std::size_t workspace_bytes = h3_vdn_sage_workspace_size(
        g, h3_vdn_sage_pv_mode::bf16);
    CHECK(workspace_bytes > 0 && output && output->size() == query.size());
    device_buffer dq, dk, dv, dout, workspace;
    CHECK(dq.allocate(tensor_bytes) && dk.allocate(tensor_bytes) &&
          dv.allocate(tensor_bytes) && dout.allocate(tensor_bytes) &&
          workspace.allocate(workspace_bytes));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyAsync(dq.data, query.data(), tensor_bytes,
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dk.data, key.data(), tensor_bytes,
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dv.data, value.data(), tensor_bytes,
                             hipMemcpyHostToDevice, stream));

    const h3_vdn_sage_params params = {
        dq.data, dk.data, dv.data, dout.data, g,
        1.0f / std::sqrt(static_cast<float>(g.head_dim)),
        h3_vdn_sage_pv_mode::bf16, workspace.data, workspace_bytes, stream};
    HIP_CHECK(h3_vdn_sage_launch(params));
    HIP_CHECK(hipMemcpyAsync(output->data(), dout.data, tensor_bytes,
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<hip_bfloat16> repeat(output->size());
    HIP_CHECK(h3_vdn_sage_launch_prepared(params));
    HIP_CHECK(hipMemcpyAsync(repeat.data(), dout.data, tensor_bytes,
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    CHECK(std::memcmp(output->data(), repeat.data(), tensor_bytes) == 0);

    std::vector<sageattention::q_task> generic_tasks =
        build_generic_tasks(g);
    CHECK(!generic_tasks.empty());
    const sageattention::descriptor operation =
        generic_descriptor(g, generic_tasks.size());
    const sageattention::interval_plan plan = {
        generic_tasks.data(), generic_tasks.size()};
    CHECK(sageattention::query_support(operation) == hipSuccess);
    CHECK(sageattention::workspace_size(operation) == workspace_bytes);
    const sageattention::params generic_params = {
        dq.data,
        dk.data,
        dv.data,
        dout.data,
        operation,
        1.0f / std::sqrt(static_cast<float>(g.head_dim)),
        workspace.data,
        workspace_bytes,
        stream};
    HIP_CHECK(sageattention::prepare_workspace(
        operation, plan, workspace.data, workspace_bytes, stream));
    HIP_CHECK(sageattention::launch_prepared(generic_params));
    std::vector<hip_bfloat16> generic_output(output->size());
    HIP_CHECK(hipMemcpyAsync(generic_output.data(), dout.data, tensor_bytes,
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    CHECK(std::memcmp(output->data(), generic_output.data(), tensor_bytes) ==
          0);
    HIP_CHECK(hipStreamDestroy(stream));
    return true;
}

bool run_generic_operator(
    const sageattention::descriptor &operation,
    const sageattention::interval_plan &plan,
    const std::vector<hip_bfloat16> &query,
    const std::vector<hip_bfloat16> &key,
    const std::vector<hip_bfloat16> &value,
    std::vector<hip_bfloat16> *output) {
    const std::size_t tensor_bytes = query.size() * sizeof(query[0]);
    const std::size_t workspace_bytes =
        sageattention::workspace_size(operation);
    CHECK(workspace_bytes > 0 && output && output->size() == query.size() &&
          key.size() == query.size() && value.size() == query.size());
    device_buffer dq, dk, dv, dout, workspace;
    CHECK(dq.allocate(tensor_bytes) && dk.allocate(tensor_bytes) &&
          dv.allocate(tensor_bytes) && dout.allocate(tensor_bytes) &&
          workspace.allocate(workspace_bytes));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyAsync(dq.data, query.data(), tensor_bytes,
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dk.data, key.data(), tensor_bytes,
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dv.data, value.data(), tensor_bytes,
                             hipMemcpyHostToDevice, stream));
    const sageattention::params params = {
        dq.data,
        dk.data,
        dv.data,
        dout.data,
        operation,
        1.0f / std::sqrt(
                   static_cast<float>(operation.shape.head_dimension)),
        workspace.data,
        workspace_bytes,
        stream};
    HIP_CHECK(sageattention::launch(params, plan));
    HIP_CHECK(hipMemcpyAsync(output->data(), dout.data, tensor_bytes,
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<hip_bfloat16> repeat(output->size());
    HIP_CHECK(sageattention::launch_prepared(params));
    HIP_CHECK(hipMemcpyAsync(repeat.data(), dout.data, tensor_bytes,
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    CHECK(std::memcmp(output->data(), repeat.data(), tensor_bytes) == 0);
    HIP_CHECK(hipStreamDestroy(stream));
    return true;
}

bool test_manual_generic_interval_plan() {
    const sageattention::q_task tasks[] = {
        {0, 10, 2, {{0, 5}, {20, 35}}},
        {10, 22, 1, {{4, 17}}},
        {32, 3, 1, {{0, 35}}},
    };
    const sageattention::descriptor operation = {
        {1, 35, 35, 1, 1, 128, sageattention::layout::nhd},
        {sageattention::data_type::bf16,
         sageattention::data_type::bf16,
         sageattention::qk_mode::symmetric_i8,
         sageattention::pv_mode::bf16,
         sageattention::kernel_id::automatic_select},
        3};
    const sageattention::interval_plan plan = {tasks, 3};
    CHECK(sageattention::validate_interval_plan(operation, plan) ==
          hipSuccess);

    const std::size_t elements = static_cast<std::size_t>(35) * 128;
    std::vector<hip_bfloat16> query(elements, hip_bfloat16(0.0f));
    std::vector<hip_bfloat16> key(elements, hip_bfloat16(0.0f));
    std::vector<hip_bfloat16> value(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        const int centered = static_cast<int>((index * 17) % 59) - 29;
        value[index] = hip_bfloat16(static_cast<float>(centered) * 0.015f);
    }
    std::vector<hip_bfloat16> output(elements);
    CHECK(run_generic_operator(operation, plan, query, key, value, &output));

    float max_absolute = 0.0f;
    for (std::uint32_t query_row = 0; query_row < 35; ++query_row) {
        std::uint32_t allowed_count = 0;
        for (std::uint32_t key_row = 0; key_row < 35; ++key_row)
            if (sageattention::key_allowed(operation, plan, query_row,
                                           key_row))
                ++allowed_count;
        CHECK(allowed_count > 0);
        for (std::uint32_t dimension = 0; dimension < 128; ++dimension) {
            float expected = 0.0f;
            for (std::uint32_t key_row = 0; key_row < 35; ++key_row) {
                if (!sageattention::key_allowed(operation, plan, query_row,
                                                key_row))
                    continue;
                expected += static_cast<float>(
                    value[static_cast<std::size_t>(key_row) * 128 +
                          dimension]);
            }
            expected /= static_cast<float>(allowed_count);
            const float observed = static_cast<float>(
                output[static_cast<std::size_t>(query_row) * 128 +
                       dimension]);
            CHECK(std::isfinite(observed));
            max_absolute =
                std::max(max_absolute, std::fabs(observed - expected));
        }
    }
    std::printf("generic manual intervals: max_abs=%.7g\n", max_absolute);
    CHECK(max_absolute < 0.005f);
    return true;
}

bool test_correctness_and_determinism() {
    const h3_vdn_sage_geometry geometry = {
        83, 2, 128, 11, 6, 9, 1, 2, true};
    const std::size_t elements = static_cast<std::size_t>(geometry.sequence) *
                                 geometry.heads * geometry.head_dim;
    std::vector<hip_bfloat16> q(elements), k(elements), v(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        q[index] = hip_bfloat16(std::sin(static_cast<float>(index) * 0.013f) *
                                0.35f);
        k[index] = hip_bfloat16(std::cos(static_cast<float>(index) * 0.017f) *
                                0.31f);
        const int centered = static_cast<int>(index % 47) - 23;
        v[index] = hip_bfloat16(static_cast<float>(centered) * 0.025f);
    }
    const float scale = 1.0f / std::sqrt(128.0f);
    const std::vector<float> reference = cpu_attention(geometry, q, k, v,
                                                       scale);
    std::vector<hip_bfloat16> actual(elements);
    CHECK(run_operator(geometry, q, k, v, &actual));

    double squared_error = 0.0;
    double squared_reference = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    float max_absolute = 0.0f;
    std::size_t non_finite = 0;
    std::vector<double> row_rmse;
    row_rmse.reserve(static_cast<std::size_t>(geometry.sequence) *
                     geometry.heads);
    for (std::uint32_t row = 0; row < geometry.sequence; ++row) {
        for (std::uint32_t head = 0; head < geometry.heads; ++head) {
            double row_error2 = 0.0;
            const std::size_t base =
                (static_cast<std::size_t>(row) * geometry.heads + head) *
                geometry.head_dim;
            for (std::uint32_t dimension = 0;
                 dimension < geometry.head_dim; ++dimension) {
                const double error =
                    static_cast<float>(actual[base + dimension]) -
                    reference[base + dimension];
                row_error2 += error * error;
            }
            row_rmse.push_back(
                std::sqrt(row_error2 / geometry.head_dim));
        }
    }
    for (std::size_t index = 0; index < elements; ++index) {
        const float observed = static_cast<float>(actual[index]);
        const float expected = reference[index];
        if (!std::isfinite(observed)) ++non_finite;
        const float error = observed - expected;
        max_absolute = std::max(max_absolute, std::fabs(error));
        squared_error += static_cast<double>(error) * error;
        squared_reference += static_cast<double>(expected) * expected;
        dot += static_cast<double>(expected) * observed;
        actual_norm += static_cast<double>(observed) * observed;
    }
    const double relative_rmse =
        std::sqrt(squared_error / squared_reference);
    const double rmse = std::sqrt(squared_error / elements);
    const double cosine = dot / std::sqrt(squared_reference * actual_norm);
    std::sort(row_rmse.begin(), row_rmse.end());
    const auto percentile = [&row_rmse](double fraction) {
        const std::size_t index = static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(row_rmse.size())) - 1.0);
        return row_rmse[std::min(index, row_rmse.size() - 1)];
    };
    std::printf("GPU correctness: max_abs=%.7g rmse=%.7g rel_rmse=%.7g "
                "cosine=%.9f row_rmse[p50/p95/p99/max]="
                "%.7g/%.7g/%.7g/%.7g non_finite=%zu hash=%016llx\n",
                max_absolute, rmse, relative_rmse, cosine, percentile(0.50),
                percentile(0.95), percentile(0.99), row_rmse.back(), non_finite,
                static_cast<unsigned long long>(hash_bf16(actual)));
    CHECK(non_finite == 0);
    CHECK(relative_rmse <= 0.05);
    CHECK(cosine >= 0.999);
    return true;
}

bool test_h3_17_frame_misaligned_geometry() {
    /* Scaled-down H3 analogue: same 17/chunk5/radius1 structure, while both
     * video boundaries cut 16/32/64-row groups and a suffix remains. */
    const h3_vdn_sage_geometry geometry = {
        353, 2, 128, 58, 17, 15, 1, 5, true};
    const std::size_t elements = static_cast<std::size_t>(geometry.sequence) *
                                 geometry.heads * geometry.head_dim;
    std::vector<hip_bfloat16> q(elements), k(elements), v(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        const int qv = static_cast<int>((index * 13) % 127) - 63;
        const int kv = static_cast<int>((index * 19) % 131) - 65;
        const int vv = static_cast<int>((index * 23) % 137) - 68;
        q[index] = hip_bfloat16(static_cast<float>(qv) * 0.004f);
        k[index] = hip_bfloat16(static_cast<float>(kv) * 0.004f);
        v[index] = hip_bfloat16(static_cast<float>(vv) * 0.006f);
    }
    const std::vector<float> reference = cpu_attention(
        geometry, q, k, v, 1.0f / std::sqrt(128.0f));
    std::vector<hip_bfloat16> actual(elements);
    CHECK(run_operator(geometry, q, k, v, &actual));
    double error2 = 0.0;
    double reference2 = 0.0;
    double actual2 = 0.0;
    double dot = 0.0;
    for (std::size_t index = 0; index < elements; ++index) {
        const double observed = static_cast<float>(actual[index]);
        const double expected = reference[index];
        const double error = observed - expected;
        error2 += error * error;
        reference2 += expected * expected;
        actual2 += observed * observed;
        dot += observed * expected;
    }
    const double relative_rmse = std::sqrt(error2 / reference2);
    const double cosine = dot / std::sqrt(reference2 * actual2);
    std::printf("17-frame H3 analogue: rel_rmse=%.7g cosine=%.9f\n",
                relative_rmse, cosine);
    CHECK(relative_rmse <= 0.05);
    CHECK(cosine >= 0.999);
    return true;
}

bool check_targeted_geometry(const char *name,
                             const h3_vdn_sage_geometry &geometry,
                             bool extreme_scores) {
    const std::size_t elements = static_cast<std::size_t>(geometry.sequence) *
                                 geometry.heads * geometry.head_dim;
    std::vector<hip_bfloat16> q(elements), k(elements), v(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        if (extreme_scores) {
            const float q_sign = (index & 1) ? -1.0f : 1.0f;
            const float k_sign = (index % 5) < 2 ? -1.0f : 1.0f;
            q[index] = hip_bfloat16(q_sign * 8.0f);
            k[index] = hip_bfloat16(k_sign * 8.0f);
        } else {
            const int qv = static_cast<int>((index * 7) % 61) - 30;
            const int kv = static_cast<int>((index * 11) % 67) - 33;
            q[index] = hip_bfloat16(static_cast<float>(qv) * 0.01f);
            k[index] = hip_bfloat16(static_cast<float>(kv) * 0.01f);
        }
        const int vv = static_cast<int>((index * 13) % 71) - 35;
        v[index] = hip_bfloat16(static_cast<float>(vv) * 0.0125f);
    }

    const std::vector<float> reference = cpu_attention(
        geometry, q, k, v, 1.0f / std::sqrt(128.0f));
    std::vector<hip_bfloat16> actual(elements);
    CHECK(run_operator(geometry, q, k, v, &actual));

    double error2 = 0.0;
    double reference2 = 0.0;
    double actual2 = 0.0;
    double dot = 0.0;
    std::size_t non_finite = 0;
    for (std::size_t index = 0; index < elements; ++index) {
        const double observed = static_cast<float>(actual[index]);
        const double expected = reference[index];
        if (!std::isfinite(observed)) ++non_finite;
        const double error = observed - expected;
        error2 += error * error;
        reference2 += expected * expected;
        actual2 += observed * observed;
        dot += observed * expected;
    }
    const double relative_rmse = std::sqrt(error2 / reference2);
    const double cosine = dot / std::sqrt(reference2 * actual2);
    std::printf("targeted %s: S=%u H=%u rel_rmse=%.7g cosine=%.9f "
                "non_finite=%zu\n",
                name, geometry.sequence, geometry.heads, relative_rmse,
                cosine, non_finite);
    CHECK(non_finite == 0);
    CHECK(relative_rmse <= 0.05);
    CHECK(cosine >= 0.999);
    return true;
}

bool test_targeted_h3_geometries() {
    const h3_vdn_sage_geometry single_dense_tail = {
        13, 1, 128, 0, 1, 13, 0, 1, true};
    const h3_vdn_sage_geometry one_key_intervals = {
        13, 1, 128, 0, 13, 1, 0, 0, false};
    const h3_vdn_sage_geometry anchor_off_multi_interval = {
        47, 2, 128, 5, 7, 5, 2, 3, false};
    const h3_vdn_sage_geometry extreme_scores = {
        31, 1, 128, 3, 7, 4, 0, 2, false};
    return check_targeted_geometry("single-dense-tail", single_dense_tail,
                                   false) &&
           check_targeted_geometry("one-key-intervals", one_key_intervals,
                                   false) &&
           check_targeted_geometry("anchor-off-multi-interval",
                                   anchor_off_multi_interval, false) &&
           check_targeted_geometry("extreme-scores", extreme_scores, true);
}

bool test_output_and_workspace_guards() {
    const h3_vdn_sage_geometry geometry = {
        29, 1, 128, 3, 5, 4, 0, 0, false};
    const std::size_t elements = static_cast<std::size_t>(geometry.sequence) *
                                 geometry.heads * geometry.head_dim;
    const std::size_t tensor_bytes = elements * sizeof(hip_bfloat16);
    const std::size_t workspace_bytes = h3_vdn_sage_workspace_size(
        geometry, h3_vdn_sage_pv_mode::bf16);
    constexpr std::size_t guard_bytes = 4096;
    constexpr unsigned char guard_value = 0xa5;

    std::vector<hip_bfloat16> q(elements), k(elements), v(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        q[index] = hip_bfloat16(static_cast<float>(index % 17) * 0.01f);
        k[index] = hip_bfloat16(static_cast<float>(index % 19) * -0.01f);
        v[index] = hip_bfloat16(static_cast<float>(index % 23) * 0.02f);
    }

    device_buffer dq, dk, dv, guarded_output, guarded_workspace;
    CHECK(dq.allocate(tensor_bytes) && dk.allocate(tensor_bytes) &&
          dv.allocate(tensor_bytes) &&
          guarded_output.allocate(tensor_bytes + 2 * guard_bytes) &&
          guarded_workspace.allocate(workspace_bytes + 2 * guard_bytes));
    auto *output = static_cast<unsigned char *>(guarded_output.data) +
                   guard_bytes;
    auto *workspace = static_cast<unsigned char *>(guarded_workspace.data) +
                      guard_bytes;
    HIP_CHECK(hipMemcpy(dq.data, q.data(), tensor_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dk.data, k.data(), tensor_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dv.data, v.data(), tensor_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(guarded_output.data, guard_value,
                        tensor_bytes + 2 * guard_bytes));
    HIP_CHECK(hipMemset(guarded_workspace.data, guard_value,
                        workspace_bytes + 2 * guard_bytes));

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    const h3_vdn_sage_params params = {
        dq.data, dk.data, dv.data, output, geometry,
        1.0f / std::sqrt(128.0f), h3_vdn_sage_pv_mode::bf16,
        workspace, workspace_bytes, stream};
    HIP_CHECK(h3_vdn_sage_launch(params));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<unsigned char> guard(guard_bytes);
    const auto guard_is_intact = [&guard]() {
        return std::all_of(guard.begin(), guard.end(), [](unsigned char byte) {
            return byte == guard_value;
        });
    };
    HIP_CHECK(hipMemcpy(guard.data(), guarded_output.data, guard_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(guard_is_intact());
    HIP_CHECK(hipMemcpy(guard.data(), output + tensor_bytes, guard_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(guard_is_intact());
    HIP_CHECK(hipMemcpy(guard.data(), guarded_workspace.data, guard_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(guard_is_intact());
    HIP_CHECK(hipMemcpy(guard.data(), workspace + workspace_bytes,
                        guard_bytes, hipMemcpyDeviceToHost));
    CHECK(guard_is_intact());
    HIP_CHECK(hipStreamDestroy(stream));
    std::puts("output/workspace guard canaries: intact");
    return true;
}

bool test_uniform_scores_is_masked_mean() {
    const h3_vdn_sage_geometry geometry = {
        35, 1, 128, 5, 5, 5, 0, 2, true};
    const std::size_t elements = static_cast<std::size_t>(geometry.sequence) *
                                 geometry.heads * geometry.head_dim;
    std::vector<hip_bfloat16> q(elements, hip_bfloat16(0.0f));
    std::vector<hip_bfloat16> k(elements, hip_bfloat16(0.0f));
    std::vector<hip_bfloat16> v(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        const int centered = static_cast<int>((index * 11) % 53) - 26;
        v[index] = hip_bfloat16(static_cast<float>(centered) * 0.02f);
    }
    const std::vector<float> reference = cpu_attention(
        geometry, q, k, v, 1.0f / std::sqrt(128.0f));
    std::vector<hip_bfloat16> actual(elements);
    CHECK(run_operator(geometry, q, k, v, &actual));
    float max_absolute = 0.0f;
    for (std::size_t index = 0; index < elements; ++index)
        max_absolute = std::max(
            max_absolute,
            std::fabs(static_cast<float>(actual[index]) - reference[index]));
    std::printf("uniform-score masked mean: max_abs=%.7g\n", max_absolute);
    if (max_absolute >= 0.005f) {
        for (std::uint32_t row = 0; row < 4; ++row) {
            std::printf("row %u d0..7:", row);
            for (std::uint32_t d = 0; d < 8; ++d) {
                const std::size_t index =
                    static_cast<std::size_t>(row) * geometry.head_dim + d;
                std::printf(" %.4f/%.4f", static_cast<float>(actual[index]),
                            reference[index]);
            }
            std::putchar('\n');
        }
    }
    CHECK(max_absolute < 0.005f);
    return true;
}

bool test_masked_key_and_value_do_not_leak() {
    const h3_vdn_sage_geometry geometry = {
        83, 1, 128, 11, 6, 9, 0, 2, false};
    const std::size_t elements = static_cast<std::size_t>(geometry.sequence) *
                                 geometry.heads * geometry.head_dim;
    std::vector<hip_bfloat16> q(elements, hip_bfloat16(0.125f));
    std::vector<hip_bfloat16> k(elements, hip_bfloat16(-0.125f));
    std::vector<hip_bfloat16> v(elements);
    for (std::size_t index = 0; index < elements; ++index)
        v[index] = hip_bfloat16(static_cast<float>(index % 31) * 0.01f);
    std::vector<hip_bfloat16> changed_k = k;
    std::vector<hip_bfloat16> changed_v = v;

    const std::uint32_t query = geometry.video_start + 2 * 9 + 3;
    const std::uint32_t masked_key = geometry.video_start + 5 * 9 + 2;
    CHECK(!h3_vdn_sage_key_allowed(geometry, query, masked_key));
    for (std::uint32_t d = 0; d < geometry.head_dim; ++d) {
        changed_k[static_cast<std::size_t>(masked_key) * geometry.head_dim +
                  d] = hip_bfloat16(0.125f);
        changed_v[static_cast<std::size_t>(masked_key) * geometry.head_dim +
                  d] = hip_bfloat16(1000.0f);
    }

    std::vector<hip_bfloat16> baseline(elements), changed(elements);
    CHECK(run_operator(geometry, q, k, v, &baseline));
    CHECK(run_operator(geometry, q, changed_k, changed_v, &changed));
    const std::size_t row_base =
        static_cast<std::size_t>(query) * geometry.head_dim;
    CHECK(std::memcmp(baseline.data() + row_base, changed.data() + row_base,
                      geometry.head_dim * sizeof(baseline[0])) == 0);
    return true;
}

}  // namespace

int main() {
    hipDeviceProp_t properties = {};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        std::fputs("unable to inspect visible GPU\n", stderr);
        return EXIT_FAILURE;
    }
    std::printf("visible GPU: %s arch=%s warp=%d\n", properties.name,
                properties.gcnArchName, properties.warpSize);
    if (std::strncmp(properties.gcnArchName, "gfx1201", 7) != 0 ||
        properties.warpSize != 32) {
        std::fputs("test requires a selected gfx1201/wave32 GPU\n", stderr);
        return EXIT_FAILURE;
    }
    if (!test_uniform_scores_is_masked_mean() ||
        !test_manual_generic_interval_plan() ||
        !test_correctness_and_determinism() ||
        !test_h3_17_frame_misaligned_geometry() ||
        !test_targeted_h3_geometries() ||
        !test_output_and_workspace_guards() ||
        !test_masked_key_and_value_do_not_leak())
        return EXIT_FAILURE;
    std::puts("gfx12 INT8 QK + BF16 PV generic/H3 SageAttention tests "
              "passed");
    return EXIT_SUCCESS;
}
