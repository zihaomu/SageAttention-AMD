#include "h3_vdn_sage.hpp"
#include "sage_attention.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

bool hip_ok(hipError_t status, const char *operation) {
    if (status == hipSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 hipGetErrorString(status));
    return false;
}

void ignore_hip_error(hipError_t status) { (void)status; }

std::uint64_t hash_bf16(const std::vector<hip_bfloat16> &values) {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (const hip_bfloat16 value : values) {
        hash ^= value.data;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

}  // namespace

int main(int argc, char **argv) {
    std::uint32_t iterations = 5;
    bool bf16_qk = false;
    if (argc >= 2) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(argv[1], &end, 10);
        if (!end || *end || parsed == 0 || parsed > UINT32_MAX) {
            std::fprintf(stderr, "usage: %s [ITERATIONS] [e27|e33]\n",
                         argv[0]);
            return 2;
        }
        iterations = static_cast<std::uint32_t>(parsed);
    }
    if (argc == 3) {
        if (std::strcmp(argv[2], "e33") == 0)
            bf16_qk = true;
        else if (std::strcmp(argv[2], "e27") != 0) {
            std::fprintf(stderr, "usage: %s [ITERATIONS] [e27|e33]\n",
                         argv[0]);
            return 2;
        }
    } else if (argc > 3) {
        std::fprintf(stderr, "usage: %s [ITERATIONS] [e27|e33]\n",
                     argv[0]);
        return 2;
    }

    const h3_vdn_sage_geometry geometry = {
        5338, 56, 128, 986, 17, 256, 1, 5, true};
    const std::size_t elements = static_cast<std::size_t>(geometry.sequence) *
                                 geometry.heads * geometry.head_dim;
    const std::size_t tensor_bytes = elements * sizeof(hip_bfloat16);
    const std::size_t task_count = h3_vdn_sage_task_count(geometry);
    std::vector<h3_vdn_q_task> h3_tasks(task_count);
    std::size_t built_tasks = 0;
    if (h3_vdn_sage_build_tasks(geometry, h3_tasks.data(), h3_tasks.size(),
                                &built_tasks) != hipSuccess ||
        built_tasks != task_count) {
        std::fputs("unable to build H3 ordered-interval plan\n", stderr);
        return EXIT_FAILURE;
    }
    std::vector<sageattention::q_task> generic_tasks(task_count);
    for (std::size_t task = 0; task < task_count; ++task) {
        generic_tasks[task].q_begin = h3_tasks[task].q_begin;
        generic_tasks[task].q_count = h3_tasks[task].q_count;
        generic_tasks[task].interval_count = h3_tasks[task].interval_count;
        for (std::uint32_t interval = 0;
             interval < h3_tasks[task].interval_count; ++interval) {
            generic_tasks[task].allowed[interval] = {
                h3_tasks[task].allowed[interval].begin,
                h3_tasks[task].allowed[interval].end};
        }
    }
    const sageattention::descriptor generic_operation = {
        {1, geometry.sequence, geometry.sequence, geometry.heads,
         geometry.heads, geometry.head_dim, sageattention::layout::nhd},
        {sageattention::data_type::bf16, sageattention::data_type::bf16,
         sageattention::qk_mode::bf16, sageattention::pv_mode::bf16,
         sageattention::kernel_id::e33_bf16_qk_gfx12_d128},
        task_count};
    const sageattention::interval_plan generic_plan = {
        generic_tasks.data(), generic_tasks.size()};
    const std::size_t workspace_bytes =
        bf16_qk
            ? sageattention::workspace_size(generic_operation)
            : h3_vdn_sage_workspace_size(
                  geometry, h3_vdn_sage_pv_mode::bf16);
    if (!workspace_bytes) {
        std::fputs("unsupported benchmark descriptor\n", stderr);
        return EXIT_FAILURE;
    }
    std::vector<hip_bfloat16> q(elements), k(elements), v(elements),
        output(elements), warmup_output(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        const int qv = static_cast<int>(index % 251) - 125;
        const int kv = static_cast<int>((index * 17) % 239) - 119;
        const int vv = static_cast<int>((index * 7) % 227) - 113;
        q[index] = hip_bfloat16(static_cast<float>(qv) * 0.002f);
        k[index] = hip_bfloat16(static_cast<float>(kv) * 0.002f);
        v[index] = hip_bfloat16(static_cast<float>(vv) * 0.003f);
    }

    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    void *workspace = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    bool ok = hip_ok(hipMalloc(&dq, tensor_bytes), "hipMalloc(q)") &&
              hip_ok(hipMalloc(&dk, tensor_bytes), "hipMalloc(k)") &&
              hip_ok(hipMalloc(&dv, tensor_bytes), "hipMalloc(v)") &&
              hip_ok(hipMalloc(&dout, tensor_bytes), "hipMalloc(output)") &&
              hip_ok(hipMalloc(&workspace, workspace_bytes),
                     "hipMalloc(workspace)") &&
              hip_ok(hipStreamCreate(&stream), "hipStreamCreate") &&
              hip_ok(hipEventCreate(&start), "hipEventCreate(start)") &&
              hip_ok(hipEventCreate(&stop), "hipEventCreate(stop)");
    if (ok)
        ok = hip_ok(hipMemcpyAsync(dq, q.data(), tensor_bytes,
                                   hipMemcpyHostToDevice, stream),
                    "copy q") &&
             hip_ok(hipMemcpyAsync(dk, k.data(), tensor_bytes,
                                   hipMemcpyHostToDevice, stream),
                    "copy k") &&
             hip_ok(hipMemcpyAsync(dv, v.data(), tensor_bytes,
                                   hipMemcpyHostToDevice, stream),
                    "copy v") &&
             hip_ok(hipStreamSynchronize(stream), "input sync");

    const h3_vdn_sage_params params = {
        dq, dk, dv, dout, geometry, 1.0f / std::sqrt(128.0f),
        h3_vdn_sage_pv_mode::bf16, workspace, workspace_bytes, stream};
    const sageattention::params generic_params = {
        dq, dk, dv, dout, generic_operation, 1.0f / std::sqrt(128.0f),
        workspace, workspace_bytes, stream};
    const auto prepare = [&]() {
        return bf16_qk
                   ? sageattention::prepare_workspace(
                         generic_operation, generic_plan, workspace,
                         workspace_bytes, stream)
                   : h3_vdn_sage_prepare_workspace(
                         geometry, h3_vdn_sage_pv_mode::bf16, workspace,
                         workspace_bytes, stream);
    };
    const auto launch_prepared = [&]() {
        return bf16_qk ? sageattention::launch_prepared(generic_params)
                       : h3_vdn_sage_launch_prepared(params);
    };
    const auto launch_profiled = [&](h3_vdn_sage_profile *profile) {
        if (!bf16_qk) return h3_vdn_sage_launch_profiled(params, profile);
        sageattention::profile generic_profile = {};
        const hipError_t error =
            sageattention::launch_profiled(generic_params, &generic_profile);
        if (error == hipSuccess) {
            profile->q_quant_ms = generic_profile.q_quant_ms;
            profile->k_quant_ms = generic_profile.k_quant_ms;
            profile->attention_ms = generic_profile.attention_ms;
            profile->total_ms = generic_profile.total_ms;
        }
        return error;
    };
    double metadata_ms = 0.0;
    if (ok) {
        const auto metadata_start = std::chrono::steady_clock::now();
        ok = hip_ok(prepare(), "metadata prepare") &&
             hip_ok(hipStreamSynchronize(stream), "metadata sync");
        metadata_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - metadata_start)
                          .count();
    }
    for (int warmup = 0; warmup < 3 && ok; ++warmup)
        ok = hip_ok(launch_prepared(), "warm-up launch");
    if (ok) ok = hip_ok(hipStreamSynchronize(stream), "warm-up sync");
    if (ok) {
        ok = hip_ok(hipMemcpy(warmup_output.data(), dout, tensor_bytes,
                              hipMemcpyDeviceToHost),
                    "copy warm-up output");
    }
    /* The large D2H copy can let clocks fall. Re-warm immediately before the
     * consecutive profiling and measured runs. */
    if (ok)
        ok = hip_ok(launch_prepared(), "re-warm launch") &&
             hip_ok(hipStreamSynchronize(stream), "re-warm sync");

    h3_vdn_sage_profile profile = {};
    for (int profile_run = 0; profile_run < 3 && ok; ++profile_run)
        ok = hip_ok(launch_profiled(&profile), "profiled launch");

    std::vector<float> milliseconds;
    milliseconds.reserve(iterations);
    const auto wall_start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < iterations && ok;
         ++iteration) {
        ok = hip_ok(hipEventRecord(start, stream), "record start") &&
             hip_ok(launch_prepared(), "measured launch") &&
             hip_ok(hipEventRecord(stop, stream), "record stop") &&
             hip_ok(hipEventSynchronize(stop), "event sync");
        float elapsed = 0.0f;
        if (ok) ok = hip_ok(hipEventElapsedTime(&elapsed, start, stop),
                            "elapsed time");
        if (ok) milliseconds.push_back(elapsed);
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - wall_start)
                               .count();
    if (ok)
        ok = hip_ok(hipMemcpy(output.data(), dout, tensor_bytes,
                              hipMemcpyDeviceToHost),
                    "copy output");

    if (ok) {
        hipDeviceProp_t properties = {};
        ignore_hip_error(hipGetDeviceProperties(&properties, 0));
        std::sort(milliseconds.begin(), milliseconds.end());
        const float median = milliseconds[milliseconds.size() / 2];
        const float minimum = milliseconds.front();
        const float maximum = milliseconds.back();
        const std::uint64_t warmup_hash = hash_bf16(warmup_output);
        const std::uint64_t final_hash = hash_bf16(output);
        std::size_t non_finite = 0;
        for (const hip_bfloat16 value : output)
            if (!std::isfinite(static_cast<float>(value))) ++non_finite;
        std::printf(
            "H3 VDN %s: arch=%s S=%u H=%u D=%u tasks=%zu "
            "workspace=%.3fMiB metadata_cold_ms=%.3f iterations=%u "
            "q_quant_ms=%.3f k_quant_ms=%.3f attention_ms=%.3f "
            "profile_total_ms=%.3f gpu_ms median=%.3f min=%.3f "
            "max=%.3f wall_ms_per_iter=%.3f non_finite=%zu "
            "deterministic=%s hash=%016llx\n",
            bf16_qk ? "e33-bf16-qk-bf16-pv" : "e27-i8-qk-bf16-pv",
            properties.gcnArchName, geometry.sequence, geometry.heads,
            geometry.head_dim, task_count,
            static_cast<double>(workspace_bytes) / (1024.0 * 1024.0),
            metadata_ms, iterations, profile.q_quant_ms, profile.k_quant_ms,
            profile.attention_ms, profile.total_ms, median, minimum, maximum,
            wall_ms / iterations, non_finite,
            warmup_hash == final_hash ? "yes" : "no",
            static_cast<unsigned long long>(final_hash));
        ok = non_finite == 0 && warmup_hash == final_hash;
    }

    ignore_hip_error(hipEventDestroy(stop));
    ignore_hip_error(hipEventDestroy(start));
    ignore_hip_error(hipStreamDestroy(stream));
    ignore_hip_error(hipFree(workspace));
    ignore_hip_error(hipFree(dout));
    ignore_hip_error(hipFree(dv));
    ignore_hip_error(hipFree(dk));
    ignore_hip_error(hipFree(dq));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
