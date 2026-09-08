#include "sage_attention.hpp"
#include "sage_attention_portable_reference.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kWarmupIterations = 3;
constexpr std::uint32_t kProfileIterations = 3;
constexpr long double kCpuReferenceOperationLimit = 50000000.0L;
constexpr std::size_t kGuardBytes = 4096;
constexpr unsigned char kGuardValue = 0xa5;

enum class benchmark_kernel {
    automatic_select,
    e27_gfx12_d128,
    portable_exact_reference,
};

bool hip_ok(hipError_t status, const char *operation) {
    if (status == hipSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 hipGetErrorString(status));
    return false;
}

void ignore_hip_error(hipError_t status) { (void)status; }

bool parse_u32(const char *text, std::uint32_t *result) {
    if (!text || !result) return false;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (!end || *end || !parsed || parsed > UINT32_MAX) return false;
    *result = static_cast<std::uint32_t>(parsed);
    return true;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t *result) {
    if (!result || (a && b > std::numeric_limits<std::size_t>::max() / a))
        return false;
    *result = a * b;
    return true;
}

bool load_plan(const char *path, std::vector<sageattention::q_task> *tasks) {
    if (!path || !tasks) return false;
    std::ifstream input(path);
    std::string magic;
    std::size_t count = 0;
    if (!(input >> magic >> count) || magic != "sage-interval-plan-v1" ||
        !count || count > UINT32_MAX) {
        std::fprintf(stderr, "invalid interval plan header: %s\n", path);
        return false;
    }
    try {
        tasks->assign(count, {});
    } catch (...) {
        std::fputs("cannot allocate interval plan\n", stderr);
        return false;
    }
    for (sageattention::q_task &task : *tasks) {
        if (!(input >> task.q_begin >> task.q_count >> task.interval_count) ||
            task.interval_count > sageattention::max_intervals_per_task) {
            std::fprintf(stderr, "invalid interval task in %s\n", path);
            return false;
        }
        for (std::uint32_t index = 0; index < task.interval_count; ++index) {
            if (!(input >> task.allowed[index].begin >> task.allowed[index].end)) {
                std::fprintf(stderr, "truncated interval task in %s\n", path);
                return false;
            }
        }
    }
    std::string trailing;
    if (input >> trailing) {
        std::fprintf(stderr, "unexpected trailing interval-plan data: %s\n",
                     path);
        return false;
    }
    return true;
}

std::uint64_t hash_bf16(const std::vector<hip_bfloat16> &values) {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (const hip_bfloat16 value : values) {
        hash ^= value.data;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool device_guards_intact(const void *allocation, const void *payload,
                          std::size_t payload_bytes) {
    std::vector<unsigned char> observed(kGuardBytes);
    const auto all_guard = [&observed]() {
        return std::all_of(observed.begin(), observed.end(),
                           [](unsigned char value) {
                               return value == kGuardValue;
                           });
    };
    if (hipMemcpy(observed.data(), allocation, kGuardBytes,
                  hipMemcpyDeviceToHost) != hipSuccess ||
        !all_guard()) {
        return false;
    }
    const auto *payload_bytes_pointer =
        static_cast<const unsigned char *>(payload);
    return hipMemcpy(observed.data(), payload_bytes_pointer + payload_bytes,
                     kGuardBytes, hipMemcpyDeviceToHost) == hipSuccess &&
           all_guard();
}

benchmark_kernel parse_kernel(const char *name, bool *valid) {
    *valid = true;
    if (std::string(name) == "automatic_select")
        return benchmark_kernel::automatic_select;
    if (std::string(name) == "e27_gfx12_d128")
        return benchmark_kernel::e27_gfx12_d128;
    if (std::string(name) == "portable_exact_reference")
        return benchmark_kernel::portable_exact_reference;
    *valid = false;
    return benchmark_kernel::automatic_select;
}

const char *kernel_name(benchmark_kernel kernel) {
    switch (kernel) {
        case benchmark_kernel::automatic_select:
            return "automatic_select";
        case benchmark_kernel::e27_gfx12_d128:
            return "e27_gfx12_d128";
        case benchmark_kernel::portable_exact_reference:
            return "portable_exact_reference";
    }
    return "invalid";
}

struct cpu_comparison {
    const char *status;
    float max_absolute;
    double relative_rmse;
};

cpu_comparison compare_with_cpu_reference(
    std::uint32_t sequence, std::uint32_t heads,
    std::uint32_t head_dimension,
    const std::vector<sageattention::q_task> &tasks,
    const std::vector<hip_bfloat16> &query,
    const std::vector<hip_bfloat16> &key,
    const std::vector<hip_bfloat16> &value,
    const std::vector<hip_bfloat16> &output, float scale) {
    const long double operation_count =
        static_cast<long double>(sequence) * sequence * heads *
        head_dimension * 2.0L;
    if (operation_count > kCpuReferenceOperationLimit)
        return {"skipped-large-workload", 0.0f, 0.0};

    std::vector<float> reference(output.size(), 0.0f);
    for (const sageattention::q_task &task : tasks) {
        for (std::uint32_t local_query = 0; local_query < task.q_count;
             ++local_query) {
            const std::uint32_t query_row = task.q_begin + local_query;
            for (std::uint32_t head = 0; head < heads; ++head) {
                const std::size_t query_base =
                    (static_cast<std::size_t>(query_row) * heads + head) *
                    head_dimension;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::uint32_t interval_index = 0;
                     interval_index < task.interval_count; ++interval_index) {
                    const sageattention::key_interval interval =
                        task.allowed[interval_index];
                    for (std::uint32_t key_row = interval.begin;
                         key_row < interval.end; ++key_row) {
                        const std::size_t key_base =
                            (static_cast<std::size_t>(key_row) * heads + head) *
                            head_dimension;
                        float dot = 0.0f;
                        for (std::uint32_t dimension = 0;
                             dimension < head_dimension; ++dimension) {
                            dot += static_cast<float>(
                                       query[query_base + dimension]) *
                                   static_cast<float>(key[key_base + dimension]);
                        }
                        maximum = std::max(maximum, dot * scale);
                    }
                }
                float denominator = 0.0f;
                for (std::uint32_t interval_index = 0;
                     interval_index < task.interval_count; ++interval_index) {
                    const sageattention::key_interval interval =
                        task.allowed[interval_index];
                    for (std::uint32_t key_row = interval.begin;
                         key_row < interval.end; ++key_row) {
                        const std::size_t key_base =
                            (static_cast<std::size_t>(key_row) * heads + head) *
                            head_dimension;
                        float dot = 0.0f;
                        for (std::uint32_t dimension = 0;
                             dimension < head_dimension; ++dimension) {
                            dot += static_cast<float>(
                                       query[query_base + dimension]) *
                                   static_cast<float>(key[key_base + dimension]);
                        }
                        const float probability =
                            std::exp(dot * scale - maximum);
                        denominator += probability;
                        for (std::uint32_t dimension = 0;
                             dimension < head_dimension; ++dimension) {
                            reference[query_base + dimension] +=
                                probability * static_cast<float>(
                                                  value[key_base + dimension]);
                        }
                    }
                }
                for (std::uint32_t dimension = 0;
                     dimension < head_dimension; ++dimension)
                    reference[query_base + dimension] /= denominator;
            }
        }
    }

    double error_squared = 0.0;
    double reference_squared = 0.0;
    float max_absolute = 0.0f;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const double expected = reference[index];
        const double observed = static_cast<float>(output[index]);
        const double error = observed - expected;
        error_squared += error * error;
        reference_squared += expected * expected;
        max_absolute = std::max(
            max_absolute, static_cast<float>(std::fabs(error)));
    }
    const double relative_rmse =
        std::sqrt(error_squared / std::max(reference_squared, 1.0e-30));
    const bool passed = max_absolute <= 0.005f && relative_rmse <= 0.05;
    return {passed ? "pass" : "fail", max_absolute, relative_rmse};
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 7) {
        std::fprintf(
            stderr,
            "usage: %s PLAN QUERY_SEQUENCE HEADS HEAD_DIM ITERATIONS KERNEL\n",
            argv[0]);
        return 2;
    }
    std::uint32_t sequence = 0;
    std::uint32_t heads = 0;
    std::uint32_t head_dimension = 0;
    std::uint32_t iterations = 0;
    bool kernel_valid = false;
    const benchmark_kernel selected_kernel =
        parse_kernel(argv[6], &kernel_valid);
    if (!parse_u32(argv[2], &sequence) || !parse_u32(argv[3], &heads) ||
        !parse_u32(argv[4], &head_dimension) ||
        !parse_u32(argv[5], &iterations) || !kernel_valid) {
        std::fputs("invalid benchmark argument\n", stderr);
        return 2;
    }

    std::vector<sageattention::q_task> tasks;
    if (!load_plan(argv[1], &tasks)) return 2;
    const sageattention::descriptor operation = {
        {1, sequence, sequence, heads, heads, head_dimension,
         sageattention::layout::nhd},
        {sageattention::data_type::bf16, sageattention::data_type::bf16,
         sageattention::qk_mode::symmetric_i8,
         sageattention::pv_mode::bf16,
         selected_kernel == benchmark_kernel::automatic_select
             ? sageattention::kernel_id::automatic_select
             : sageattention::kernel_id::e27_gfx12_d128},
        tasks.size()};
    const sageattention::interval_plan plan = {tasks.data(), tasks.size()};
    const bool portable =
        selected_kernel == benchmark_kernel::portable_exact_reference;
    const hipError_t support =
        portable
            ? sageattention::reference::query_portable_exact_support(
                  sequence, heads, head_dimension)
            : sageattention::query_support(operation);
    if (!hip_ok(sageattention::validate_interval_plan(operation, plan),
                "validate interval plan") ||
        !hip_ok(support, "query support")) {
        return EXIT_FAILURE;
    }

    std::size_t elements = 0;
    if (!checked_mul(sequence, heads, &elements) ||
        !checked_mul(elements, head_dimension, &elements)) {
        std::fputs("tensor size overflow\n", stderr);
        return EXIT_FAILURE;
    }
    const std::size_t tensor_bytes = elements * sizeof(hip_bfloat16);
    const std::size_t workspace_bytes =
        portable ? tasks.size() * sizeof(tasks[0])
                 : sageattention::workspace_size(operation);
    if (!workspace_bytes) {
        std::fputs("unsupported workspace descriptor\n", stderr);
        return EXIT_FAILURE;
    }

    std::vector<hip_bfloat16> query(elements), key(elements), value(elements),
        output(elements), warmup_output(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        const int qv = static_cast<int>(index % 251) - 125;
        const int kv = static_cast<int>((index * 17) % 239) - 119;
        const int vv = static_cast<int>((index * 7) % 227) - 113;
        query[index] = hip_bfloat16(static_cast<float>(qv) * 0.002f);
        key[index] = hip_bfloat16(static_cast<float>(kv) * 0.002f);
        value[index] = hip_bfloat16(static_cast<float>(vv) * 0.003f);
    }

    void *device_query = nullptr;
    void *device_key = nullptr;
    void *device_value = nullptr;
    void *device_output_allocation = nullptr;
    void *workspace_allocation = nullptr;
    void *device_output = nullptr;
    void *workspace = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    bool ok = hip_ok(hipMalloc(&device_query, tensor_bytes), "hipMalloc(query)") &&
              hip_ok(hipMalloc(&device_key, tensor_bytes), "hipMalloc(key)") &&
              hip_ok(hipMalloc(&device_value, tensor_bytes), "hipMalloc(value)") &&
              hip_ok(hipMalloc(&device_output_allocation,
                               tensor_bytes + 2 * kGuardBytes),
                     "hipMalloc(output)") &&
              hip_ok(hipMalloc(&workspace_allocation,
                               workspace_bytes + 2 * kGuardBytes),
                     "hipMalloc(workspace)") &&
              hip_ok(hipStreamCreate(&stream), "hipStreamCreate") &&
              hip_ok(hipEventCreate(&start), "hipEventCreate(start)") &&
              hip_ok(hipEventCreate(&stop), "hipEventCreate(stop)");
    if (ok) {
        device_output =
            static_cast<unsigned char *>(device_output_allocation) + kGuardBytes;
        workspace =
            static_cast<unsigned char *>(workspace_allocation) + kGuardBytes;
        ok = hip_ok(hipMemset(device_output_allocation, kGuardValue,
                              tensor_bytes + 2 * kGuardBytes),
                    "initialize output guards") &&
             hip_ok(hipMemset(workspace_allocation, kGuardValue,
                              workspace_bytes + 2 * kGuardBytes),
                    "initialize workspace guards");
    }
    if (ok) {
        ok = hip_ok(hipMemcpyAsync(device_query, query.data(), tensor_bytes,
                                   hipMemcpyHostToDevice, stream),
                    "copy query") &&
             hip_ok(hipMemcpyAsync(device_key, key.data(), tensor_bytes,
                                   hipMemcpyHostToDevice, stream),
                    "copy key") &&
             hip_ok(hipMemcpyAsync(device_value, value.data(), tensor_bytes,
                                   hipMemcpyHostToDevice, stream),
                    "copy value") &&
             hip_ok(hipStreamSynchronize(stream), "input sync");
    }

    const sageattention::params parameters = {
        device_query, device_key, device_value, device_output, operation,
        1.0f / std::sqrt(static_cast<float>(head_dimension)), workspace,
        workspace_bytes, stream};
    const auto launch_selected = [&]() {
        return portable
                   ? sageattention::reference::launch_portable_exact(
                         device_query, device_key, device_value, device_output,
                         static_cast<const sageattention::q_task *>(workspace),
                         tasks.size(), sequence, heads, head_dimension,
                         parameters.scale, stream)
                   : sageattention::launch_prepared(parameters);
    };
    double metadata_ms = 0.0;
    if (ok) {
        const auto metadata_start = std::chrono::steady_clock::now();
        const hipError_t prepare_error =
            portable
                ? hipMemcpyAsync(workspace, tasks.data(), workspace_bytes,
                                 hipMemcpyHostToDevice, stream)
                : sageattention::prepare_workspace(
                      operation, plan, workspace, workspace_bytes, stream);
        ok = hip_ok(prepare_error, "prepare workspace") &&
             hip_ok(hipStreamSynchronize(stream), "metadata sync");
        metadata_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - metadata_start)
                          .count();
    }
    for (std::uint32_t warmup = 0; warmup < kWarmupIterations && ok; ++warmup)
        ok = hip_ok(launch_selected(), "warm-up launch");
    if (ok) ok = hip_ok(hipStreamSynchronize(stream), "warm-up sync");
    if (ok)
        ok = hip_ok(hipMemcpy(warmup_output.data(), device_output, tensor_bytes,
                              hipMemcpyDeviceToHost),
                    "copy warm-up output");
    if (ok)
        ok = hip_ok(launch_selected(), "re-warm launch") &&
             hip_ok(hipStreamSynchronize(stream), "re-warm sync");

    sageattention::profile profile = {};
    for (std::uint32_t profile_run = 0;
         profile_run < kProfileIterations && ok; ++profile_run) {
        if (portable) {
            ok = hip_ok(hipEventRecord(start, stream), "profile start") &&
                 hip_ok(launch_selected(), "profiled launch") &&
                 hip_ok(hipEventRecord(stop, stream), "profile stop") &&
                 hip_ok(hipEventSynchronize(stop), "profile sync") &&
                 hip_ok(hipEventElapsedTime(&profile.attention_ms, start, stop),
                        "profile elapsed");
            profile.total_ms = profile.attention_ms;
        } else {
            ok = hip_ok(sageattention::launch_profiled(parameters, &profile),
                        "profiled launch");
        }
    }

    std::vector<float> milliseconds;
    milliseconds.reserve(iterations);
    const auto wall_start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < iterations && ok; ++iteration) {
        ok = hip_ok(hipEventRecord(start, stream), "record start") &&
             hip_ok(launch_selected(), "measured launch") &&
             hip_ok(hipEventRecord(stop, stream), "record stop") &&
             hip_ok(hipEventSynchronize(stop), "event sync");
        float elapsed = 0.0f;
        if (ok)
            ok = hip_ok(hipEventElapsedTime(&elapsed, start, stop),
                        "elapsed time");
        if (ok) milliseconds.push_back(elapsed);
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - wall_start)
                               .count();
    if (ok)
        ok = hip_ok(hipMemcpy(output.data(), device_output, tensor_bytes,
                              hipMemcpyDeviceToHost),
                    "copy output");

    if (ok) {
        hipDeviceProp_t properties = {};
        ok = hip_ok(hipGetDeviceProperties(&properties, 0),
                    "inspect visible device");
        std::sort(milliseconds.begin(), milliseconds.end());
        const float median = milliseconds[milliseconds.size() / 2];
        const float minimum = milliseconds.front();
        const float maximum = milliseconds.back();
        const std::uint64_t warmup_hash = hash_bf16(warmup_output);
        const std::uint64_t final_hash = hash_bf16(output);
        std::size_t non_finite = 0;
        for (const hip_bfloat16 element : output)
            if (!std::isfinite(static_cast<float>(element))) ++non_finite;
        const bool deterministic = warmup_hash == final_hash;
        const bool guard_canaries =
            device_guards_intact(device_output_allocation, device_output,
                                 tensor_bytes) &&
            device_guards_intact(workspace_allocation, workspace,
                                 workspace_bytes);
        const cpu_comparison cpu = compare_with_cpu_reference(
            sequence, heads, head_dimension, tasks, query, key, value, output,
            parameters.scale);
        std::printf(
            "{\"schema_version\":1,\"runner\":\"generic-interval-v1\","
            "\"kernel\":\"%s\",\"architecture\":\"%s\","
            "\"shape\":{\"sequence\":%u,"
            "\"heads\":%u,\"head_dimension\":%u},\"task_count\":%zu,"
            "\"workspace_bytes\":%zu,\"warmup_iterations\":%u,"
            "\"profile_iterations\":%u,\"measured_iterations\":%u,"
            "\"metrics_ms\":{\"metadata_cold\":%.6f,\"q_quant\":%.6f,"
            "\"k_quant\":%.6f,\"attention\":%.6f,\"profile_total\":%.6f,"
            "\"event_median\":%.6f,\"event_min\":%.6f,"
            "\"event_max\":%.6f,\"wall_per_iteration\":%.6f},"
            "\"correctness\":{\"non_finite\":%zu,\"deterministic\":%s,"
            "\"guard_canaries\":%s,\"output_hash\":\"%016llx\","
            "\"cpu_reference\":\"%s\","
            "\"cpu_max_abs\":%.9g,\"cpu_relative_rmse\":%.9g}}\n",
            kernel_name(selected_kernel), properties.gcnArchName, sequence,
            heads, head_dimension,
            tasks.size(), workspace_bytes, kWarmupIterations,
            kProfileIterations, iterations, metadata_ms, profile.q_quant_ms,
            profile.k_quant_ms, profile.attention_ms, profile.total_ms, median,
            minimum, maximum, wall_ms / static_cast<double>(iterations),
            non_finite, deterministic ? "true" : "false",
            guard_canaries ? "true" : "false",
            static_cast<unsigned long long>(final_hash), cpu.status,
            cpu.max_absolute, cpu.relative_rmse);
        ok = non_finite == 0 && deterministic && guard_canaries &&
             std::string(cpu.status) != "fail";
    }

    if (stop) ignore_hip_error(hipEventDestroy(stop));
    if (start) ignore_hip_error(hipEventDestroy(start));
    if (stream) ignore_hip_error(hipStreamDestroy(stream));
    if (workspace_allocation)
        ignore_hip_error(hipFree(workspace_allocation));
    if (device_output_allocation)
        ignore_hip_error(hipFree(device_output_allocation));
    if (device_value) ignore_hip_error(hipFree(device_value));
    if (device_key) ignore_hip_error(hipFree(device_key));
    if (device_query) ignore_hip_error(hipFree(device_query));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
