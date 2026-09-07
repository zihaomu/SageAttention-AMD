#ifndef H3_VDN_SAGE_HPP
#define H3_VDN_SAGE_HPP

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

/* v0.1 ships the first SageAttention-AMD specialization: the sparse interval
 * layout used by H3 VDN. The API owns no device memory and has no dependency
 * on H3 tensor types; callers provide raw device pointers, a HIP stream, and
 * reusable workspace. The public API is experimental until the generic
 * interval-plan interface is finalized. */

enum class h3_vdn_sage_pv_mode : std::uint32_t {
    bf16 = 0,
    fp16 = 1,
    fp8_e4m3 = 2,
};

struct h3_vdn_sage_geometry {
    std::uint32_t sequence;
    std::uint32_t heads;
    std::uint32_t head_dim;
    std::uint32_t video_start;
    std::uint32_t frames;
    std::uint32_t tokens_per_frame;
    std::uint32_t radius;
    std::uint32_t chunk;
    bool anchor_both;
};

struct h3_vdn_key_interval {
    std::uint32_t begin;
    std::uint32_t end;
};

/* Each task is one gfx12 Q super-tile of up to 32 rows. Each wave owns 16
 * rows. All valid rows in a task have the
 * same VDN mask. Short tasks preserve mask-class and non-aligned boundaries. */
struct h3_vdn_q_task {
    std::uint32_t q_begin;
    std::uint32_t q_count;
    std::uint32_t interval_count;
    h3_vdn_key_interval allowed[5];
};

struct h3_vdn_sage_params {
    const void *query_bf16;
    const void *key_bf16;
    const void *value_bf16;
    void *output_bf16;
    h3_vdn_sage_geometry geometry;
    float scale;
    h3_vdn_sage_pv_mode pv_mode;
    void *workspace;
    std::size_t workspace_bytes;
    hipStream_t stream;
};

struct h3_vdn_sage_profile {
    float q_quant_ms;
    float k_quant_ms;
    float attention_ms;
    float total_ms;
};

/* Geometry-only helpers are host functions and do not touch a HIP device. */
hipError_t h3_vdn_sage_validate_geometry(
    const h3_vdn_sage_geometry &geometry);

std::size_t h3_vdn_sage_task_count(
    const h3_vdn_sage_geometry &geometry);

hipError_t h3_vdn_sage_build_tasks(
    const h3_vdn_sage_geometry &geometry,
    h3_vdn_q_task *tasks,
    std::size_t task_capacity,
    std::size_t *task_count);

bool h3_vdn_sage_key_allowed(
    const h3_vdn_sage_geometry &geometry,
    std::uint32_t query,
    std::uint32_t key);

std::int8_t h3_vdn_sage_quantize_symmetric_i8(float value, float scale);

/* Returns zero on invalid geometry, unsupported mode, or size overflow. */
std::size_t h3_vdn_sage_workspace_size(
    const h3_vdn_sage_geometry &geometry,
    h3_vdn_sage_pv_mode mode);

/* Upload immutable geometry metadata into workspace. H3 should call this only
 * on a geometry cache miss, then use h3_vdn_sage_launch_prepared() for its 50
 * blocks. The copy is ordered on stream. */
hipError_t h3_vdn_sage_prepare_workspace(
    const h3_vdn_sage_geometry &geometry,
    h3_vdn_sage_pv_mode mode,
    void *workspace,
    std::size_t workspace_bytes,
    hipStream_t stream);

/* Asynchronous on params.stream and requires matching prepared metadata. */
hipError_t h3_vdn_sage_launch_prepared(const h3_vdn_sage_params &params);

/* Synchronous diagnostic path. Production dispatch should use the async API. */
hipError_t h3_vdn_sage_launch_profiled(
    const h3_vdn_sage_params &params,
    h3_vdn_sage_profile *profile);

/* Convenience cold path: prepare metadata and launch. It is useful for
 * standalone calls; cached H3 execution should use the two calls above. */
hipError_t h3_vdn_sage_launch(const h3_vdn_sage_params &params);

#endif
