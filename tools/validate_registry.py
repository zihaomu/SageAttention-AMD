#!/usr/bin/env python3
"""Validate the dependency-free capability and benchmark registry."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"^[a-z0-9]+(?:[a-z0-9.-]*[a-z0-9])?$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ERRORS: list[str] = []


def error(path: Path, message: str) -> None:
    ERRORS.append(f"{path.relative_to(ROOT)}: {message}")


def require(record: dict[str, Any], path: Path, keys: set[str]) -> None:
    missing = sorted(keys - record.keys())
    if missing:
        error(path, f"missing required keys: {', '.join(missing)}")


def load_records(directory: str, id_key: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    records: dict[str, tuple[Path, dict[str, Any]]] = {}
    base = ROOT / directory
    for path in sorted(base.glob("*.json")):
        raw = path.read_text(encoding="utf-8")
        if "/home/" in raw or '"hostname"' in raw or '"username"' in raw:
            error(path, "contains host- or user-specific identity data")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            error(path, f"invalid JSON: {exc}")
            continue
        if not isinstance(record, dict):
            error(path, "top-level value must be an object")
            continue
        canonical = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        if raw != canonical:
            error(path, "must use canonical two-space JSON formatting")
        record_id = record.get(id_key)
        if not isinstance(record_id, str) or not ID_RE.fullmatch(record_id):
            error(path, f"{id_key} must be a stable lowercase ID")
            continue
        if record_id in records:
            error(path, f"duplicate {id_key}: {record_id}")
        records[record_id] = (path, record)
    if not records:
        error(base, "contains no JSON records")
    return records


def validate_platforms(records: dict[str, tuple[Path, dict[str, Any]]]) -> None:
    for record_id, (path, record) in records.items():
        require(record, path, {"schema_version", "platform_id", "status", "host", "gpu", "software"})
        if path.stem != record_id:
            error(path, "filename must match platform_id")
        if record.get("schema_version") != 1:
            error(path, "unsupported schema_version")
        if record.get("status") not in {"experimental", "validated", "retired"}:
            error(path, "invalid platform status")
        host = record.get("host", {})
        gpu = record.get("gpu", {})
        software = record.get("software", {})
        if not isinstance(host, dict) or not isinstance(gpu, dict) or not isinstance(software, dict):
            error(path, "host, gpu, and software must be objects")
            continue
        require(host, path, {"cpu_model", "architecture", "operating_system"})
        require(gpu, path, {"vendor", "model", "architecture", "vram_bytes", "wavefront_sizes"})
        require(software, path, {"rocm_version", "hip_version", "compiler"})
        if gpu.get("vendor") != "AMD":
            error(path, "gpu.vendor must be AMD")
        if not isinstance(gpu.get("vram_bytes"), int) or gpu.get("vram_bytes", 0) <= 0:
            error(path, "gpu.vram_bytes must be a positive integer")
        waves = gpu.get("wavefront_sizes")
        if not isinstance(waves, list) or not waves or any(w not in {32, 64} for w in waves):
            error(path, "gpu.wavefront_sizes must contain 32 and/or 64")


def validate_kernels(records: dict[str, tuple[Path, dict[str, Any]]]) -> None:
    for record_id, (path, record) in records.items():
        require(record, path, {"schema_version", "kernel_id", "status", "implementation", "dispatch", "algorithm", "validation"})
        if path.stem != record_id:
            error(path, "filename must match kernel_id")
        if record.get("schema_version") != 1:
            error(path, "unsupported schema_version")
        if record.get("status") not in {"experimental", "candidate", "supported", "retired"}:
            error(path, "invalid kernel status")
        implementation = record.get("implementation", {})
        dispatch = record.get("dispatch", {})
        if not isinstance(implementation, dict) or not isinstance(dispatch, dict):
            error(path, "implementation and dispatch must be objects")
            continue
        require(dispatch, path, {"gpu_architectures", "wavefront_size", "layouts", "batch_sizes", "head_dimensions", "input_dtypes", "output_dtypes", "mask_plan_classes"})
        for source in implementation.values():
            if not isinstance(source, str) or source.startswith("/") or not (ROOT / source).is_file():
                error(path, f"implementation path does not exist: {source!r}")
        for field in ("gpu_architectures", "layouts", "batch_sizes", "head_dimensions", "input_dtypes", "output_dtypes", "mask_plan_classes"):
            value = dispatch.get(field)
            if not isinstance(value, list) or not value:
                error(path, f"dispatch.{field} must be a non-empty list")
        if dispatch.get("wavefront_size") not in {32, 64}:
            error(path, "dispatch.wavefront_size must be 32 or 64")


def validate_workloads(records: dict[str, tuple[Path, dict[str, Any]]]) -> None:
    for record_id, (path, record) in records.items():
        require(record, path, {"schema_version", "workload_id", "description", "measurement_scope", "shape", "mask"})
        if path.stem != record_id:
            error(path, "filename must match workload_id")
        if record.get("schema_version") != 1:
            error(path, "unsupported schema_version")
        shape = record.get("shape", {})
        mask = record.get("mask", {})
        if not isinstance(shape, dict) or not isinstance(mask, dict):
            error(path, "shape and mask must be objects")
            continue
        require(shape, path, {"batch", "sequence", "heads", "head_dimension", "layout", "input_dtype", "output_dtype"})
        require(mask, path, {"plan_class"})
        for field in ("batch", "sequence", "heads", "head_dimension"):
            if not isinstance(shape.get(field), int) or shape.get(field, 0) <= 0:
                error(path, f"shape.{field} must be a positive integer")


def validate_results(
    records: dict[str, tuple[Path, dict[str, Any]]],
    platforms: dict[str, tuple[Path, dict[str, Any]]],
    kernels: dict[str, tuple[Path, dict[str, Any]]],
    workloads: dict[str, tuple[Path, dict[str, Any]]],
) -> None:
    for _, (path, record) in records.items():
        require(record, path, {"schema_version", "result_id", "recorded_at", "evidence_level", "measurement_scope", "git_commit", "source_dirty", "platform_id", "kernel_id", "workload_id", "session", "metrics_ms", "correctness"})
        if record.get("schema_version") != 1:
            error(path, "unsupported schema_version")
        if record.get("evidence_level") not in {"research", "candidate", "promotion"}:
            error(path, "invalid evidence_level")
        if not isinstance(record.get("git_commit"), str) or not COMMIT_RE.fullmatch(record["git_commit"]):
            error(path, "git_commit must be a full lowercase SHA-1")
        if record.get("evidence_level") in {"candidate", "promotion"} and record.get("source_dirty") is not False:
            error(path, "candidate and promotion evidence require source_dirty=false")
        platform_id = record.get("platform_id")
        kernel_id = record.get("kernel_id")
        workload_id = record.get("workload_id")
        if platform_id not in platforms:
            error(path, f"unknown platform_id: {platform_id}")
        if kernel_id not in kernels:
            error(path, f"unknown kernel_id: {kernel_id}")
        if workload_id not in workloads:
            error(path, f"unknown workload_id: {workload_id}")
        if platform_id not in platforms or kernel_id not in kernels or workload_id not in workloads:
            continue
        platform = platforms[platform_id][1]
        kernel = kernels[kernel_id][1]
        workload = workloads[workload_id][1]
        dispatch = kernel["dispatch"]
        shape = workload["shape"]
        if platform["gpu"]["architecture"] not in dispatch["gpu_architectures"]:
            error(path, "platform architecture is outside kernel dispatch")
        if dispatch["wavefront_size"] not in platform["gpu"]["wavefront_sizes"]:
            error(path, "kernel wavefront size is unsupported by platform")
        checks = (
            (shape["batch"], dispatch["batch_sizes"], "batch"),
            (shape["head_dimension"], dispatch["head_dimensions"], "head dimension"),
            (shape["layout"], dispatch["layouts"], "layout"),
            (shape["input_dtype"], dispatch["input_dtypes"], "input dtype"),
            (shape["output_dtype"], dispatch["output_dtypes"], "output dtype"),
            (workload["mask"]["plan_class"], dispatch["mask_plan_classes"], "mask plan"),
        )
        for value, allowed, label in checks:
            if value not in allowed:
                error(path, f"workload {label} is outside kernel dispatch")
        if record.get("measurement_scope") != workload.get("measurement_scope"):
            error(path, "measurement_scope differs from workload")
        session = record.get("session", {})
        metrics = record.get("metrics_ms", {})
        correctness = record.get("correctness", {})
        if not isinstance(session, dict) or not isinstance(metrics, dict) or not isinstance(correctness, dict):
            error(path, "session, metrics_ms, and correctness must be objects")
            continue
        require(session, path, {"physical_device_index", "command", "warmup_iterations", "measured_iterations"})
        require(metrics, path, {"profile_total", "event_median", "event_min", "event_max"})
        require(correctness, path, {"contract_test", "gpu_test", "isa_test", "non_finite", "deterministic"})
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or value < 0:
                error(path, f"metrics_ms.{key} must be non-negative")
        if all(isinstance(metrics.get(k), (int, float)) for k in ("event_min", "event_median", "event_max")):
            if not metrics["event_min"] <= metrics["event_median"] <= metrics["event_max"]:
                error(path, "event_min <= event_median <= event_max is required")


def validate_campaigns(
    records: dict[str, tuple[Path, dict[str, Any]]],
    platforms: dict[str, tuple[Path, dict[str, Any]]],
    kernels: dict[str, tuple[Path, dict[str, Any]]],
    workloads: dict[str, tuple[Path, dict[str, Any]]],
) -> None:
    for record_id, (path, record) in records.items():
        require(
            record,
            path,
            {
                "schema_version",
                "campaign_id",
                "status",
                "platform_id",
                "workload_ids",
                "baseline_kernel_id",
                "candidate_kernel_ids",
                "hypothesis",
                "objective",
                "search_space",
                "gates",
                "acceptance",
            },
        )
        if path.stem != record_id:
            error(path, "filename must match campaign_id")
        if record.get("schema_version") != 1:
            error(path, "unsupported schema_version")
        if record.get("status") not in {"experimental", "active", "complete", "retired"}:
            error(path, "invalid campaign status")
        platform_id = record.get("platform_id")
        baseline_id = record.get("baseline_kernel_id")
        candidate_ids = record.get("candidate_kernel_ids")
        workload_ids = record.get("workload_ids")
        if platform_id not in platforms:
            error(path, f"unknown platform_id: {platform_id}")
        if baseline_id not in kernels:
            error(path, f"unknown baseline_kernel_id: {baseline_id}")
        if not isinstance(candidate_ids, list) or not candidate_ids:
            error(path, "candidate_kernel_ids must be a non-empty list")
            candidate_ids = []
        for kernel_id in candidate_ids:
            if kernel_id not in kernels:
                error(path, f"unknown candidate kernel: {kernel_id}")
            if kernel_id == baseline_id:
                error(path, "baseline cannot also be a candidate")
        if not isinstance(workload_ids, list) or not workload_ids:
            error(path, "workload_ids must be a non-empty list")
            workload_ids = []
        for workload_id in workload_ids:
            if workload_id not in workloads:
                error(path, f"unknown workload: {workload_id}")
        if not isinstance(record.get("hypothesis"), str) or not record["hypothesis"].strip():
            error(path, "hypothesis must be a non-empty string")
        objective = record.get("objective", {})
        search_space = record.get("search_space", {})
        gates = record.get("gates", {})
        acceptance = record.get("acceptance", {})
        if not all(
            isinstance(value, dict)
            for value in (objective, search_space, gates, acceptance)
        ):
            error(path, "objective, search_space, gates, and acceptance must be objects")
            continue
        require(objective, path, {"metric", "direction", "measurement_scope"})
        if objective.get("metric") not in {
            "profile_total",
            "event_median",
            "wall_per_iteration",
        }:
            error(path, "unsupported campaign objective metric")
        if objective.get("direction") != "minimize":
            error(path, "only minimize campaign objectives are currently supported")
        require(search_space, path, {"kind", "kernel_ids"})
        if search_space.get("kind") != "registered-kernel-comparison":
            error(path, "unsupported campaign search-space kind")
        if search_space.get("kernel_ids") != candidate_ids:
            error(path, "search_space.kernel_ids must match candidate_kernel_ids")
        require(
            gates,
            path,
            {
                "require_cpu_reference",
                "require_guard_canaries",
                "require_determinism",
                "short_iterations",
                "confirmation_iterations",
                "confirmation_runs",
                "top_k_for_confirmation",
            },
        )
        for key in (
            "short_iterations",
            "confirmation_iterations",
            "confirmation_runs",
            "top_k_for_confirmation",
        ):
            if not isinstance(gates.get(key), int) or gates.get(key, 0) <= 0:
                error(path, f"gates.{key} must be a positive integer")
        if isinstance(candidate_ids, list) and isinstance(
            gates.get("top_k_for_confirmation"), int
        ) and gates["top_k_for_confirmation"] > len(candidate_ids):
            error(path, "top_k_for_confirmation exceeds candidate count")
        require(acceptance, path, {"minimum_speedup", "stopping_rule"})
        if not isinstance(acceptance.get("minimum_speedup"), (int, float)) or acceptance.get(
            "minimum_speedup", 0
        ) <= 0:
            error(path, "acceptance.minimum_speedup must be positive")

        if (
            platform_id not in platforms
            or baseline_id not in kernels
            or any(kernel_id not in kernels for kernel_id in candidate_ids)
            or any(workload_id not in workloads for workload_id in workload_ids)
        ):
            continue
        platform = platforms[platform_id][1]
        for kernel_id in [baseline_id, *candidate_ids]:
            kernel = kernels[kernel_id][1]
            dispatch = kernel["dispatch"]
            if platform["gpu"]["architecture"] not in dispatch["gpu_architectures"]:
                error(path, f"platform architecture is outside {kernel_id} dispatch")
            if dispatch["wavefront_size"] not in platform["gpu"]["wavefront_sizes"]:
                error(path, f"platform wavefront size is outside {kernel_id} dispatch")
            for workload_id in workload_ids:
                workload = workloads[workload_id][1]
                shape = workload["shape"]
                checks = (
                    (shape["batch"], dispatch["batch_sizes"], "batch"),
                    (shape["head_dimension"], dispatch["head_dimensions"], "head dimension"),
                    (shape["layout"], dispatch["layouts"], "layout"),
                    (shape["input_dtype"], dispatch["input_dtypes"], "input dtype"),
                    (shape["output_dtype"], dispatch["output_dtypes"], "output dtype"),
                    (workload["mask"]["plan_class"], dispatch["mask_plan_classes"], "mask plan"),
                )
                for value, allowed, label in checks:
                    if value not in allowed:
                        error(path, f"{workload_id} {label} is outside {kernel_id} dispatch")
                if objective.get("measurement_scope") != workload.get(
                    "measurement_scope"
                ):
                    error(path, f"objective scope differs from {workload_id}")


def main() -> int:
    platforms = load_records("registry/platforms", "platform_id")
    kernels = load_records("registry/kernels", "kernel_id")
    workloads = load_records("registry/workloads", "workload_id")
    campaigns = load_records("benchmarks/campaigns", "campaign_id")
    results = load_records("benchmarks/results", "result_id")
    validate_platforms(platforms)
    validate_kernels(kernels)
    validate_workloads(workloads)
    validate_campaigns(campaigns, platforms, kernels, workloads)
    validate_results(results, platforms, kernels, workloads)
    if ERRORS:
        for message in ERRORS:
            print(f"error: {message}", file=sys.stderr)
        return 1
    print(
        "registry valid: "
        f"{len(platforms)} platform(s), {len(kernels)} kernel(s), "
        f"{len(workloads)} workload(s), {len(campaigns)} campaign(s), "
        f"{len(results)} result(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
