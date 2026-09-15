#!/usr/bin/env python3
"""Host-only tests for the dependency-free sagectl development CLI."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import math
import uuid
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader("sagectl", str(ROOT / "tools" / "sagectl"))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
SAGECTL = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(SAGECTL)


class SagectlTests(unittest.TestCase):
    def test_parse_rocm_smi(self) -> None:
        devices = SAGECTL.parse_rocm_smi(
            "device,GPU use (%),GPU Memory Allocated (VRAM%),PCI Bus,Card Series,Node ID,GFX Version\r\n"
            "card0,2,0,0000:83:00.0,AMD Example GPU,14,gfx1234\n"
            "card3,70,51,0000:63:00.0,AMD Busy GPU,8,gfx9999\n"
        )
        self.assertEqual(devices[0]["index"], 0)
        self.assertEqual(devices[0]["gpu_use_percent"], 2)
        self.assertEqual(devices[0]["memory_use_percent"], 0)
        self.assertEqual(devices[0]["architecture"], "gfx1234")
        self.assertEqual(devices[1]["index"], 3)
        self.assertEqual(devices[0]["node_id"], 14)
        self.assertEqual(devices[1]["pci_bus"], "0000:63:00.0")
        self.assertEqual(
            SAGECTL.select_smi_device_for_hip_ordinal(devices, 0)["index"], 3
        )
        self.assertEqual(
            SAGECTL.select_smi_device_for_hip_ordinal(devices, 1)["index"], 0
        )
        self.assertEqual(
            SAGECTL.require_benchmark_device_idle(
                devices, 0, "gfx9999", 80, 60
            )["pci_bus"],
            "0000:63:00.0",
        )
        with self.assertRaises(SAGECTL.SagectlError):
            SAGECTL.require_benchmark_device_idle(
                devices, 0, "gfx9999", 10, 60
            )
        with self.assertRaises(SAGECTL.SagectlError):
            SAGECTL.select_smi_device_for_hip_ordinal(devices, 2)

    def test_version_and_id_parsing(self) -> None:
        hip, compiler = SAGECTL.parse_hipcc_version(
            "HIP version: 7.2.123\nAMD clang version 22.0.0git (example)\n"
        )
        self.assertEqual(hip, "7.2.123")
        self.assertEqual(compiler, "AMD clang 22.0.0git")
        self.assertEqual(SAGECTL.gpu_slug("AMD Radeon AI PRO R9700"), "r9700")
        self.assertEqual(SAGECTL.gpu_slug("AMD Instinct MI300X"), "mi300x")

    def test_idle_wait_accepts_sampling_tail_but_stays_bounded(self) -> None:
        busy = SAGECTL.parse_rocm_smi(
            "device,GPU use (%),GPU Memory Allocated (VRAM%),PCI Bus,Card Series,Node ID,GFX Version\n"
            "card7,37,0,0000:e3:00.0,AMD Example GPU,12,gfx1201\n"
        )
        idle = SAGECTL.parse_rocm_smi(
            "device,GPU use (%),GPU Memory Allocated (VRAM%),PCI Bus,Card Series,Node ID,GFX Version\n"
            "card7,2,0,0000:e3:00.0,AMD Example GPU,12,gfx1201\n"
        )
        with mock.patch.object(
            SAGECTL, "inspect_rocm_smi_devices", side_effect=[busy, idle]
        ), mock.patch.object(SAGECTL.time, "sleep") as sleep:
            selected, polls = SAGECTL.wait_for_benchmark_device_idle(
                "rocm-smi", "test", 0, "gfx1201", 5, 1, 2, 0.25
            )
        self.assertEqual(selected["pci_bus"], "0000:e3:00.0")
        self.assertEqual(polls, 2)
        sleep.assert_called_once_with(0.25)

        with mock.patch.object(
            SAGECTL, "inspect_rocm_smi_devices", return_value=busy
        ), mock.patch.object(SAGECTL.time, "sleep") as sleep:
            with self.assertRaises(SAGECTL.SagectlError):
                SAGECTL.wait_for_benchmark_device_idle(
                    "rocm-smi", "test", 0, "gfx1201", 5, 1, 1, 0.25
                )
        self.assertEqual(sleep.call_count, 1)

    def test_platform_record_excludes_transient_identity(self) -> None:
        record = SAGECTL.make_platform_record(
            probe={
                "name": "AMD Instinct MI300X",
                "architecture": "gfx942:sramecc+:xnack-",
                "vram_bytes": "206158430208",
                "wavefront_size": "64",
            },
            device_count=8,
            rocm="7.2.0",
            hip="7.2.0",
            compiler="AMD clang 22.0.0git",
            operating_system="Ubuntu 24.04.1 LTS",
            os_slug="ubuntu24.04",
        )
        self.assertEqual(
            record["platform_id"], "mi300x-gfx942-rocm7.2.0-ubuntu24.04"
        )
        self.assertEqual(record["gpu"]["wavefront_sizes"], [64])
        serialized = str(record).lower()
        self.assertNotIn("hostname", serialized)
        self.assertNotIn("username", serialized)
        self.assertNotIn("device_index", serialized)

    def test_generated_files_cannot_escape_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "draft.json"
            with self.assertRaises(SAGECTL.SagectlError):
                SAGECTL.ensure_inside_repository(outside)

    def test_dense_workload_plan(self) -> None:
        workload = {
            "shape": {"sequence": 35},
            "mask": {"plan_class": "ordered-interval-v1", "generator": "dense-v1", "q_tile_rows": 32},
        }
        tasks = SAGECTL.generate_interval_tasks(workload)
        self.assertEqual(
            tasks,
            [
                {"q_begin": 0, "q_count": 32, "intervals": [(0, 35)]},
                {"q_begin": 32, "q_count": 3, "intervals": [(0, 35)]},
            ],
        )

    def test_h3_production_plan_matches_existing_task_count(self) -> None:
        workload = {
            "shape": {"sequence": 5338},
            "mask": {
                "plan_class": "ordered-interval-v1",
                "planner": "h3-vdn-v1",
                "video_start": 986,
                "frames": 17,
                "tokens_per_frame": 256,
                "radius": 1,
                "chunk": 5,
                "anchor_both": True,
            },
        }
        tasks = SAGECTL.generate_interval_tasks(workload)
        self.assertEqual(len(tasks), 167)
        self.assertEqual(tasks[0], {"q_begin": 0, "q_count": 32, "intervals": [(0, 5338)]})
        self.assertEqual(tasks[-1]["q_begin"] + tasks[-1]["q_count"], 5338)

    def test_campaign_gate_evaluation(self) -> None:
        gates = {
            "require_cpu_reference": True,
            "require_guard_canaries": True,
            "require_determinism": True,
            "max_paired_relative_rmse": 0.00361012,
            "min_paired_cosine": 0.999993587,
        }
        passing = {
            "correctness": {
                "non_finite": 0,
                "deterministic": True,
                "guard_canaries": True,
                "cpu_reference": "pass",
                "paired_reference": "pass",
                "paired_relative_rmse": 0.001,
                "paired_cosine": 0.999999,
                "paired_non_finite": 0,
            }
        }
        self.assertTrue(SAGECTL.trial_passes_gates(passing, gates))
        passing["correctness"]["guard_canaries"] = False
        self.assertFalse(SAGECTL.trial_passes_gates(passing, gates))
        passing["correctness"]["guard_canaries"] = True
        passing["correctness"]["paired_relative_rmse"] = 0.004
        self.assertFalse(SAGECTL.trial_passes_gates(passing, gates))
        passing["correctness"]["paired_relative_rmse"] = 0.001
        passing["correctness"]["paired_cosine"] = 0.99
        self.assertFalse(SAGECTL.trial_passes_gates(passing, gates))
        passing["correctness"]["paired_cosine"] = math.nan
        self.assertFalse(SAGECTL.trial_passes_gates(passing, gates))
        passing["correctness"]["paired_cosine"] = 0.999999
        passing["correctness"]["paired_non_finite"] = False
        self.assertFalse(SAGECTL.trial_passes_gates(passing, gates))
        passing["correctness"]["paired_non_finite"] = 0
        passing["correctness"]["non_finite"] = False
        self.assertFalse(SAGECTL.trial_passes_gates(passing, gates))
        passing["correctness"]["non_finite"] = 0
        self.assertEqual(
            SAGECTL.objective_metric_ms({"metrics_ms": {"event_median": 1}}, "event_median"),
            1.0,
        )
        for invalid in (True, 0, -1.0, math.nan, math.inf, "1.0"):
            with self.assertRaises(SAGECTL.SagectlError):
                SAGECTL.objective_metric_ms(
                    {"metrics_ms": {"event_median": invalid}}, "event_median"
                )
        invalid_json_path = (
            ROOT / "build" / f"invalid-nonfinite-{uuid.uuid4().hex}.json"
        )
        with self.assertRaises(SAGECTL.SagectlError):
            SAGECTL.write_canonical_json(
                invalid_json_path, {"metric": math.nan}, force=False
            )
        self.assertFalse(invalid_json_path.exists())

    def test_architecture_builds_are_isolated(self) -> None:
        self.assertEqual(
            SAGECTL.architecture_build_directory("gfx942:sramecc+:xnack-"),
            Path("build/sagectl/architectures/gfx942-sramecc-xnack"),
        )

    def test_portable_kernel_draft_is_platform_scoped(self) -> None:
        platform_record = {
            "gpu": {
                "architecture": "gfx942",
                "wavefront_sizes": [64],
            }
        }
        workload_record = {
            "shape": {
                "batch": 1,
                "head_dimension": 128,
                "layout": "NHD",
                "input_dtype": "bf16",
                "output_dtype": "bf16",
            },
            "mask": {"plan_class": "ordered-interval-v1"},
        }
        record = SAGECTL.make_portable_kernel_record(
            platform_record, workload_record
        )
        self.assertEqual(
            record["kernel_id"], "exact-portable-bf16-gfx942-d128"
        )
        self.assertEqual(record["dispatch"]["gpu_architectures"], ["gfx942"])
        self.assertEqual(record["dispatch"]["wavefront_size"], 64)


if __name__ == "__main__":
    unittest.main()
