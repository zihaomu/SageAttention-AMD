#!/usr/bin/env python3
"""Host-only tests for the dependency-free sagectl development CLI."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import tempfile
import unittest
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
            "device,GPU use (%),GPU Memory Allocated (VRAM%),Card Series,GFX Version\r\n"
            "card0,2,0,AMD Example GPU,gfx1234\n"
            "card3,70,51,AMD Busy GPU,gfx9999\n"
        )
        self.assertEqual(devices[0]["index"], 0)
        self.assertEqual(devices[0]["gpu_use_percent"], 2)
        self.assertEqual(devices[0]["memory_use_percent"], 0)
        self.assertEqual(devices[0]["architecture"], "gfx1234")
        self.assertEqual(devices[1]["index"], 3)

    def test_version_and_id_parsing(self) -> None:
        hip, compiler = SAGECTL.parse_hipcc_version(
            "HIP version: 7.2.123\nAMD clang version 22.0.0git (example)\n"
        )
        self.assertEqual(hip, "7.2.123")
        self.assertEqual(compiler, "AMD clang 22.0.0git")
        self.assertEqual(SAGECTL.gpu_slug("AMD Radeon AI PRO R9700"), "r9700")
        self.assertEqual(SAGECTL.gpu_slug("AMD Instinct MI300X"), "mi300x")

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
        }
        passing = {
            "correctness": {
                "non_finite": 0,
                "deterministic": True,
                "guard_canaries": True,
                "cpu_reference": "pass",
            }
        }
        self.assertTrue(SAGECTL.trial_passes_gates(passing, gates))
        passing["correctness"]["guard_canaries"] = False
        self.assertFalse(SAGECTL.trial_passes_gates(passing, gates))

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
