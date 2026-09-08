# Contributing

SageAttention AMD accepts architecture ports, kernel-shape specializations,
correctness improvements, benchmark evidence, and documentation fixes. Every
support or performance claim must identify the hardware, software, shape, and
measurement scope that produced it.

## Before starting

Open an issue for a new architecture, public API change, numerical-mode change,
or large optimization direction. Small fixes and new benchmark records can go
directly to a pull request. Search the optimization ledger before retrying a
previously rejected experiment.

## Development flow

1. Create a focused branch from `main`.
2. Add or update the relevant files under `registry/`.
3. Implement the smallest specialization or planner change that covers the
   declared dispatch domain.
4. Run the applicable checks.
5. Add benchmark evidence only after testing a clean commit on an idle device.
6. Complete the pull-request checklist and call out untested platforms.

Use concise commits that separate mechanical metadata changes from device-code
changes where practical.

## Adding a platform

Create `registry/platforms/<platform-id>.json`. A platform describes a
reproducible machine class: GPU model and architecture, VRAM, host CPU and OS,
ROCm/HIP/compiler versions, and wave sizes. Do not include a hostname, username,
serial number, GUID, or transient device index.

## Adding a kernel specialization

Create `registry/kernels/<kernel-id>.json` before claiming support. Declare:

- architecture and wave-size constraints;
- data types, tensor layout, batch and head dimensions;
- tile and quantization geometry;
- supported mask-plan classes;
- implementation sources and validation status.

The dispatcher must reject inputs outside that domain. Generic dispatch should
select specialized kernels; it should not force every specialization into one
branch-heavy implementation.

## Adding a workload or benchmark result

Workloads live in `registry/workloads/` and describe shape plus mask semantics.
Results live in `benchmarks/results/` and refer to stable platform, kernel, and
workload IDs. Follow [the benchmark protocol](doc/benchmarking.md). Historical
results are append-only.

## Adding an optimization campaign

Create `benchmarks/campaigns/<campaign-id>.json` after its platform, exact
baseline, candidate kernels, and workloads exist. State the hypothesis,
objective, search space, gates, acceptance threshold, and stopping rule. Run
`make metadata-check`, then execute the campaign on an explicitly selected idle
GPU with `./tools/sagectl search --gpu <index> --campaign <campaign-id>`.

Generated trials and reports under `build/sagectl/` are research artifacts.
Do not copy one into `benchmarks/results/` until the source commit is clean and
the benchmark evidence requirements are satisfied.

## Required checks

Metadata and documentation:

```sh
make metadata-check
```

Host planner or API logic:

```sh
make contract-test
```

gfx12 device code, using an idle physical device:

```sh
make contract-test
H3_PHYSICAL_GPU=4 make gpu-test
make isa
H3_PHYSICAL_GPU=4 make bench
```

Use the actual available device index; `4` is only an example. State every
check not run and why.

## Review policy

Maintainers review numerical behavior, dispatch boundaries, reproducibility,
and evidence independently. A speedup that weakens correctness or downstream
quality remains experimental. See [the support policy](doc/support-policy.md).
