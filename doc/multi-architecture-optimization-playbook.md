# Multi-architecture optimization playbook

Status: active implementation target for the v0.2 development series. The
command-line interface described below is partially implemented; consult the
live ledger before relying on a command.

## Implementation status

Last updated: 2026-09-08.

This table is the live implementation ledger for the playbook. A stage is
marked complete only after its commands and stated validation have passed in
the repository.

| Stage | Status | Acceptance record |
|---|---|---|
| Playbook and repository entry points | Complete | Playbook added and linked from README and ROADMAP |
| Machine probe and platform draft | Complete | `make tool-test`; `sagectl doctor` compiled and ran a minimal HIP kernel on physical GPU 4 and generated a canonical ignored draft |
| Registry-driven generic benchmark | Complete | Registry workloads generate canonical plans consumed by `bench_interval`; small dense and H3 production runs passed |
| Portable exact GPU baseline | Complete for gfx1201/D128 onboarding | Small dense CPU reference, non-finite, determinism, and output/workspace canaries passed; other architectures still require their own evidence |
| Campaign schema and validator | Complete | One registered-kernel smoke campaign is canonical and metadata-validated |
| Search funnel and result draft | Complete for registered-kernel comparison | Static gates, short ranking, paired baseline, three-round confirmation, local result drafts, and report implemented |
| Descriptor-based specialization registration | Complete for E27 | Host descriptor and device capability checks resolve through a registration table; public E27/H3 declarations are unchanged |
| gfx1201 end-to-end reproduction | Complete as research evidence | Full E27 GPU suite passed and the registry-driven production run reproduced the accepted output hash |
| First non-gfx12 campaign | Blocked on hardware | Requires an explicitly selected target machine |

## Goal

A contributor or coding agent on a new AMD machine should be able to clone the
repository, identify the platform, establish a correct local baseline, search
a bounded kernel-configuration space, and produce reviewable evidence without
rewriting repository infrastructure.

The output is not one universally "best AMD kernel." A best implementation is
valid only for a declared support tuple:

```text
(GPU architecture, wave size, ROCm/toolchain, layout, dtype, batch,
 head dimension, mask-plan class, numerical mode, workload, metric scope)
```

Results from one tuple may seed another campaign, but they are not evidence for
it. The dispatcher selects among separately validated specializations.

## Current starting point

The repository already has the pieces that must remain stable during this
work:

- a model-independent ordered-interval plan API;
- an explicit kernel selector and unsupported-input behavior;
- the E27 gfx12/wave32/D128 specialization;
- an unchanged H3 VDN compatibility API;
- platform, kernel, workload, and immutable result records;
- contract, GPU correctness, canary, determinism, ISA, and metadata gates.

It does not yet provide zero-touch onboarding for another architecture. The
default build architecture is gfx1201, GPU tests require gfx1201/wave32, and
the production benchmark is still an H3-shaped executable. Machine discovery,
a portable exact GPU baseline, parameterized workloads, campaign execution,
and result generation must be added before automated cross-architecture
search is credible.

## Target operator experience

The intended interface is a development-only `sagectl` command. `doctor` is
available now; the other examples describe the target workflow and must not be
treated as implemented until the live ledger marks them complete:

```sh
./tools/sagectl doctor --gpu 0
./tools/sagectl baseline --gpu 0 \
  --platform-draft build/sagectl/platforms/<platform-id>.json \
  --workload ordered-interval-d128
./tools/sagectl search --gpu 0 \
  --campaign benchmarks/campaigns/gfx-example-d128.json
./tools/sagectl report --campaign gfx-example-d128
```

`doctor` must never guess which physical device is safe to benchmark. The user
or automation selects one explicitly after checking occupancy.

## Machine onboarding

### 1. Probe the environment

`sagectl doctor` should collect and print:

- GPU marketing name, GFX architecture, VRAM, and supported wave sizes;
- device count and the explicitly selected physical device;
- ROCm, HIP runtime, compiler, operating system, and host architecture;
- whether a minimal HIP program can compile and execute;
- current GPU utilization and memory allocation;
- relevant matrix-instruction capabilities when they can be established by a
  compile-and-disassemble probe.

It should generate a draft `registry/platforms/<platform-id>.json`, but it must
not include hostnames, usernames, serial numbers, absolute paths, or transient
device indices. A tool-generated record remains experimental until reviewed.
The current implementation writes this draft under
`build/sagectl/platforms/`. `sagectl baseline --platform-draft` can qualify it
before review; if no compatible exact kernel record exists, it also emits a
platform-scoped kernel draft under `build/sagectl/kernels/`.

### 2. Establish an exact baseline

Every new architecture needs a correctness oracle and a local performance
baseline before approximate kernels are considered. The project should add a
portable, exact HIP reference path that favors clarity and coverage over peak
speed. Small CPU references remain useful for independent contract checks;
large production shapes should compare against the exact GPU path.

The reference path must have its own kernel registry entry and must not be
silently presented as an optimized default.

### 3. Run qualification before search

The baseline must pass descriptor validation, boundary shapes, reference
comparison, non-finite detection, output and workspace guard canaries, and
determinism. A campaign cannot begin if the baseline or environment probe
fails.

## Workload model

The benchmark executable should consume workload registry records instead of
embedding one geometry in C++ source. Initial reusable workload classes should
cover:

- dense;
- causal;
- sliding window;
- caller-provided ordered intervals;
- a later caller-provided block plan.

Head dimensions 64, 128, and 256, layouts, batch sizes, sequence boundaries,
and numerical modes require separate records and evidence. H3 VDN remains a
valuable production workload and adapter, but it is not the benchmark API.

Input generation must be deterministic and identified in the result. The
runner must state whether planning, metadata upload, allocation, quantization,
or layout conversion is inside the measured scope.

## Optimization campaigns

A campaign is a versioned, reviewable definition of one optimization question.
It binds a platform, workloads, baseline, objective, search space, gates, and
stopping policy. A future campaign record may resemble:

```json
{
  "schema_version": 1,
  "campaign_id": "gfx-example-ordered-interval-d128",
  "platform_id": "example-platform-id",
  "workload_ids": [
    "ordered-interval-d128-small",
    "ordered-interval-d128-production"
  ],
  "baseline_kernel_id": "exact-portable-d128",
  "objective": {
    "metric": "event_median_ms",
    "measurement_scope": "operator-including-qk-quantization"
  },
  "search_space": {
    "q_tile_rows": [16, 32, 64],
    "k_tile_rows": [32, 64, 128],
    "waves_per_block": [1, 2, 4],
    "pipeline_stages": [1, 2],
    "vector_width_bytes": [4, 8, 16]
  },
  "gates": {
    "require_correctness": true,
    "require_zero_scratch": true,
    "top_k_for_confirmation": 3,
    "confirmation_runs": 3
  }
}
```

The values above are illustrative, not recommended settings for an actual
architecture. Architecture-specific matrix modes and resource limits must come
from that campaign's probe and hypothesis.

Generated binaries, raw profiler databases, and transient measurements belong
in an ignored run directory. Accepted campaign definitions and immutable
result records belong in version control. Rejected algorithmic directions
must remain in an optimization ledger with enough evidence to prevent an agent
from repeating them without a new hypothesis.

## Search funnel

The first implementation should use a deterministic grid plus successive
halving rather than an opaque optimizer:

1. Generate compile-time variants from a bounded campaign search space.
2. Compile and disassemble every viable candidate.
3. Reject missing required instructions, unexpected private scratch, and
   resource use outside campaign limits.
4. Run small correctness, boundary, canary, and determinism tests.
5. Run short measurements on surviving candidates.
6. Advance only the top candidates to production shapes and repeated sessions.
7. Re-run the baseline in the same session before claiming a speedup.
8. Emit a machine-readable report and a draft immutable result record.

Compile-time specialization is preferred to adding runtime branches to a hot
kernel. A fundamentally different algorithm is a distinct candidate source or
kernel family, not another unexplained tuning parameter.

## Automation boundary

| Safe to automate | Requires explicit review or real evidence |
|---|---|
| Environment and capability probes | Numerical modes and error tolerances |
| Draft platform metadata | Consumer model-quality acceptance |
| Compile matrices and disassembly | Whether an approximate mode becomes a default |
| Resource, correctness, and canary gates | Extrapolation to another GPU or ROCm version |
| Idle-device preflight and benchmark scheduling | Selection between materially different algorithms |
| Result JSON generation and ranking | Promotion to supported status |

Occupancy checks can detect an obviously busy GPU, but they cannot prove that
a shared machine will remain uncontended. Promotion measurements require a
controlled session and repeatability.

## Agent operating contract

An agent running a campaign must:

1. Read the repository instructions, platform/kernel/workload records, and the
   relevant optimization ledger before editing.
2. State one hypothesis, objective, acceptance threshold, and stopping rule.
3. Preserve the accepted kernel while candidates are being explored.
4. Keep the E27 device path and H3 compatibility declarations unchanged unless
   the task explicitly targets them.
5. Use an explicitly selected, confirmed-idle physical GPU and never terminate
   another user's work.
6. Run correctness and static gates before performance measurements.
7. Record rejected candidates as well as winners.
8. Never combine cross-platform timings into a kernel speedup claim.
9. Leave generated artifacts, machine identity, and credentials out of Git.
10. Produce a diff, exact commands, result records, and remaining limitations
    for review.

Agents may propose new algorithms, but automation should only rank candidates
inside a declared campaign. The framework must not turn an unbounded source
rewrite into an apparently reproducible parameter search.

## Dispatcher and promotion

Each accepted implementation needs a stable kernel ID and an explicit dispatch
domain. Registration should provide capability matching, workspace size,
launch entry points, numerical contract, and evidence status without forcing
architecture checks into every hot kernel.

A candidate may become an automatic selection only when:

- its full dispatch tuple is represented in the registry;
- unsupported combinations fail clearly or use an explicitly documented exact
  fallback;
- full operator gates and required ISA checks pass on every claimed platform;
- benchmark results are repeatable from a clean commit;
- the dispatcher and documentation agree with the registry;
- any consumer-owned model-quality gate required for default use passes.

The existing E27 specialization stays independently selectable while other
architecture families are added. A faster candidate on another platform does
not supersede E27 evidence.

## Hardware CI model

Hosted CI should continue to validate metadata, formatting, host contracts,
and any architecture compile checks that do not require a GPU. Trusted
self-hosted AMD runners should provide two levels:

- pull-request or manual quick gates: build, ISA, small correctness, canaries,
  and determinism;
- scheduled or promotion gates: controlled occupancy, production workloads,
  repeated baselines, and immutable result generation.

Untrusted fork code must not execute on persistent self-hosted GPU runners with
repository secrets. Hardware jobs should use trusted commits, explicit runner
labels, minimal permissions, and preferably ephemeral workers.

## Implementation sequence

1. Add `sagectl doctor` and canonical platform-record generation.
2. Parameterize the generic benchmark around workload records.
3. Add the portable exact GPU reference and cross-architecture correctness
   harness.
4. Define and validate campaign records and local run artifacts.
5. Implement compile, static-gate, correctness, short-run, and confirmation
   stages.
6. Generate draft result records and human-readable comparisons.
7. Refactor specialization registration behind the existing public dispatcher
   without changing E27 or the H3 compatibility API.
8. Validate the entire workflow on gfx1201 by reproducing the accepted E27
   result.
9. Use the first available non-gfx12 AMD machine as the first true portability
   and search campaign.

The proposed milestone for steps 1 through 6 is
`v0.2.0-alpha.1: portable onboarding and optimization campaigns`.

## Validation log

### 2026-09-08: machine probe and platform draft

- `make tool-test`: four host-only parser, ID, record-safety, and path-safety
  tests passed.
- `./tools/sagectl doctor --gpu 4 --platform-id
  r9700-gfx1201-rocm7.2.3-ubuntu24.04 --require-idle --force`: passed with a
  compiled minimal HIP kernel on an idle gfx1201 device.
- The generated canonical draft matched the selected GPU's architecture,
  wave32 mode, VRAM, ROCm 7.2.3, HIP version, compiler, OS, CPU, and device
  count. The draft contains no hostname, username, serial number, GUID,
  absolute path, or physical device index.
- The draft is written below `build/sagectl/platforms/`, which remains ignored
  by Git until a human reviews and copies it into the platform registry.

### 2026-09-08: registry-driven interval benchmark

- `make tool-test metadata-check`: six host tests passed and the registry
  validated with two workloads.
- `sagectl plan` generated two canonical tasks for the model-independent dense
  S35/H1/D128 smoke workload and 167 tasks for the existing H3 production
  workload.
- `sagectl benchmark` ran both workloads on explicitly selected physical GPU
  4 after an idle-device check and emitted ignored research-result drafts.
- The generic runner's H3 production output hash was
  `d8fccefb0ea98938`, identical to the accepted H3 benchmark record. This
  confirms plan generation and generic launch parity; the dirty development
  tree means the new timing is research data and is not committed evidence.

### 2026-09-08: portable exact baseline

- Added a development-only BF16/FP32 portable interval reference that is not
  reachable from automatic public dispatch.
- `sagectl baseline` on the dense S35/H1/D128 workload passed the CPU reference
  with max absolute error `0.000121757388` and relative RMSE
  `0.00168825021` after BF16 output conversion.
- Non-finite count was zero, repeated output was deterministic, and both sides
  of output and workspace allocations retained their guard canaries.
- The kernel registry entry deliberately claims only the tested
  gfx1201/wave32/D128 tuple. Portability of the source is not evidence for a
  second architecture.

### 2026-09-08: first campaign funnel

- Added and validated the
  `gfx1201-ordered-interval-d128-smoke` registered-kernel campaign.
- `sagectl search` passed metadata, host contract, and E27 ISA gates; ran a
  short comparison; then re-ran the exact baseline before each of three paired
  confirmations.
- On this intentionally tiny smoke workload, E27's paired median speedup was
  `0.9180x` relative to the scalar portable baseline. The campaign correctly
  returned no recommendation because the configured `1.0x` threshold was not
  met. This is a workflow test, not a production performance conclusion.
- All trial records and the report remained under ignored `build/sagectl/` and
  were labelled research evidence because the implementation tree was dirty.

### 2026-09-08: E27 descriptor registration and regression

- Existing host descriptor and current-device capability checks now resolve
  through an internal specialization registration table. The public kernel ID,
  E27 device loop, generic API declarations, and H3 compatibility declarations
  were not changed.
- Clean rebuild, contract tests, required INT8/BF16 WMMA ISA checks, registry
  validation, and the full gfx1201 GPU correctness/canary/determinism suite
  passed on explicitly selected physical GPU 4.
- The registry-driven H3 production run retained output hash
  `d8fccefb0ea98938`, matching the accepted result. Its development timing is
  research-only and was not promoted to immutable evidence.
- `BUILD_DIR` is now overridable, and `sagectl` builds each target below a
  separate GFX-specific directory to prevent stale cross-architecture objects.
- A doctor-generated platform draft can be passed directly to
  `sagectl baseline`; the tool emits a matching local portable-kernel draft if
  the reviewed registry does not yet contain one.
- The generic runner and portable reference cross-compiled successfully for
  gfx942 into its isolated build directory. This is compile-only evidence: no
  gfx942 runtime, correctness, ISA, or performance support is claimed.

## Definition of done for a new machine

A machine is ready to search when it can, from a clean clone:

- generate a reviewable platform draft without personal identifiers;
- build and pass the exact reference qualification suite;
- load at least one generic workload record;
- execute a bounded campaign without editing the runner;
- rank candidates against a same-session baseline;
- reproduce the winner in repeated confirmation runs;
- emit registry-valid draft evidence while keeping the source tree clean.

Until all of these hold, results from that machine are research evidence, not
a new supported architecture.

## Current implementation limits

- Architecture-specific objects are isolated below
  `build/sagectl/architectures/<gfx>/`; switching GFX targets cannot reuse an
  object compiled for another architecture.
- The campaign runner currently compares registered kernel implementations.
  Compile-time candidate generation and resource extraction are the next
  search-space extension.
- The portable exact source is designed without wave intrinsics, but only its
  gfx1201/wave32/D128 record has real hardware evidence in this repository.
- No non-gfx12 support claim can be added until the workflow runs on explicitly
  identified hardware for that tuple.
