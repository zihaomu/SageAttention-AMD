# Agent instructions

## Project mission

Build an evidence-backed AMD attention kernel library that keeps fast,
shape-specific implementations while presenting a stable, model-independent
dispatch and planning layer. A result on one GPU, ROCm version, workload, or
shape is not evidence for another.

User instructions and platform safety rules take precedence over this file.
More specific `AGENTS.md` files add rules for their directory.

## Source of truth

- `registry/platforms/` defines reproducible hardware and software profiles.
- `registry/kernels/` defines the exact dispatch domain of each kernel.
- `registry/workloads/` defines benchmark geometry and mask semantics.
- `benchmarks/results/` contains immutable measured evidence.
- `doc/h3-vdn-sageattention-optimization.md` records rejected experiments and
  must be checked before repeating an optimization direction.
- README tables summarize these records; they are not the primary evidence.

Never infer support or performance across architectures, wave sizes, data
types, layouts, masks, or head dimensions. Add a registry entry and evidence
for each new combination.

## Required workflow

1. Inspect `git status`, the relevant registry entries, and scoped agent
   instructions before editing.
2. Preserve unrelated user changes. Keep changes limited to the requested
   kernel, dispatcher, metadata, or documentation boundary.
3. State the optimization hypothesis and success metric in the PR or
   optimization ledger before adding a non-trivial kernel experiment.
4. Keep model-specific mask construction in adapters or planners. Keep hot
   kernels specialized and branch-light.
5. Run the checks proportional to the change. Metadata-only changes require
   `make metadata-check`. Host logic requires `make contract-test`. Device
   code requires contract, GPU correctness, canaries, determinism, and ISA
   checks on every claimed architecture.
6. Performance claims require a new result record produced from a clean source
   commit. Do not replace or edit a historical result to make a regression
   disappear.

## GPU and benchmark safety

- Inspect device occupancy before selecting a GPU. Never assume device 0 is
  idle and never terminate another user's process.
- Select a physical GPU explicitly with `H3_PHYSICAL_GPU=<index>` for current
  test recipes. Do not run competing work on the measurement device.
- Record the platform, kernel, workload, commit, command, warm-up count,
  iteration count, and correctness outcome with every performance result.
- Compare results only under the same workload and measurement scope. Label
  cross-platform numbers separately; do not present them as a kernel speedup.
- Treat operator correctness, operator performance, and downstream model
  quality as separate gates. A fast operator does not become a default when a
  downstream quality gate fails.

## Promotion gates

A kernel/profile may move from `experimental` to `supported` only when:

- its dispatch domain is explicit and unsupported inputs fail clearly;
- host contracts, reference comparison, boundary shapes, non-finite checks,
  guard canaries, and determinism pass;
- required instructions are present in generated ISA;
- benchmark evidence is repeatable on every claimed platform;
- public API, registry, documentation, and downstream policy agree;
- any required application-level quality gates pass.

Rejected experiments stay in the ledger with their measurements. Revisit one
only when a new hypothesis explains why the earlier limiting factor changed.

## Repository hygiene

- Do not commit build products, profiler databases, model weights, generated
  media, credentials, hostnames, or user-specific absolute paths.
- Use relative repository paths in documentation and scripts.
- Keep the runtime dependency-free from Python, PyTorch, ATen, Triton, and
  CUDA unless a future public design explicitly changes that contract. Python
  is allowed for development-time metadata validation.
- Prefer append-only benchmark evidence and small, reviewable commits.
