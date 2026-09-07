## Change

Describe the kernel, platform, workload, planner, API, or documentation change.

## Evidence

- Platform ID:
- Kernel ID:
- Workload ID:
- Source commit used for measurement:
- Correctness result:
- Performance result:

## Checklist

- [ ] Dispatch boundaries are explicit and unsupported inputs fail clearly.
- [ ] Registry records match the implementation and documentation.
- [ ] `make metadata-check` passes.
- [ ] I ran the applicable host, GPU, canary, determinism, and ISA checks.
- [ ] New performance claims have an append-only benchmark result record.
- [ ] Operator and downstream model-quality results are reported separately.
- [ ] I documented checks or platforms that were not available.
- [ ] I did not add generated artifacts, credentials, or machine-specific paths.
