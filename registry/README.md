# Capability registry

The registry makes every support and performance statement addressable. IDs
are stable, lowercase, hyphen-separated strings and must not contain hostnames,
usernames, serial numbers, or transient device indices.

## Records

- `platforms/`: reproducible host, GPU, and software profiles;
- `kernels/`: specialization dispatch domains and numerical contracts;
- `workloads/`: benchmark shapes and attention-mask semantics;
- `../benchmarks/results/`: measured sessions joining the three IDs above.

JSON is used so CI and reporting tools can consume the data without an extra
parser dependency. Run `make metadata-check` after editing any record.

Platform records describe a validated configuration rather than every possible
machine with the same GPU. A new ROCm version or materially different host may
use a new platform ID. Kernel IDs change when dispatch or numerical behavior
changes materially. Workload IDs change when geometry, layout, mask semantics,
or measurement scope changes.
