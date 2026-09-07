# Support and compatibility policy

## Status levels

- `experimental`: API, numerical behavior, or dispatch coverage may change;
  operator evidence can exist while downstream quality is unresolved.
- `candidate`: the specialization passes its operator gates on declared
  platforms and is being evaluated for stable integration.
- `supported`: dispatch bounds, tests, benchmark reproducibility, documentation,
  and required downstream quality gates are complete.
- `retired`: kept for provenance or compatibility but not selected by default.

## What a support claim covers

A support claim applies only to the tuple declared by a kernel registry entry
and the validated platform records linked from benchmark evidence. Sharing a
marketing GPU family or a GFX prefix does not automatically extend support.

ROCm updates can change code generation, runtime behavior, and performance.
New versions require at least build, ISA, correctness, and representative
benchmark revalidation before they are added to a platform profile.

## API stability

Until version 1.0, public APIs are experimental unless explicitly marked
stable. Breaking changes require release notes and a migration path once an API
has a downstream consumer outside this repository.
