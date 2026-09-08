# Optimization campaigns

Campaigns bind one platform, one or more workloads, an exact baseline, a
bounded candidate set, an objective, correctness gates, and a stopping rule.
They define an experiment; they are not benchmark evidence.

The first supported search-space kind is `registered-kernel-comparison`. It
validates the end-to-end funnel before compile-time parameter generation is
added. Generated reports and result drafts stay under ignored
`build/sagectl/`. Accepted measurements are copied to `benchmarks/results/`
only after review and the evidence requirements in `doc/benchmarking.md`.

Run `make metadata-check` after adding or changing a campaign.
