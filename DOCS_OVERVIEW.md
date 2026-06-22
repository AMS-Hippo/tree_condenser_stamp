# Documentation map

The documentation is intentionally split by audience. The schema document is
normative; the other files explain, trace, or demonstrate it.

## Normative API contract

- `CURRENT_SCHEMA1_API.md` — frozen schema-1 contract. This is the source of
  truth for implementation and tests.
- `tree_coarsening_api.md` — byte-identical copy of the same contract for
  discoverability.
- `docs/tree_coarsening_api.md` — byte-identical copy used by documentation
  tooling and readers who look under `docs/` first.

The copies are guarded by tests and should not diverge. Edit the contract only
by first discussing the API change, then updating all copies identically and
recording the reason.

## User-facing package docs

- `README.md` — installation, raw input format, quick examples, and the public
  coarseners.
- `notebooks/` — small executable examples for individual coarseners.
- `benchmarks/README.md` — how to run the release performance notebooks.
- `experiments/README.md` — CARBANAK-specific legacy/project notebooks that
  require local data.

## Contributor and audit docs

- `CONTRIBUTING_COARSENERS.md` — how to implement a new coarsener without
  reimplementing schema mechanics.
- `CONTRACT_TRACEABILITY.md` — maps contract sections to implementation files
  and tests.
- `audit/` — release evidence, schema clarifications, protected BPE baselines,
  and benchmark/audit records. These are historical evidence, not a second API.
- `docs/edge_bpe_numba_experiment.md` — design notes for the optional optimized
  BPE backend.

## Practical rule

When documents seem to disagree, use this priority order:

1. `CURRENT_SCHEMA1_API.md`
2. tests and `CONTRACT_TRACEABILITY.md`
3. README and contributor guide
4. notebooks, benchmarks, experiments, and audit notes
