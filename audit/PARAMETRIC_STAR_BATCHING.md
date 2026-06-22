# Parametric Star batching audit

## Problem

The first schema-1 adapter applied every qualifying sibling contraction through
one generic graph rebuild. With many independent occurrences of one rule, that
made transform effectively quadratic. The first repair batched all
vertex-disjoint occurrences of one rule, but still rebuilt the graph once for
every learned rule. A corpus with many unrelated label pairs therefore retained
a second avoidable `O(r * |V|)` adapter cost.

Neither problem belongs to the Parametric Star algorithm. They were scheduling
costs around the generic schema contraction engine.

## Current scheduling

The encoder now batches at two levels:

1. **Within one rule:** a deterministic parent-before-child greedy scan chooses
   the same nonoverlapping occurrences as the former sequential traversal, then
   applies them in one generic contraction batch.
2. **Across independent rules:** ordered rules are partitioned into conservative
   waves. Rules share a wave only when an earlier rule cannot remove or create a
   label used by a later rule and their selected forests cannot become adjacent
   through the earlier contraction. All contractions in one wave are applied by
   one graph rebuild.

Rules that can affect a later matching view start a new wave and retain exact
rule-by-rule temporal semantics. The wave partition is computed once when the
fitted encoder is constructed. The public API, rule records, output labels,
fitting sizes, exact types, attachment maps, provenance, lineage, and decoder
are unchanged.

## Semantic verification

- `tests/test_parametric_star_schema1.py` protects the historical
  parent-before-child self-overlap policy.
- `tests/test_parametric_star_batching.py` retains the former sequential
  transform as a test-only oracle and compares complete NetworkX graphs,
  including graph/node/edge metadata, over randomized raw inputs.
- The same file differentially checks encoded inputs whose labels and exact
  geometry were produced by an upstream stage.
- It proves that 40 unrelated rules use one wave and that interacting rules
  remain in separate temporal waves.
- Additional audit sweeps compared 1,000 randomized raw cases and 200 randomized
  encoded cases against the sequential oracle with no mismatch.
- Both benchmark scripts reject timing results unless the optimized and
  reference paths produce equal occurrence semantics.

## Many occurrences of one rule

`benchmarks/benchmark_parametric_star_batching.py` builds one learned `P -> C`
rule with many independent qualifying parents. Both paths use full public
validation. Values are medians of three runs from
`audit/parametric_star_batching_benchmark.json`.

| Qualifying parents | Input nodes | Batched seconds | One rebuild per occurrence | Speed-up |
|---:|---:|---:|---:|---:|
| 50 | 151 | 0.0045 | 0.0540 | 11.9x |
| 100 | 301 | 0.0094 | 0.2114 | 22.5x |
| 200 | 601 | 0.0200 | 0.8831 | 44.2x |
| 400 | 1,201 | 0.0439 | 3.6622 | 83.4x |
| 533 | 1,600 | 0.0668 | 6.2087 | 93.0x |

## Many unrelated rules

`benchmarks/benchmark_parametric_star_many_rules.py` gives each learned rule a
distinct parent and child label. The comparison path already has the first fix
(one batch per rule); it isolates the additional benefit of independent-rule
waves. Both paths use full public validation. Values are medians of three runs
from `audit/parametric_star_many_rules_benchmark.json`.

| Rules | Input nodes | Wave seconds | One batch per rule | Speed-up | Waves |
|---:|---:|---:|---:|---:|---:|
| 25 | 101 | 0.0049 | 0.0336 | 6.8x | 1 |
| 50 | 201 | 0.0087 | 0.1146 | 13.2x | 1 |
| 100 | 401 | 0.0212 | 0.4511 | 21.2x | 1 |
| 200 | 801 | 0.0458 | 1.8839 | 41.1x | 1 |

Wall-clock values are environment-specific. The durable regression properties
are structural: one graph rebuild per vertex-disjoint occurrence batch, and one
rebuild per conservative independent-rule wave rather than per rule.
