# Coarsener authoring and generic-path performance audit

## Scope

This audit checks the two costs most likely to be hidden by a stricter schema:

1. whether a simple coarsener must implement occurrence geometry, provenance,
   lineage, rewiring, and decoding itself; and
2. whether the shared schema machinery causes material transform/decode
   regressions compared with v0.12.1.

The comparisons are evidence about the current implementation, not changes to
the frozen schema-1 API.

## Authoring surface

An ordinary method can subclass `RuleBasedEncoder`, provide an ordered
`EncodingRule`, and implement only `select_contractions(graph, rule)`. The
shared path performs:

- raw normalization and encoded-input validation;
- stage lineage and fitting-size updates;
- deterministic component ordering;
- `CompositeType` construction;
- UID/site concatenation and attachment rewiring;
- output renumbering and validation;
- full and partial exact decoding.

`tests/test_authoring_surface.py` contains a complete edge coarsener whose method
logic is one rule plus one small selection method. `NamedVertexCoarsener` is the
production example. Its current module is 200 physical lines versus 567 in
v0.12.1; line count is only a rough proxy, but the removed code is primarily
method-local schema/decoder bookkeeping now shared by every rule-based method.

A specialized algorithm may still subclass `TreeEncoder` directly. Edge BPE
does so to preserve its compact incremental pair index and overlap machinery;
the generic authoring path does not force the optimized kernel into a less
suitable abstraction.

## Named Vertex cross-generation performance

`benchmarks/benchmark_named_cross_generation.py` runs the current schema-1 and
v0.12.1 implementations in isolated subprocesses, uses each generation's full
or default validation, and verifies exact raw round trips and equal encoded
node counts before accepting timings. Values are medians of five runs from
`audit/named_cross_generation_benchmark.json`.

| Raw nodes | Encoded nodes | Transform schema-1 / old | Decode schema-1 / old |
|---:|---:|---:|---:|
| 601 | 201 | 0.916x | 1.055x |
| 3,001 | 1,001 | 0.892x | 0.943x |
| 6,001 | 2,001 | 0.716x | 0.708x |

Lower is faster. The initial schema-1 implementation had an accidental
`O(|V| * components)` ordering loop and was roughly 2.1x slower at 6,001 nodes.
Sorting each component through one precomputed position map removes that
regression. Current transform is at parity or faster across the retained sweep;
decode is within about 6% at the smallest case and faster in the larger cases.

## Parametric Star cross-generation performance

`benchmarks/benchmark_parametric_star_cross_generation.py` learns exactly one
`P -> C` family, runs each generation in an isolated subprocess with full or
default validation, and verifies exact raw round trips and equal encoded node
counts. Values are medians of five runs from
`audit/parametric_star_cross_generation_benchmark.json`.

| Raw nodes | Encoded nodes | Transform schema-1 / old | Decode schema-1 / old |
|---:|---:|---:|---:|
| 501 | 301 | 0.820x | 1.007x |
| 2,501 | 1,501 | 0.846x | 1.177x |
| 5,001 | 3,001 | 0.875x | 0.842x |

Current transform is faster throughout this retained sweep. Decode is mixed but
comparable: effectively equal at 501 nodes, about 18% slower at 2,501, and about
16% faster at 5,001. The algorithm-focused batching benchmarks separately guard
against the severe occurrence- and rule-count scaling regressions repaired in
this audit.

## Edge BPE

BPE is not routed through `RuleBasedEncoder`. Its Numba kernel remains
byte-identical to v0.12.1, 23 protected Python algorithm functions remain
AST-identical, Python/Numba rule histories agree, and the retained benchmark
shows no observed regression. See `audit/BPE_PARITY_AND_PERFORMANCE.md`.

## Assessment

The strict schema increases the work required in the shared framework and in a
coarsener's conformance tests. It does not increase the amount of schema code a
simple method author must write; the ordinary path is shorter and more
centralized than before. Optimized methods retain an escape hatch for their
internal state machine while still producing the same occurrence schema at the
boundary.
