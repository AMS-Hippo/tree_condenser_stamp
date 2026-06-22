# Schema-1 release audit

**Date:** 2026-06-20  
**Package version:** 0.15.0  
**Encoded graph schema:** 1.0

## Contract status

`CURRENT_SCHEMA1_API.md` remains the frozen normative contract. The root,
package-facing, documentation, and uploaded copies are byte-identical and have
SHA-256:

```text
30a3005bbf0266ce8058cc6eb52fcd159b722b71b3f811231e91a2070c64b593
```

No contract text or public schema was changed. The implementation-only
interpretations previously agreed remain in `audit/SCHEMA_CLARIFICATIONS.md`.

## Source verification

The final source checks are:

```text
pytest -q -W error                         156 passed
  ordinary schema/general suite            144 passed
  BPE parity/protection suite                12 passed
python -m ruff check .                     passed
python -m ruff format --check .            51 files formatted
python -m compileall -q tree_coarsening tests benchmarks audit/run_notebooks.py
                                              passed
python audit/run_notebooks.py              3 passed
```

The suite covers malformed-input rejection, public and refit atomicity, exact
provenance, tuple/custom UIDs, read-only fitted artifacts, validation levels,
full and partial boundary decoding, generic contraction fuzzing, simulator
reproducibility, all six Star/BPE/Named stage orders, randomized pipelines,
deterministic output, and Python/Numba parity.

An additional audit sweep completed 180 randomized three-stage round trips and
compared the optimized Parametric Star transform with its retained sequential
oracle over 1,000 raw and 200 encoded randomized cases without a mismatch.

## BPE preservation

- The complete Numba kernel is byte-identical to v0.12.1.
- Twenty-three protected Python scoring, indexing, token, UID-rope, and compact
  contraction functions remain AST-identical to v0.12.1.
- Python and Numba learn identical ordered rules and event counts for all three
  score modes, including variable-geometry encoded inputs.
- Cross-generation histories match after normalizing only the schema-1
  model-ID namespace in output labels.
- The retained BPE benchmark shows no observed fit or transform regression.

No BPE source file was changed in this audit pass. Details and machine-readable
evidence are in `audit/BPE_PARITY_AND_PERFORMANCE.md` and its referenced JSON
files.

## Performance repairs outside BPE

### Parametric Star

Two schema-adapter costs were removed without changing rule semantics:

1. independent occurrences of one rule are applied in one generic batch;
2. conservatively independent ordered rules are applied in one fitted wave.

Interacting rules remain sequential. Complete graph differentials protect the
historical parent-before-child overlap policy on raw and encoded inputs.

Retained full-validation benchmarks show:

- 533 qualifying parents / 1,600 nodes: about 6.21 s to 0.067 s, or 93x;
- 200 unrelated rules / 801 nodes: about 1.88 s to 0.046 s, or 41x compared
  with the already-improved one-batch-per-rule adapter.

The cross-generation fixture reports schema-1 transform ratios of 0.82x,
0.85x, and 0.87x at 501, 2,501, and 5,001 raw nodes. Decode is mixed but
comparable across that sweep.

### Generic component ordering and Named Vertex

Repeated whole-tree scans in generic/Named component ordering were replaced by
sorting through one precomputed deterministic position map. This removes an
accidental `O(|V| * components)` path.

Against v0.12.1, retained full/default-validation ratios at 6,001 raw nodes are
0.716x for transform and 0.708x for decode; smaller cases are at parity or
faster apart from a roughly 6% decode difference at 601 nodes.

Reproducible details are in `audit/PARAMETRIC_STAR_BATCHING.md` and
`audit/AUTHORING_AND_GENERIC_PERFORMANCE.md`.

## Artifact and authoring audit

`Vocabulary` now enforces its documented read-only semantics with write-once
slots and a mapping proxy. `EncodingRule`, encoder, decoder, and method-specific
semantic state remain immutable from ordinary public use.

`RuleBasedEncoder` and `StructuralStageDecoder` centralize normalization,
lineage, exact-type construction, attachment rewiring, provenance, validation,
and full/partial decoding. The minimal conformance example supplies one rule
and one node-selection method. `NamedVertexCoarsener` uses this path and does
not engage with Star/BPE-specific machinery.

The current Named module is 200 physical lines versus 567 in v0.12.1. Line
count is not a semantic metric, but here the reduction is primarily duplicated
schema and decoder bookkeeping moved into shared infrastructure. Verification
is broader and more demanding; writing the simple method itself is not.

## Distribution verification

The final source distribution and universal wheel build successfully. The sdist
content audit rejects caches, build output, bytecode, and repository internals,
and verifies inclusion of the frozen contract, audit baselines, performance
evidence, tests, notebooks, and contributor guide. The complete 156-test suite
passes from an extracted sdist.

A fresh source-isolated wheel environment passes:

- import and package-version verification outside the repository;
- Star -> BPE -> Named exact raw round trip;
- lazy Star+BPE composition round trip;
- Python/Numba BPE history parity after one warm-up fit.

SHA-256 checksums are written beside the distributions.

## Deferred items

No release-blocking audit item remains. Only items explicitly deferred by
schema 1.0 remain deferred: fitted-artifact persistence/migration, pickle
guarantees, materialized composition, combined targeted partial decoding,
arbitrary raw edge-attribute restoration, deep-copy isolation of nested mutable
user values, and specialized optimization for adversarially deep exact-type
nesting.
