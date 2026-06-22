# Schema-1 contract traceability

This matrix maps the frozen `CURRENT_SCHEMA1_API.md` contract to the current
implementation and reusable verification. It is an audit aid, not a replacement
for the normative text.

| Contract area | Primary implementation | Principal verification |
|---|---|---|
| §§3–6 occurrence model, raw schema, normalization, encoded fields | `schema.py`, `structural.py`, `provenance.py`, `validation.py` | `test_schema_primitives.py`, `test_contract_malformed.py`, `test_validation_regressions.py` |
| §7 stage lineage | `schema.append_stage`, `schema.pop_latest_stage`, `validation.py` | `test_cross_stage_matrix.py`, `test_contract_malformed.py` |
| §§8–9 fitting sizes and stage vocabulary | `encoder.EncodingRule`, read-only `encoder.Vocabulary`, `schema.fit_corpus_fitting_sizes` | `test_schema_primitives.py`, `test_api_and_failure_atomicity.py`, `test_artifact_aware_stage_validation.py` |
| §10 exact structural types | `structural.CompositeType` and derived geometry functions | `test_schema_primitives.py`, `test_contract_malformed.py` |
| §§11–13 attachments and generic contraction | `contraction.py` | `test_generic_contraction_and_partial_decode.py`, `test_generic_contraction_fuzz.py` |
| §14 edge and sibling primitives | `contraction.apply_mixed_contraction_batch` | `test_generic_contraction_and_partial_decode.py`, Star/BPE method tests |
| §§15–17 rules, encoders, decoders | `encoder.py`, `decoder.py`, `stage_decoder.py` | `test_artifact_aware_stage_validation.py`, `test_authoring_surface.py` |
| §§18–20 expansion, partial decode, full reversal | `stage_decoder.py` | `test_generic_contraction_and_partial_decode.py`, `test_partial_decoding_recursive_boundary.py`, `test_cross_stage_matrix.py` |
| §21 public coarsener API and fit atomicity | `coarsener.py` | `test_api_and_failure_atomicity.py`, `test_fit_atomicity.py` |
| §22 Parametric Star | `coarseners/parametric_star.py` | `test_parametric_star_schema1.py`, raw/encoded differential `test_parametric_star_batching.py`, `audit/PARAMETRIC_STAR_BATCHING.md` |
| §23 Edge BPE | `coarseners/edge_bpe.py`, protected `edge_bpe_numba.py` | `test_edge_bpe_schema1.py`, `test_edge_bpe_parity_and_protection.py`, `audit/BPE_PARITY_AND_PERFORMANCE.md` |
| §24 Named Vertex | `coarseners/named_vertices.py` | `test_named_vertex_schema1.py`, `audit/AUTHORING_AND_GENERIC_PERFORMANCE.md` |
| §25 lazy composition | `compose.py` | `test_composition.py`, `test_cross_stage_matrix.py` |
| §26 validation modes | `schema.normalize_validation_level`, `validation.py`, operation-local checks | `test_contract_malformed.py`, `test_validation_regressions.py`, `test_api_and_failure_atomicity.py` |
| §27 determinism | `validation.deterministic_node_order`, method-specific ordering | key-renaming tests, stage matrix, randomized pipeline, Python/Numba parity |
| §28 errors | `exceptions.py` and concept-specific raises throughout | malformed, selection, stage-order, composition, and configuration tests |
| §29 nonmutation and failure atomicity | graph copies in public operations; `TreeCoarsener` fit snapshots | `test_api_and_failure_atomicity.py`, `test_fit_atomicity.py`, boundary-raise tests |
| §30 persistence/versioning | schema version rejection implemented; artifact persistence intentionally deferred | `test_contract_malformed.py::test_encoded_graph_rejects_unsupported_schema_version` |
| §31 contributor obligations | shared authoring surface plus method and pipeline suites | `CONTRIBUTING_COARSENERS.md`, `test_authoring_surface.py`, method/conformance suites, `audit/AUTHORING_AND_GENERIC_PERFORMANCE.md` |
| §32 locked decisions | frozen contract hash and byte-identical public copies | `test_documentation_sync.py`, `audit/FROZEN_SCHEMA.sha256` |

## Cross-cutting evidence

- `tests/test_randomized_pipeline.py` runs all six orderings of Star, BPE, and
  Named stages over randomized corpora, including partial latest-stage decoding.
- `tests/test_generic_contraction_fuzz.py` exercises randomized contractible
  forests and simultaneous disjoint batches.
- `tests/test_uid_contract.py` covers tuple and custom hashable UIDs without
  scalar/collection coercion; `tests/test_simulators.py` restores deterministic
  simulator and round-trip coverage.
- The three notebooks under `notebooks/` are executable schema-1 syntax
  examples; notebook execution is part of the release audit.
- `audit/SCHEMA_CLARIFICATIONS.md` records implementation interpretations that
  did not modify the frozen API.
