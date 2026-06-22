# Schema 1.0 implementation clarifications

This file records implementation interpretations only. It does not modify
`CURRENT_SCHEMA1_API.md`, whose SHA-256 digest is frozen in
`audit/FROZEN_SCHEMA.sha256`.

1. `CompositeType` validates the normative semantic form and therefore accepts
   any acyclic component-parent order when constructed directly. Package-produced
   graph occurrences are additionally validated to be parent-before-child, as
   required by section 10.2.
2. Complete stage reversal always uses the boundary-expansion behavior required
   by section 20, regardless of the partial-decoding `boundary_policy` argument.
3. `validate=False` skips optional eager scans only. Structural checks needed to
   avoid unsafe rewiring or invented provenance still run, following section 26.
4. A lazy combined artifact has a synthetic inspection `model_id`; graph lineage
   records continue to contain only the atomic component stage IDs that actually
   own exact types.
5. Deterministic traversal ties are resolved from occurrence semantics (time,
   matching label, provenance order, and exact type), not from NetworkX node
   keys.  Edge BPE retains the pre-refactor raw-input ordering and uses a
   structural ordering of encoded provenance UIDs so downstream overlap choices
   remain stable without reintroducing key dependence.
6. A stage record with no visible contraction does not by itself switch Edge BPE
   to encoded-occurrence tie ordering. When every visible occurrence is still a
   one-site base type, BPE retains the historical raw-input ordering. This fixes
   lineage-only overlap drift at the schema adapter without changing the compact
   BPE algorithm or its optimized backend.
