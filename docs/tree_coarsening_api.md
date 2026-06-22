# Tree Coarsening Normative API — Schema 1.0

**Specification status:** current handoff contract  
**Encoded graph schema version:** `1.0`  
**Compatibility policy:** implementations of schema `1.0` are not required to read earlier encoded schemas.

This document is the implementation target for the next stable `tree_coarsening` package. It supersedes earlier contracts based on finite static `(P, L, A)` vocabularies, nested `super_label`, fixed geometry per matching label, or scalar `attach_index`.

## 1. Normative language

The terms **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are normative.

- **MUST / MUST NOT**: required for interoperability or correctness.
- **SHOULD / SHOULD NOT**: strong recommendation; deviations require a documented reason.
- **MAY**: permitted but optional.

## 2. Scope

A tree coarsener replaces a larger labeled directed rooted tree with a smaller labeled directed rooted tree by applying an ordered contraction program.

The public graph object is always `networkx.DiGraph`. Auxiliary fitted artifacts include encoders, decoders, matching vocabularies, rule records, exact structural types, stage lineage, and raw provenance.

Schema `1.0` supports:

- directed edge contraction;
- sibling contraction in the **current encoded tree**;
- contraction of connected current-tree components;
- parametric rule families whose occurrences have different exact geometry;
- exact full-stage reversal;
- stage-local partial decoding;
- lazy composition of fitted stages.

Schema `1.0` does not require:

- backward compatibility with earlier graph schemas;
- a stable fitted-artifact disk format;
- pickle compatibility;
- preservation of arbitrary raw edge attributes;
- materialized composition;
- targeted decoding through a combined multi-stage decoder;
- optimized support for adversarially deep exact-type nesting.

## 3. Central occurrence model

Every encoded node occurrence has these package fields:

```python
node["label"]       # matching identity
node["type"]        # exact occurrence-specific structure
node["size"]        # exact fully expanded site count
node["time"]        # representative time
node["super_uids"]  # original UIDs in exact site order
```

Every encoded edge has:

```python
edge["attach_map"]  # exposed child roots -> sites of the parent occurrence
```

The core invariants are:

1. Matching and statistical fitting use `label`.
2. Structural transformation and decoding use `type`, `size`, `super_uids`, and `attach_map`.
3. Equal labels MAY have unequal exact types, exact sizes, root counts, arities, and attachment maps.
4. A downstream coarsener MUST NOT parse another producer's label to infer hidden geometry or wildcard positions.
5. Every concrete graph carries the exact information needed to transform it without consulting an upstream decoder.

For example, all stars in one family may use:

```python
("star", "P", "S")
```

while one occurrence represents 12 immediate components and another represents 37.

## 4. Raw graph schema

A raw input MUST be a nonempty, non-multigraph `nx.DiGraph` satisfying:

- exactly one node has in-degree zero;
- every other node has in-degree one;
- all edges point from parent to child;
- the graph is acyclic;
- the underlying undirected graph is connected;
- `|E| = |V| - 1`.

Every raw node MUST contain:

```python
node["label"]  # str
node["time"]   # finite real, excluding bool
node["uid"]    # stable hashable UID, unique within this raw graph
```

A UID SHOULD be a stable primitive or recursively tuple-valued object. If a custom hashable object is used, its equality, hash, and representation MUST remain stable during package operations.

Raw nodes MAY contain arbitrary additional user attributes. Raw graph attributes MAY also be arbitrary except that names beginning with `tree_coarsening_` are reserved.

Raw node fields `type`, `size`, and `super_uids` are reserved and MUST NOT be supplied as user data.

Raw edge attributes are outside the round-trip guarantee. A decoder reconstructs raw parent-child UID edges; original edge attributes can be joined back from an externally retained original graph.

## 5. Raw normalization

Before transformation, every raw node is normalized privately as:

```python
label       = raw_node["label"]
type        = ("base", raw_node["label"])
size        = 1
time        = float(raw_node["time"])
super_uids  = (raw_node["uid"],)
```

Every raw edge becomes:

```python
attach_map = (0,)
```

The initial encoded metadata is:

```python
schema = {"version": "1.0", "stages": ()}
fitting_sizes[raw_label] = 1
provenance = snapshot of raw node attributes and nonreserved graph attributes
```

Different coarseners MUST use the same base normalization.

## 6. Encoded graph schema

An encoded graph is also a directed rooted tree. Package-produced encoded nodes contain exactly:

```python
label
type
size
time
super_uids
```

Package-produced encoded edges contain exactly:

```python
attach_map
```

The following are not schema-`1.0` encoded-node fields:

```text
uid
raw_label
super_label
super_time
time_span
```

Scalar `attach_index` is not part of schema `1.0`.

Consumer-added attributes MAY be accepted, but package methods do not depend on them and need not preserve them.

For every encoded node `z`:

```python
z.label == exact_type_label(z.type)
z.size == exact_site_count(z.type)
z.size == len(z.super_uids)
```

For every encoded edge `u -> v`:

```python
len(attach_map) == exact_root_count(v.type)
all(0 <= q < u.size for q in attach_map)
```

The encoded root MUST satisfy:

```python
exact_root_count(root.type) == 1
```

Package-produced encoded node keys SHOULD be consecutive integers in deterministic parent-before-child order. Correctness MUST NOT depend on those keys.

## 7. Graph metadata and lineage

Encoded graphs contain these reserved graph attributes:

```python
graph["tree_coarsening_schema"]
graph["tree_coarsening_fitting_sizes"]
graph["tree_coarsening_provenance"]
```

The schema record is:

```python
{
    "version": "1.0",
    "stages": (
        {
            "model_id": str,
            "introduced_labels": tuple[MatchingLabel, ...],
        },
        ...
    ),
}
```

Stage records are ordered from earliest to latest active stage. Model IDs MUST be unique within one lineage.

A complete stage decode may run only when its decoder owns the latest active stage.

When a stage is fully reversed, its record and only the fitting-size labels it introduced are removed. Earlier stage records and inherited fitting sizes remain unchanged.

A no-op transform still appends one stage record, so encoder/decoder temporal order remains explicit.

## 8. Matching labels and fitting sizes

A matching label is an immutable, hashable identity compared by ordinary equality. Package-generated labels SHOULD use stable strings, non-Boolean integers, and recursively tuple-valued labels.

Exact node `size` and label-level fitting size are different concepts:

```python
node["size"]                 # exact represented original-site count
fitting_sizes[node["label"]] # statistical size of this matching label
```

All occurrences with one label MUST share one fitting size even when exact sizes differ.

An encoded fitting-size mapping MUST cover:

- every raw base label in provenance;
- every label introduced by an active stage;
- every label reachable in visible or nested exact types.

A fit corpus MUST reject conflicting fitting sizes for the same label.

## 9. Stage vocabulary

A fitted stage's vocabulary is the finite read-only mapping:

```python
output_matching_label -> positive fitting size
```

It is derived from the stage's fitted rules. It does not contain exact types, fixed geometry, or decoder recipes.

Suggested API:

```python
vocab.labels
vocab.has_label(label)
vocab.fitting_size(label)
vocab.as_dict()
```

There is no public fixed-geometry registry and no static recipe vocabulary in schema `1.0`.

## 10. Exact structural types

```python
ExactType = BaseType | CompositeType
```

### 10.1 Base type

A raw label `A` has base exact type:

```python
("base", A)
```

For a base type:

```python
exact_type_label(T) == A
exact_site_count(T) == 1
exact_root_count(T) == 1
```

### 10.2 Composite type

Normative semantic form:

```python
@dataclass(frozen=True)
class CompositeType:
    model_id: str
    label: MatchingLabel
    parent: tuple[int, ...]
    components: tuple[ExactType, ...]
    attach: tuple[int, ...]
```

Let `n = len(parent) = len(components)`. Requirements:

- `n >= 1`;
- `model_id` is nonempty;
- every `parent[i]` belongs to `{-1, 0, ..., n-1}`;
- `parent[i] != i`;
- component-parent relations are acyclic;
- package-produced composites place parents before children;
- every component is a valid exact type;
- nested stage ownership never points to a later stage than its container.

`parent[i] == -1` means component `i` is an exposed root component.

`attach` is a flat vector. For every component with an internal parent, its slice has width `exact_root_count(components[i])`. Slices are concatenated in component order, skipping exposed-root components.

If `parent[i] == j`, each value in component `i`'s slice is a valid site of `components[j]`.

Derived geometry:

```python
exact_site_count(T) = sum(exact_site_count(C) for C in T.components)

exact_root_count(T) = sum(
    exact_root_count(T.components[i])
    for i, p in enumerate(T.parent)
    if p == -1
)
```

Composite site order is concatenation of component site orders. Composite exposed-root order is concatenation of exposed roots of root components.

Two equal labels do not imply equal composite types.

## 11. Attachment semantics

For an encoded edge:

```text
u -> v
```

`attach_map[k]` is the site of `u` to which exposed root `k` of `v` attaches.

A multi-root child still has one current encoded parent node. Duplicate parent-site values are valid.

Broad sibling contraction is therefore exact. Siblings need only share the same current encoded parent; their incoming attachment maps need not be equal.

## 12. Provenance

Graph-level provenance contains:

```python
{
    "node_attrs_by_uid": Mapping[UID, Mapping[str, Any]],
    "graph_attrs": Mapping[Any, Any],
}
```

Every raw node's complete attribute mapping is stored by UID. `graph_attrs` stores original nonreserved graph attributes.

Across current encoded nodes, `super_uids` MUST form an exact one-time partition of provenance UIDs.

`super_uids` follows exact site order. Nested provenance is derived by splitting the flat UID tuple using component exact site counts. A separate nested `super_label` is redundant and not part of schema `1.0`.

Representative encoded time is:

```python
max(provenance[uid]["time"] for uid in super_uids)
```

A complete decode MUST NOT invent missing labels, UIDs, times, or raw attributes, including under `validate=False`.

## 13. Generic contraction

A contraction selects a nonempty current-node forest and orders its nodes deterministically.

Let selected nodes be `v_0, ..., v_{n-1}`. The new exact type is:

```python
CompositeType(
    model_id=encoder.model_id,
    label=rule.output_label,
    parent=selected_parent_vector,
    components=tuple(G.nodes[v_i]["type"] for i in range(n)),
    attach=concatenated_internal_attachment_maps,
)
```

The replacement node has:

```python
label       = rule.output_label
type        = composite above
size        = sum(component sizes)
super_uids  = concatenation of component super_uids
time        = max(component times)
```

A selected forest is contractible when either:

- it contains the graph root and has exactly one selected root; or
- all selected roots share one unselected current parent.

Incoming maps for selected roots are concatenated in selected-root order.

Outgoing edges from selected component `i` are offset by:

```python
offset_i = sum(component_size[h] for h in range(i))
```

so an old map `M` becomes:

```python
tuple(offset_i + q for q in M)
```

Simultaneous contractions MUST be vertex-disjoint and deterministic.

## 14. Primitive contractions

### 14.1 Edge contraction

For current edge `u -> v` with map `M`:

```python
parent     = (-1, 0)
components = (type(u), type(v))
attach     = M
```

Outgoing sites from `v` shift by `size(u)`.

### 14.2 Sibling contraction

For selected current siblings `v_1, ..., v_r`:

```python
parent     = (-1,) * r
components = tuple(type(v_i) for i in range(r))
attach     = ()
```

The replacement incoming edge concatenates their old incoming maps.

## 15. Encoding rules

Normative semantic form:

```python
@dataclass(frozen=True)
class EncodingRule:
    rule_index: int
    operation: str
    output_label: MatchingLabel
    output_fitting_size: int
    pattern: Mapping[str, Any]
    parameter_names: tuple[str, ...] = ()
    score: float | None = None
```

Requirements:

- rule indices are consecutive and define temporal order;
- `operation` is nonempty;
- output label is immutable and hashable;
- output fitting size is a positive non-Boolean integer;
- pattern is immutable from the caller's perspective;
- parameter names identify occurrence-specific data omitted from the matching label;
- generic rewiring and decoding do not branch on `operation` or parameter names.

The vocabulary MUST equal the distinct ordered rule-output labels and their declared fitting sizes.

## 16. Encoder artifact

A fitted encoder exposes:

```python
encoder.model_id
encoder.rules
encoder.vocab
```

Conceptually it is an immutable ordered contraction program plus a method-specific matching engine.

Transform accepts one raw or schema-`1.0` encoded graph and returns a new schema-`1.0` encoded graph.

Transform MUST:

- not mutate caller input;
- preserve provenance and active lineage;
- preserve inherited fitting sizes;
- match via labels/topology/configured UIDs, not upstream label parsing;
- use exact occurrence data for rewiring;
- append one stage record;
- remain semantically immutable across calls.

A stage model ID already active in the input graph MUST be rejected.

## 17. Decoder artifact

A fitted decoder exposes:

```python
decoder.model_id
decoder.rules
decoder.vocab
```

It owns a composite exactly when:

```python
isinstance(type_value, CompositeType)
and type_value.model_id == decoder.model_id
```

The decoder expands only owned types. Earlier-stage types remain opaque until their stage is reached.

## 18. One-layer expansion

Expanding one composite occurrence:

1. splits flat `super_uids` by immediate component site counts;
2. creates one current node per immediate component;
3. restores component `type`, `label`, `size`, `super_uids`, and representative `time`;
4. adds internal component edges from `parent` and `attach`;
5. splits incoming maps among exposed-root components;
6. translates outgoing maps to local component sites.

If one outgoing child attaches to several immediate components, leaving that child collapsed would create multiple current parents. Section 19 applies.

## 19. Partial decoding

Stage-local decode supports:

```python
decode(
    graph,
    *,
    target=None,
    by="node",           # "node", "label", or "type"
    recursive=True,
    boundary_policy="expand",  # or "raise"
    validate="full",
)
```

- `target is None`: complete stage reversal.
- `by="node"`: select one current node key.
- `by="label"`: select all owned occurrences with equal matching label.
- `by="type"`: select all owned occurrences with equal exact type.
- `recursive=False`: expand one owned layer.
- `recursive=True`: continue through nested types owned by this stage.

A missing node, nonowned node, or label/type with no owned match raises a selection/ownership error.

Boundary policy:

- `"raise"`: reject before mutating caller input;
- `"expand"`: compute the least fixed-point closure of same-stage boundary descendants needed to retain one current parent per node.

A valid sequential pipeline decoded in reverse order cannot require expansion of an earlier-stage boundary child. Such a case indicates malformed geometry or out-of-order stage handling and MUST raise.

Partial decoding preserves the active stage record and complete fitting-size mapping.

## 20. Complete stage reversal

A full stage decode:

1. verifies decoder ownership of the latest stage;
2. recursively expands every type owned by that stage;
3. verifies that no owned type remains, including nested owned types;
4. removes the latest stage record;
5. removes only fitting-size labels introduced by that stage;
6. returns the preceding encoded graph, or a raw graph if no stages remain.

Final raw materialization:

- keys raw nodes by UID;
- restores all original node attributes from provenance;
- reconstructs original parent-child UID edges;
- restores original nonreserved graph attributes;
- removes package schema metadata;
- does not promise raw edge-attribute restoration.

## 21. Coarsener API

```python
class TreeCoarsener(ABC):
    encoder_: TreeEncoder | None
    decoder_: TreeDecoder | None

    def fit(
        self,
        graphs: Sequence[nx.DiGraph],
        *,
        validate="full",
    ) -> Self: ...

    def transform(
        self,
        graph: nx.DiGraph,
        *,
        validate="full",
    ) -> nx.DiGraph: ...

    def fit_transform(
        self,
        graphs: Sequence[nx.DiGraph],
        *,
        validate="full",
    ) -> list[nx.DiGraph]: ...

    def decode(
        self,
        graph: nx.DiGraph,
        *,
        target=None,
        by="node",
        recursive=True,
        boundary_policy="expand",
        validate="full",
    ) -> nx.DiGraph: ...
```

`inverse_transform` MAY be an exact alias of `decode`.

`fit` accepts a nonempty sequence and returns `self`. It sets paired `encoder_` and `decoder_` artifacts atomically. A failed fit leaves prior fitted state unchanged.

## 22. ParametricStarCoarsener

Constructor semantics:

```python
d                 # minimum fit-time matching children for one witness
m                 # minimum number of witnessing parents
contract_d = d    # transform-time minimum
```

Fit learns pair `(A, B)` when at least `m` current nodes labeled `A` each have at least `d` current children labeled `B`.

Each learned pair emits one rule:

```python
operation = "siblings"
output_label = ("star", A, B)
output_fitting_size = 2
parameter_names = ("arity",)
```

Transform visits current `A` parents in deterministic parent-before-child order. If at least `contract_d` current `B` children exist, it contracts **all** such children into one sibling composite. The parent remains separate.

All qualifying arities share one label. Actual arity and exact component geometry remain in `CompositeType`.

## 23. EdgeBPECoarsener

BPE pair matching uses only:

```python
(parent_node["label"], child_node["label"])
```

Exact type, exact node size, arity, and attachment map are not part of the pair key.

Rule frequency is raw matching-edge count, including overlapping occurrences. A deterministic vertex-disjoint subset is actually contracted per merge step.

Supported score concepts include:

```text
count
normalized
size_weighted
```

Size-aware formulas use label-level fitting sizes, not exact node sizes.

Rule rank `r` emits a stage-namespaced label such as:

```python
("edge_bpe", model_id, r)
```

Output fitting size is:

```python
fitting_size(parent_label) + fitting_size(child_label)
```

Transformation constructs each exact composite from the actual parent type, child type, and edge attachment map. BPE therefore needs no special knowledge of star arity.

An optimized backend MUST learn the same ordered rules and counts as the reference Python backend. JIT warm-up is a benchmark concern, not a semantic difference.

## 24. NamedVertexCoarsener

Exactly one selection mode is configured:

```python
uids={...}
# or
labels={...}
```

Both arguments are explicit nonempty collections. A single tuple-valued label or UID is wrapped in a set or list to avoid tuple/collection ambiguity.

UID selection chooses a current occurrence only when all of its `super_uids` are selected. Partial overlap raises rather than selecting the whole supernode.

Label selection uses ordinary current matching-label equality.

Component policies:

```python
"all"      # every maximal qualifying connected current component
"largest"  # one deterministic largest component
```

Singletons remain unchanged.

One fitted named-component rule uses one fixed output label and a configured positive fitting size, defaulting to 1. Exact component topology remains occurrence-specific.

## 25. Lazy composition

```python
combine(
    encoders: Sequence[TreeEncoder],
    decoders: Sequence[TreeDecoder],
) -> tuple[TreeEncoder, TreeDecoder]
```

Encoders are supplied in application order and run forward. Decoders are paired in the same order and run in reverse.

`combine` rejects:

- empty or unequal-length sequences;
- mismatched encoder/decoder model IDs;
- mismatched paired rules or vocabularies;
- duplicate stage model IDs, including nested combined pipelines;
- conflicting fitting sizes for one label;
- unsupported schema versions.

Combined targeted partial decoding is deferred; callers use the current component decoder directly.

Materialized composition is not part of schema `1.0`.

## 26. Validation modes

```python
ValidationLevel = Literal["full", "structural", False]
```

`"full"` checks topology, all schema fields, type/label/size equations, provenance partition, representative times, fitting-size closure, lineage ownership, attachment bounds, and deterministic output conventions.

`"structural"` checks everything required for safe rewiring and decoding, while permitting omission of expensive global provenance/time scans.

`False` skips optional eager scans but does not change semantics. Operations still reject missing or malformed information when they use it. It never authorizes geometry inference from labels, invented provenance, invalid attachments, non-tree outputs, ownership bypass, or unsupported schema versions.

## 27. Determinism

Given the same fitted artifact and semantically equal input, transform and decode MUST be deterministic apart from an automatically generated model ID.

Methods define deterministic behavior for:

- fit traversal;
- sibling/component order;
- pair-score ties;
- overlapping occurrence selection;
- rule numbering;
- UID concatenation;
- output node numbering.

Package-generated labels use stable primitive tuples. Custom UIDs SHOULD have stable representation if they participate in ordering ties.

## 28. Errors

Recommended public hierarchy:

```text
TreeCoarseningError
├── ConfigurationError
├── InternalInvariantError
├── NotFittedError
├── ValidationError
│   ├── GraphSchemaError
│   ├── TreeStructureError
│   ├── LabelMetadataError
│   ├── FittingSizeError
│   ├── ExactTypeError
│   ├── AttachmentError
│   ├── ProvenanceError
│   ├── StageOrderError
│   └── TypeOwnershipError
├── DecodeSelectionError
│   └── TargetNotFoundError
├── BoundaryExpansionError
└── CompositionError
```

Public errors identify the violated concept and relevant node, edge, model ID, or rule when practical.

## 29. Nonmutation and failure atomicity

Public fit, transform, decode, and composition operations MUST NOT mutate caller graphs.

A failed public operation leaves caller graphs unchanged. A failed refit leaves the prior fitted artifacts and public fit diagnostics unchanged.

Only top-level graph/node/edge attribute mappings require isolation. Deep-copy isolation of arbitrary nested mutable user values is deferred.

Fitted artifacts are semantically immutable. Private caches MAY change if they do not alter rule meaning, vocabulary, equality of results, or inspection output.

## 30. Persistence and versioning

No pickle compatibility is promised.

A future persistence API SHOULD use a versioned plain-data state representation such as:

```python
encoder.to_state()
TreeEncoder.from_state(state)
```

Package version, encoded graph schema version, and future artifact serialization version are distinct.

A schema-`1.0` implementation rejects unsupported graph schema versions explicitly. No migration layer is required.

## 31. Contributor conformance obligations

Every new coarsener documents and tests:

1. fitting criterion;
2. rule application order;
3. output-label construction;
4. output fitting-size formula;
5. omitted parametric fields;
6. deterministic ties and overlap handling;
7. exact contraction selection constraints;
8. raw and encoded input support;
9. equal-label variable-geometry support;
10. full and partial stage decoding;
11. one-stage and multi-stage round trips;
12. input nonmutation and failure atomicity;
13. malformed-input rejection;
14. backend parity when optimized implementations exist.

Reusable conformance tests SHOULD cover:

- raw normalization;
- schema and provenance equations;
- variable geometry per label;
- broad sibling attachment maps;
- partial boundary expansion;
- stage ownership and order;
- fitting-size restoration;
- deterministic output;
- randomized multi-stage round trips;
- reference/optimized backend parity.

## 32. Locked schema-1 decisions

1. Labels are matching identities, not structural recipes.
2. Equal labels may have variable exact geometry.
3. Exact structure lives in occurrence-specific `CompositeType` values.
4. Exact `size` and label-level fitting size are distinct.
5. Parametric matching is achieved by label design, not wildcard-aware downstream code.
6. One generic encoder/decoder architecture supports parametric and nonparametric rules.
7. Encoded edges use tuple-valued `attach_map` only.
8. Broad sibling contraction is valid.
9. Flat `super_uids` is the required occurrence provenance field.
10. There is no public fixed-geometry registry or static recipe vocabulary.
11. Graphs carry exact active stage lineage and fitting-size closure.
12. Full stage reversal restores the immediately preceding metadata state.
13. Partial decoding either preserves a tree through minimal boundary expansion or raises.
14. Public graph operations are nonmutating.
15. Raw node and graph attributes round-trip; raw edge attributes do not.
16. Unsupported schema versions fail explicitly.
17. Persistence and materialized composition are deferred.
