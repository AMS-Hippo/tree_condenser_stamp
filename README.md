# tree-coarsening

Exact, reversible coarsening of labeled directed rooted trees represented as
NetworkX `DiGraph` objects.

The implementation targets encoded graph schema **1.0**. The frozen normative
contract is [`CURRENT_SCHEMA1_API.md`](CURRENT_SCHEMA1_API.md); the copies at
`tree_coarsening_api.md` and `docs/tree_coarsening_api.md` are kept
byte-identical for discoverability.


## Documentation map

Start with [`DOCS_OVERVIEW.md`](DOCS_OVERVIEW.md) if you are unsure which
document to read. In brief, `CURRENT_SCHEMA1_API.md` is the frozen normative
contract; `README.md` is user-facing; `CONTRIBUTING_COARSENERS.md` is for new
coarsener authors; `CONTRACT_TRACEABILITY.md` maps contract clauses to code and
tests; `audit/` records release evidence.

## Raw inputs

A raw input is a nonempty directed rooted tree. Every node supplies:

```python
node["label"]  # str
node["time"]   # finite real, excluding bool
node["uid"]    # unique, stable, hashable identifier
```

Additional raw node and graph attributes round-trip. Raw edge attributes are not
part of the round-trip guarantee.

```python
import networkx as nx

G = nx.DiGraph()
G.add_node("root", label="P", time=0.0, uid="root")
G.add_node("left", label="C", time=1.0, uid="left")
G.add_node("right", label="C", time=2.0, uid="right")
G.add_edges_from([("root", "left"), ("root", "right")])
```

## Common public API

All coarseners share the same fitted interface:

```python
model.fit([G])                 # fit always receives a nonempty sequence
H = model.transform(G)         # one graph in, one new graph out
G2 = model.decode(H)           # exact full-stage reversal
Hs = model.fit_transform([G])  # list out
```

Public operations do not mutate caller graphs. A failed refit preserves the
previous fitted artifacts and diagnostics.

Encoded nodes use exactly these package fields:

```python
node["label"]       # matching identity
node["type"]        # exact occurrence-specific structure
node["size"]        # exact represented raw-site count
node["time"]        # representative maximum time
node["super_uids"]  # flat raw UID provenance in exact site order
```

Encoded edges use tuple-valued `edge["attach_map"]`. Matching uses labels;
rewiring and decoding use exact occurrence data. Equal labels may therefore
have unequal exact geometry.

## Simple named-component coarsening

A simple coarsener does not need to engage with parametric machinery:

```python
from tree_coarsening import NamedVertexCoarsener

model = NamedVertexCoarsener(
    labels={"A", "B"},
    component_policy="all",   # or "largest"
).fit([G])

H = model.transform(G)
G2 = model.decode(H)
```

Exactly one of `labels=` and `uids=` is supplied as an explicit nonempty
collection. Selected maximal connected components of at least two current
nodes are contracted. UID selection accepts an encoded occurrence only when all
of its `super_uids` are selected; partial overlap raises.

## Parametric stars

```python
from tree_coarsening import ParametricStarCoarsener

star = ParametricStarCoarsener(
    d=3,          # fit-time children required per witness
    m=2,          # witnessing parents required
    contract_d=2, # optional transform-time threshold; defaults to d
).fit(training_trees)

U = star.transform(G)
G2 = star.decode(U)
```

A learned pair `(P, C)` emits the matching label `("star", P, C)` for every
qualifying transform-time arity. The label has fitting size 2; concrete arity,
site count, and attachment geometry remain in each occurrence's exact type.

## Edge BPE

```python
from tree_coarsening import EdgeBPECoarsener

bpe = EdgeBPECoarsener(
    num_merges=32,
    min_pair_count=2,
    pair_score="count",       # "normalized" or "size_weighted" also supported
    backend="python",         # optional "numba"
).fit(training_trees)

H = bpe.transform(G)
G2 = bpe.decode(H)
```

BPE matches only `(parent_label, child_label)`. Candidate frequencies are raw
matching-edge counts, including overlaps; each step contracts a deterministic
vertex-disjoint subset. Exact types, exact sizes, arity, and attachment maps do
not enter the pair key. A learned rule at rank `r` emits
`("edge_bpe", model_id, r)` and receives the additive fitting size of its two
input labels.

The optimized Numba fitting backend is optional:

```bash
pip install "tree-coarsening[numba]"
```

For an editable source checkout, use `pip install -e ".[numba]"` instead.

The Python and Numba backends are required to learn identical ordered rules and
event counts. JIT warm-up affects benchmarks, not semantics.

## Multi-stage pipelines

Stages operate directly on schema-1 encoded graphs:

```python
star = ParametricStarCoarsener(3, 2).fit(raw_trees)
U = star.transform(G)

bpe = EdgeBPECoarsener(num_merges=16, min_pair_count=2).fit([U])
V = bpe.transform(U)

U2 = bpe.decode(V)   # reverse latest stage first
G2 = star.decode(U2)
```

Fitted stages can also be lazily combined:

```python
from tree_coarsening import combine

encoder, decoder = combine(
    [star.encoder_, bpe.encoder_],
    [star.decoder_, bpe.decoder_],
)
V = encoder.transform(G)
G2 = decoder.decode(V)
```

Combined targeted partial decoding is intentionally deferred; use the current
atomic stage decoder for targeted expansion.

## Partial decoding

```python
part = model.decode(
    H,
    target=some_node,
    by="node",                # "label" or "type" also supported
    recursive=True,
    boundary_policy="expand", # or "raise"
)
```

`"expand"` computes the minimal same-stage boundary closure required to retain
a tree. `"raise"` rejects the operation without mutating the input.

## Validation

Public methods accept:

```python
validate="full"        # complete schema/provenance validation
validate="structural"  # checks required for safe rewiring/decoding
validate=False          # skips optional eager scans, never changes semantics
```

`validate=False` does not permit invented provenance, malformed geometry,
invalid attachments, stage-order bypass, or unsupported schema versions.

## Development checks

```bash
pytest -W error
python -m compileall -q tree_coarsening tests
ruff check .
ruff format --check .
```

The repository also freezes the v0.12.1 BPE reference implementation under
`audit/`. Tests enforce byte identity of the Numba kernel, AST identity of the
unchanged Python scoring/index/contraction core, and differential backend
parity.
