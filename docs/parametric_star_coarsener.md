# ParametricStarCoarsener

`ParametricStarCoarsener` learns arity-independent sibling families while
retaining the exact geometry of every concrete occurrence.

```python
from tree_coarsening import ParametricStarCoarsener

model = ParametricStarCoarsener(
    d=3,
    m=2,
    contract_d=2,
).fit(training_trees)
```

## Fitting

For each ordered current-label pair `(P, C)`, fitting counts parents labeled
`P` that have at least `d` current children labeled `C`. The pair is learned
when at least `m` parents witness it.

`contract_d` is the transform-time threshold and defaults to `d`. Both `d` and
`contract_d` must be integers of at least 2; `m` must be positive. The
transform threshold is independent of the witness threshold, so it may be
smaller or larger than `d`.

Each learned pair creates one rule:

```python
operation = "siblings"
output_label = ("star", P, C)
output_fitting_size = 2
parameter_names = ("arity",)
```

Rules are ordered deterministically by the representations of `P` and `C`.

## Transformation

Transformation visits the current tree in deterministic parent-before-child
order. At each current parent labeled `P`, every learned child-label family is
considered. When at least `contract_d` current children have label `C`, all of
those children are contracted into one sibling occurrence. Newly created
current children are visited in their own turn.

Every qualifying arity shares the same matching label `("star", P, C)`. The
actual arity, component types, site count, exposed roots, and attachment
geometry live in the occurrence-specific `CompositeType`. Consequently:

```python
node["label"]                    # family identity
node["size"]                     # exact represented site count
node["type"]                     # exact concrete geometry
model.encoder_.vocab.fitting_size(node["label"]) == 2
```

A downstream method, including Edge BPE, matches the family label without
parsing star-specific geometry.

## Decoding

The paired decoder is the generic occurrence-driven
`StructuralStageDecoder`. It reconstructs each concrete sibling group from the
stored exact type and attachment maps. Full and partial decoding therefore do
not require an arity-indexed decoder vocabulary.

## Inspection

```python
for rule in model.encoder_.rules:
    print(rule.pattern["parent_label"], rule.pattern["child_label"])
    print(rule.output_label, rule.output_fitting_size)

for label in model.encoder_.vocab.labels:
    print(label, model.encoder_.vocab.fitting_size(label))
```

The stage vocabulary contains matching labels and fitting sizes only. Exact
recipes remain in graph occurrences.
