# Writing a schema-1 coarsener

The shared schema is intentionally more demanding to validate than to author.
An ordinary rule-based coarsener should not implement provenance, exact-type
construction, attachment rewiring, stage metadata, decoding, or graph
validation itself.

## Smallest useful pattern

A simple method needs two pieces:

1. an `EncodingRule` describing its output matching label and fitting size;
2. a `RuleBasedEncoder.select_contractions` implementation that returns the
   current node groups selected for that rule.

```python
from collections.abc import Iterable, Sequence

import networkx as nx

from tree_coarsening import (
    EncodingRule,
    RuleBasedEncoder,
    StructuralStageDecoder,
    TreeCoarsener,
    TreeDecoder,
    TreeEncoder,
)


class ABEdgeEncoder(RuleBasedEncoder):
    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[object]]:
        del rule
        # This example deliberately takes only the first match so selections
        # are vertex-disjoint. A real method defines its deterministic policy.
        matches = (
            (parent, child)
            for parent, child in graph.edges
            if graph.nodes[parent]["label"] == "A"
            and graph.nodes[child]["label"] == "B"
        )
        return tuple(matches)[:1]


class ABEdgeCoarsener(TreeCoarsener):
    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        del graphs
        rule = EncodingRule(
            rule_index=0,
            operation="edge",
            output_label=("ab_edge", self.model_id),
            output_fitting_size=2,
            pattern={"parent_label": "A", "child_label": "B"},
        )
        rules = (rule,)
        return (
            ABEdgeEncoder(model_id=self.model_id, rules=rules),
            StructuralStageDecoder(model_id=self.model_id, rules=rules),
        )
```

`TreeCoarsener.fit` normalizes raw and encoded corpora, checks corpus fitting
sizes, validates the artifact pair, and installs it atomically.
`RuleBasedEncoder.transform` appends lineage, applies rules in order, delegates
concrete geometry to the generic contraction engine, deterministically renumbers
nodes, and validates once at the public boundary. `StructuralStageDecoder`
provides full and partial exact reversal.

`NamedVertexCoarsener` is the production example of this pattern.

## What the method author must define

Every method still owns its substantive semantics:

- fitting criterion and rule order;
- output-label construction and fitting-size formula;
- concrete current-node selection;
- deterministic tie and overlap behavior;
- occurrence-varying fields omitted from the matching label;
- any method-specific optimized backend and its parity proof.

A selected group must satisfy the generic contraction contract: it is a
nonempty current-node forest; its selected roots either contain the graph root
as the sole selected root or share one unselected current parent. Groups in one
batch must be vertex-disjoint.

## When not to use RuleBasedEncoder

A specialized transform may subclass `TreeEncoder` directly when rule
application requires a custom incremental state machine. Edge BPE does this to
preserve its optimized pair index and overlap machinery. Even then, the
statistical kernel should remain isolated from schema adapters, and concrete
output occurrences should still use the generic schema primitives.

Parametric Star also uses a small specialized traversal because newly created
current children may themselves match later star rules. Its concrete
contractions still go through the shared contraction engine.

## Required verification

Use the reusable tests as templates:

- `tests/test_authoring_surface.py` for a minimal method;
- `tests/test_generic_contraction_and_partial_decode.py` for geometry;
- `tests/test_cross_stage_matrix.py` for stage ordering;
- `tests/test_api_and_failure_atomicity.py` for public behavior;
- `tests/test_randomized_pipeline.py` for multi-stage round trips;
- `tests/test_edge_bpe_parity_and_protection.py` for optimized backends.

The complete contributor obligations remain normative in section 31 of
`CURRENT_SCHEMA1_API.md`.
