from __future__ import annotations

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

from conftest import make_tree, raw_signature


class _TinyEncoder(RuleBasedEncoder):
    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[object]]:
        del rule
        return tuple(
            (parent, child)
            for parent, child in graph.edges
            if graph.nodes[parent]["label"] == "A" and graph.nodes[child]["label"] == "B"
        )[:1]


class _TinyCoarsener(TreeCoarsener):
    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        del graphs
        rule = EncodingRule(0, "edge", ("tiny", self.model_id), 2, {"edge": ("A", "B")})
        rules = (rule,)
        return (
            _TinyEncoder(model_id=self.model_id, rules=rules),
            StructuralStageDecoder(model_id=self.model_id, rules=rules),
        )


def test_simple_coarsener_only_supplies_rule_and_selection() -> None:
    graph = make_tree(["A", "B", "C"], [None, 0, 1], prefix="tiny")
    model = _TinyCoarsener(model_id="tiny-stage").fit([graph])
    encoded = model.transform(graph)
    assert encoded.number_of_nodes() == 2
    assert raw_signature(model.decode(encoded)) == raw_signature(graph)


def test_rule_based_encoder_validates_once_at_the_public_boundary(monkeypatch) -> None:
    import tree_coarsening.contraction as contraction_module

    graph = make_tree(["A", "B", "C"], [None, 0, 1], prefix="validation-count")
    model = _TinyCoarsener(model_id="validation-count").fit([graph])
    calls = 0
    original = contraction_module.validate_encoded_tree

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(contraction_module, "validate_encoded_tree", counted)
    model.transform(graph)
    assert calls == 1
