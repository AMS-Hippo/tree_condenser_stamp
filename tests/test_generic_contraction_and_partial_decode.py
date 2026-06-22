from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Hashable

import networkx as nx
import pytest

from tree_coarsening import (
    BoundaryExpansionError,
    CompositeType,
    Contraction,
    DecodeSelectionError,
    EncodingRule,
    RuleBasedEncoder,
    StageOrderError,
    StructuralStageDecoder,
    TargetNotFoundError,
    TreeCoarsener,
    TreeDecoder,
    TreeEncoder,
    TypeOwnershipError,
    ValidationError,
    Vocabulary,
    apply_contraction_batch,
    exact_root_count,
    validate_encoded_tree,
    normalize_raw_graph,
)

from tree_coarsening.schema import append_stage

from conftest import (
    assert_graph_unchanged,
    make_tree,
    raw_signature,
    snapshot_graph,
)


class _BoundaryEncoder(RuleBasedEncoder):
    """Two tiny rules whose result exercises multi-site boundary closure."""

    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[Hashable]]:
        if rule.rule_index == 0:
            matches = [
                (parent, child)
                for parent, child in graph.edges
                if graph.nodes[parent]["label"] == "R" and graph.nodes[child]["label"] == "A"
            ]
            return tuple(matches[:1])

        for parent in graph:
            members = tuple(
                child
                for child in graph.successors(parent)
                if graph.nodes[child]["label"] in {"B", "C"}
            )
            if {graph.nodes[node]["label"] for node in members} == {"B", "C"}:
                return (members,)
        return ()


class _BoundaryCoarsener(TreeCoarsener):
    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        del graphs
        rules = (
            EncodingRule(0, "edge", ("parent", self.model_id), 2, {}),
            EncodingRule(1, "siblings", ("child", self.model_id), 2, {}),
        )
        return (
            _BoundaryEncoder(model_id=self.model_id, rules=rules),
            StructuralStageDecoder(model_id=self.model_id, rules=rules),
        )


class _InvalidSelectionEncoder(RuleBasedEncoder):
    def __init__(
        self,
        *,
        model_id: str,
        rules: Sequence[EncodingRule],
        groups: Sequence[Sequence[Hashable]],
    ) -> None:
        super().__init__(model_id=model_id, rules=rules)
        self.groups = tuple(tuple(group) for group in groups)

    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[Hashable]]:
        del graph, rule
        return self.groups


def _boundary_tree() -> nx.DiGraph:
    # R-A is first contracted. B and C then become siblings attached to sites
    # 0 and 1 of the R-A occurrence and are contracted into one two-root child.
    return make_tree(
        ["R", "A", "B", "C"],
        [None, 0, 0, 1],
        prefix="boundary",
    )


def test_generic_contraction_records_broad_sibling_attachment_exactly() -> None:
    graph = _boundary_tree()
    model = _BoundaryCoarsener(model_id="boundary-stage").fit([graph])
    encoded = model.transform(graph)
    validate_encoded_tree(encoded)

    parent = next(
        node
        for node, data in encoded.nodes(data=True)
        if data["label"] == ("parent", "boundary-stage")
    )
    child = next(
        node
        for node, data in encoded.nodes(data=True)
        if data["label"] == ("child", "boundary-stage")
    )
    parent_type = encoded.nodes[parent]["type"]
    child_type = encoded.nodes[child]["type"]
    assert isinstance(parent_type, CompositeType)
    assert isinstance(child_type, CompositeType)
    assert parent_type.parent == (-1, 0)
    assert parent_type.attach == (0,)
    assert child_type.parent == (-1, -1)
    assert exact_root_count(child_type) == 2
    assert encoded.edges[parent, child]["attach_map"] == (0, 1)
    assert raw_signature(model.decode(encoded)) == raw_signature(graph)


def test_partial_decode_boundary_raise_is_atomic() -> None:
    graph = _boundary_tree()
    model = _BoundaryCoarsener(model_id="boundary-raise").fit([graph])
    encoded = model.transform(graph)
    target = next(
        node
        for node, data in encoded.nodes(data=True)
        if data["label"] == ("parent", "boundary-raise")
    )
    before = snapshot_graph(encoded)
    with pytest.raises(BoundaryExpansionError, match="multiple current parents"):
        model.decode(
            encoded,
            target=target,
            by="node",
            recursive=False,
            boundary_policy="raise",
        )
    assert_graph_unchanged(encoded, before)


def test_partial_decode_boundary_expand_is_minimal_and_round_trips() -> None:
    graph = _boundary_tree()
    model = _BoundaryCoarsener(model_id="boundary-expand").fit([graph])
    encoded = model.transform(graph)
    target = next(
        node
        for node, data in encoded.nodes(data=True)
        if data["label"] == ("parent", "boundary-expand")
    )
    partial = model.decode(
        encoded,
        target=target,
        by="node",
        recursive=False,
        boundary_policy="expand",
    )
    validate_encoded_tree(partial)
    assert partial.number_of_nodes() == 4
    assert all(not isinstance(data["type"], CompositeType) for _, data in partial.nodes(data=True))
    assert partial.graph["tree_coarsening_schema"]["stages"][-1]["model_id"] == ("boundary-expand")
    assert raw_signature(model.decode(partial)) == raw_signature(graph)


@pytest.mark.parametrize("by", ["label", "type"])
def test_partial_decode_can_select_by_label_or_exact_type(by: str) -> None:
    graph = _boundary_tree()
    model = _BoundaryCoarsener(model_id=f"select-{by}").fit([graph])
    encoded = model.transform(graph)
    occurrence = next(
        data for _, data in encoded.nodes(data=True) if data["label"] == ("child", f"select-{by}")
    )
    target = occurrence["label"] if by == "label" else occurrence["type"]
    partial = model.decode(encoded, target=target, by=by, recursive=False)
    validate_encoded_tree(partial)
    assert all(data["label"] != occurrence["label"] for _, data in partial.nodes(data=True))
    assert raw_signature(model.decode(partial)) == raw_signature(graph)


def test_partial_decode_rejects_invalid_selector_and_missing_target() -> None:
    graph = _boundary_tree()
    model = _BoundaryCoarsener(model_id="selection-errors").fit([graph])
    encoded = model.transform(graph)
    with pytest.raises(DecodeSelectionError):
        model.decode(encoded, target=0, by="unknown")
    with pytest.raises(TargetNotFoundError):
        model.decode(encoded, target="absent", by="node")
    with pytest.raises(TargetNotFoundError):
        model.decode(encoded, target="absent", by="label")


def test_partial_decode_rejects_nonowned_node() -> None:
    graph = make_tree(
        ["R", "A", "B", "C", "D"],
        [None, 0, 0, 1, 0],
        prefix="nonowned",
    )
    model = _BoundaryCoarsener(model_id="nonowned").fit([graph])
    encoded = model.transform(graph)
    raw_node = next(
        node
        for node, data in encoded.nodes(data=True)
        if not isinstance(data["type"], CompositeType)
    )
    with pytest.raises(TypeOwnershipError):
        model.decode(encoded, target=raw_node, by="node")


def test_earlier_stage_decoder_cannot_run_out_of_order() -> None:
    graph = _boundary_tree()
    first = _BoundaryCoarsener(model_id="first-stage").fit([graph])
    encoded_first = first.transform(graph)
    second = _BoundaryCoarsener(model_id="second-stage").fit([encoded_first])
    encoded_second = second.transform(encoded_first)
    with pytest.raises(StageOrderError):
        first.decode(encoded_second)
    assert raw_signature(first.decode(second.decode(encoded_second))) == raw_signature(graph)


def test_generic_batch_rejects_overlapping_groups() -> None:
    graph = make_tree(["A", "B", "C"], [None, 0, 1], prefix="overlap-batch")
    rule = EncodingRule(0, "component", "X", 1, {})
    encoder = _InvalidSelectionEncoder(
        model_id="bad-overlap",
        rules=(rule,),
        groups=((0, 1), (1, 2)),
    )
    before = deepcopy(graph)
    with pytest.raises(ValidationError, match="overlap"):
        encoder.transform(graph)
    assert_graph_unchanged(graph, before)


def test_generic_batch_rejects_noncontractible_forest() -> None:
    graph = make_tree(
        ["R", "A", "B", "C"],
        [None, 0, 0, 1],
        prefix="bad-forest",
    )
    rule = EncodingRule(0, "component", "X", 1, {})
    # A and B are siblings, but C is below A. Selecting B and C gives roots
    # with different outside parents and is therefore not contractible.
    encoder = _InvalidSelectionEncoder(
        model_id="bad-forest",
        rules=(rule,),
        groups=((2, 3),),
    )
    with pytest.raises(ValidationError, match="share one current parent"):
        encoder.transform(graph)


def test_public_contraction_batch_returns_deterministically_numbered_nodes() -> None:
    raw = make_tree(["R", "A", "B"], [None, 0, 0], prefix="public-batch")
    encoded = normalize_raw_graph(raw)
    rule = EncodingRule(0, "edge", ("joined", "public-batch"), 2, {})
    append_stage(encoded, model_id="public-batch", vocab=Vocabulary((rule,)))

    first = apply_contraction_batch(
        encoded,
        model_id="public-batch",
        rule=rule,
        contractions=(Contraction(0, (0, 1)),),
    )
    second = apply_contraction_batch(
        encoded,
        model_id="public-batch",
        rule=rule,
        contractions=(Contraction(0, (0, 1)),),
    )

    assert tuple(first.nodes) == tuple(range(first.number_of_nodes()))
    assert tuple(second.nodes) == tuple(first.nodes)
    assert raw_signature(
        StructuralStageDecoder(model_id="public-batch", rules=(rule,)).decode(first)
    ) == raw_signature(raw)


def test_generic_batch_canonicalization_uses_one_global_order_scan(monkeypatch) -> None:
    import tree_coarsening.contraction as contraction_module

    raw = make_tree(
        ["R", "A", "B", "C", "D"],
        [None, 0, 1, 0, 3],
        prefix="linear-canonicalization",
    )
    graph = normalize_raw_graph(raw)
    rule = EncodingRule(
        0,
        "edge",
        ("joined", "linear-canonicalization"),
        2,
        {},
    )
    append_stage(
        graph,
        model_id="linear-canonicalization",
        vocab=Vocabulary((rule,)),
    )
    order = tuple(nx.topological_sort(graph))

    class OnePassOrder:
        def __init__(self, values):
            self.values = values
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("global node order was rescanned")
            return iter(self.values)

    one_pass = OnePassOrder(order)
    monkeypatch.setattr(
        contraction_module,
        "deterministic_node_order",
        lambda _graph: one_pass,
    )

    result = apply_contraction_batch(
        graph,
        model_id="linear-canonicalization",
        rule=rule,
        contractions=(
            Contraction(rule_index=0, nodes=(0, 1)),
            Contraction(rule_index=0, nodes=(3, 4)),
        ),
        _validate_result=False,
    )

    assert result.number_of_nodes() == 3
    assert one_pass.iterations == 1
