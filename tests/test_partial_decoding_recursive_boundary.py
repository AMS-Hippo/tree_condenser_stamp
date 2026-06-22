from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Hashable

import networkx as nx
import pytest

from tree_coarsening import (
    BoundaryExpansionError,
    CompositeType,
    EncodingRule,
    RuleBasedEncoder,
    StructuralStageDecoder,
    TreeCoarsener,
    TreeDecoder,
    TreeEncoder,
    validate_encoded_tree,
)

from conftest import assert_graph_unchanged, make_tree, raw_signature, snapshot_graph


class _NestedBoundaryEncoder(RuleBasedEncoder):
    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[Hashable]]:
        if rule.rule_index < 3:
            parent_label = rule.pattern["parent_label"]
            child_label = rule.pattern["child_label"]
            matches = tuple(
                (parent, child)
                for parent, child in graph.edges
                if graph.nodes[parent]["label"] == parent_label
                and graph.nodes[child]["label"] == child_label
            )
            return matches[:1]

        left_label = rule.pattern["left_label"]
        right_label = rule.pattern["right_label"]
        for parent in graph:
            children = tuple(graph.successors(parent))
            left = tuple(child for child in children if graph.nodes[child]["label"] == left_label)
            right = tuple(child for child in children if graph.nodes[child]["label"] == right_label)
            if len(left) == len(right) == 1:
                return ((left[0], right[0]),)
        return ()


class _NestedBoundaryCoarsener(TreeCoarsener):
    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        del graphs
        parent_label = ("nested-parent", self.model_id)
        left_label = ("nested-left", self.model_id)
        right_label = ("nested-right", self.model_id)
        child_label = ("nested-child", self.model_id)
        rules = (
            EncodingRule(
                0,
                "edge",
                parent_label,
                2,
                {"parent_label": "R", "child_label": "A"},
            ),
            EncodingRule(
                1,
                "edge",
                left_label,
                2,
                {"parent_label": "X", "child_label": "x"},
            ),
            EncodingRule(
                2,
                "edge",
                right_label,
                2,
                {"parent_label": "Y", "child_label": "y"},
            ),
            EncodingRule(
                3,
                "siblings",
                child_label,
                4,
                {"left_label": left_label, "right_label": right_label},
                parameter_names=("topology",),
            ),
        )
        return (
            _NestedBoundaryEncoder(model_id=self.model_id, rules=rules),
            StructuralStageDecoder(model_id=self.model_id, rules=rules),
        )


def _tree() -> nx.DiGraph:
    # Contract R-A. X and Y then attach to different sites of that occurrence.
    # X-x and Y-y are contracted before those two resulting composites are
    # sibling-contracted, so boundary expansion exposes nested owned types.
    return make_tree(
        ["R", "A", "X", "x", "Y", "y"],
        [None, 0, 0, 2, 1, 4],
        prefix="nested-boundary",
    )


def test_full_decode_revisits_owned_types_exposed_by_boundary_closure() -> None:
    graph = _tree()
    model = _NestedBoundaryCoarsener(model_id="nested-boundary-stage").fit([graph])
    encoded = model.transform(graph)
    validate_encoded_tree(encoded)
    assert encoded.number_of_nodes() == 2

    decoded = model.decode(encoded)
    assert raw_signature(decoded) == raw_signature(graph)


def test_partial_boundary_expansion_is_minimal_with_nested_owned_child() -> None:
    graph = _tree()
    model = _NestedBoundaryCoarsener(model_id="nested-minimal").fit([graph])
    encoded = model.transform(graph)
    target = next(
        node
        for node, data in encoded.nodes(data=True)
        if data["label"] == ("nested-parent", "nested-minimal")
    )

    partial = model.decode(
        encoded,
        target=target,
        by="node",
        recursive=False,
        boundary_policy="expand",
    )
    validate_encoded_tree(partial)
    owned_visible = [
        data["type"]
        for _, data in partial.nodes(data=True)
        if isinstance(data["type"], CompositeType) and data["type"].model_id == "nested-minimal"
    ]
    assert {exact.label for exact in owned_visible} == {
        ("nested-left", "nested-minimal"),
        ("nested-right", "nested-minimal"),
    }
    assert raw_signature(model.decode(partial)) == raw_signature(graph)


def test_nested_boundary_raise_is_failure_atomic() -> None:
    graph = _tree()
    model = _NestedBoundaryCoarsener(model_id="nested-raise").fit([graph])
    encoded = model.transform(graph)
    target = next(
        node
        for node, data in encoded.nodes(data=True)
        if data["label"] == ("nested-parent", "nested-raise")
    )
    before = snapshot_graph(encoded)

    with pytest.raises(BoundaryExpansionError):
        model.decode(
            encoded,
            target=target,
            by="node",
            recursive=False,
            boundary_policy="raise",
        )
    assert_graph_unchanged(encoded, before)
