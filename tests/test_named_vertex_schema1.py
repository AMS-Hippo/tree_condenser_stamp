from __future__ import annotations

import networkx as nx
import pytest

from tree_coarsening import (
    NamedVertexCoarsener,
    ProvenanceError,
    SCHEMA_KEY,
    validate_encoded_tree,
)

from conftest import (
    assert_graph_unchanged,
    make_tree,
    raw_signature,
    snapshot_graph,
)


def test_named_label_component_round_trip_and_nonmutation(chain4: nx.DiGraph) -> None:
    before = snapshot_graph(chain4)
    model = NamedVertexCoarsener(labels={"B"}, model_id="named").fit([chain4])
    assert_graph_unchanged(chain4, before)

    encoded = model.transform(chain4)
    assert_graph_unchanged(chain4, before)
    validate_encoded_tree(encoded)
    assert encoded.number_of_nodes() == 3
    contracted = [
        data
        for _, data in encoded.nodes(data=True)
        if data["label"] != "A" and data["label"] != "C"
    ]
    assert len(contracted) == 1
    assert contracted[0]["size"] == 2
    assert contracted[0]["super_uids"] == (("chain", 1), ("chain", 2))

    encoded_before = snapshot_graph(encoded)
    decoded = model.decode(encoded)
    assert raw_signature(decoded) == raw_signature(chain4)
    assert_graph_unchanged(encoded, encoded_before)


def test_named_uid_selection_rejects_partial_supernode_overlap(chain4: nx.DiGraph) -> None:
    first = NamedVertexCoarsener(labels={"B"}, model_id="first").fit([chain4])
    encoded = first.transform(chain4)
    second = NamedVertexCoarsener(
        uids={("chain", 1)},
        model_id="second",
    ).fit([encoded])
    before = snapshot_graph(encoded)
    with pytest.raises(ProvenanceError, match="partially overlaps"):
        second.transform(encoded)
    assert_graph_unchanged(encoded, before)


def test_named_singletons_are_noop_but_stage_is_recorded() -> None:
    graph = make_tree(["A", "B", "A"], [None, 0, 1], prefix="singletons")
    model = NamedVertexCoarsener(labels={"A"}, model_id="noop").fit([graph])
    encoded = model.transform(graph)
    assert encoded.number_of_nodes() == graph.number_of_nodes()
    assert encoded.graph[SCHEMA_KEY]["stages"] == (
        {"model_id": "noop", "introduced_labels": (("named_component", "noop"),)},
    )
    assert raw_signature(model.decode(encoded)) == raw_signature(graph)


def test_named_largest_component_has_deterministic_semantic_tie_break() -> None:
    graph = make_tree(
        ["R", "X", "X", "N", "X", "X"],
        [None, 0, 1, 0, 3, 4],
        prefix="largest",
        node_keys=("r", "a", "b", "n", "c", "d"),
    )
    model = NamedVertexCoarsener(labels={"X"}, component_policy="largest", model_id="largest").fit(
        [graph]
    )
    encoded = model.transform(graph)
    composites = [
        data
        for _, data in encoded.nodes(data=True)
        if data["label"] == ("named_component", "largest")
    ]
    assert len(composites) == 1
    assert composites[0]["super_uids"] == (("largest", 4), ("largest", 5))

    relabeled = nx.relabel_nodes(
        graph,
        {node: ("opaque", i) for i, node in enumerate(reversed(tuple(graph.nodes)))},
        copy=True,
    )
    again = model.transform(relabeled)
    again_composite = next(
        data
        for _, data in again.nodes(data=True)
        if data["label"] == ("named_component", "largest")
    )
    assert again_composite["super_uids"] == composites[0]["super_uids"]


def test_named_partial_decode_keeps_stage_active(chain4: nx.DiGraph) -> None:
    model = NamedVertexCoarsener(labels={"B"}, model_id="partial").fit([chain4])
    encoded = model.transform(chain4)
    target = next(
        node
        for node, data in encoded.nodes(data=True)
        if data["label"] == ("named_component", "partial")
    )
    partially = model.decode(encoded, target=target, by="node", recursive=False)
    validate_encoded_tree(partially)
    assert partially.graph[SCHEMA_KEY]["stages"][-1]["model_id"] == "partial"
    assert all(
        data["label"] != ("named_component", "partial") for _, data in partially.nodes(data=True)
    )
    # Full stage reversal remains legal even after all visible owned occurrences were expanded.
    assert raw_signature(model.decode(partially)) == raw_signature(chain4)


def test_named_component_order_does_not_rescan_whole_tree_per_component(monkeypatch) -> None:
    import tree_coarsening.coarseners.named_vertices as named_module

    graph = make_tree(
        ["R", "A", "A", "X", "A", "A", "X", "A", "A"],
        [None, 0, 1, 0, 3, 4, 0, 6, 7],
        prefix="named-linear-order",
    )
    model = NamedVertexCoarsener(
        labels={"A"},
        model_id="named-linear-order",
    ).fit([graph], validate=False)
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
        named_module,
        "deterministic_node_order",
        lambda _graph: one_pass,
    )

    encoded = model.transform(graph, validate=False)

    assert one_pass.iterations == 1
    assert raw_signature(model.decode(encoded, validate=False)) == raw_signature(graph)
