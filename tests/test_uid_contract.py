from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pytest

from tree_coarsening import NamedVertexCoarsener, ProvenanceError, validate_raw_tree

from conftest import raw_signature


def _uid_tree(uids: tuple[object, ...]) -> nx.DiGraph:
    graph = nx.DiGraph()
    labels = ("R", "A", "A")
    for node, (label, uid) in enumerate(zip(labels, uids, strict=True)):
        graph.add_node(node, label=label, time=float(node), uid=uid, payload={"node": node})
        if node:
            graph.add_edge(node - 1, node)
    graph.graph["uid-kind"] = "custom"
    return graph


def test_tuple_valued_uids_remain_atomic_and_become_raw_node_keys() -> None:
    uids = (
        ("root", (0, 1)),
        ("branch", ("left", 2)),
        ("branch", ("right", 3)),
    )
    graph = _uid_tree(uids)
    model = NamedVertexCoarsener(
        labels={"A"},
        model_id="tuple-uids",
    ).fit([graph])

    encoded = model.transform(graph)
    composite = next(data for _, data in encoded.nodes(data=True) if data["size"] == 2)
    assert composite["super_uids"] == uids[1:]

    decoded = model.decode(encoded)
    assert set(decoded.nodes) == set(uids)
    assert raw_signature(decoded) == raw_signature(graph)


@dataclass(frozen=True)
class StableUID:
    namespace: str
    serial: int


def test_stable_custom_hashable_uid_round_trips_without_coercion() -> None:
    uids = tuple(StableUID("custom", i) for i in range(3))
    graph = _uid_tree(uids)
    model = NamedVertexCoarsener(
        labels={"A"},
        model_id="custom-uids",
    ).fit([graph])

    decoded = model.decode(model.transform(graph))

    assert set(decoded.nodes) == set(uids)
    assert all(isinstance(node, StableUID) for node in decoded.nodes)
    assert raw_signature(decoded) == raw_signature(graph)


def test_unhashable_raw_uid_is_rejected() -> None:
    graph = _uid_tree(("root", "middle", "leaf"))
    graph.nodes[1]["uid"] = ["not", "hashable"]

    with pytest.raises(ProvenanceError, match="must be hashable"):
        validate_raw_tree(graph)
