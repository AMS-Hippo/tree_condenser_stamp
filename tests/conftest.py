from __future__ import annotations

from collections.abc import Hashable, Sequence
from copy import deepcopy
from typing import Any

import networkx as nx
import pytest


def make_tree(
    labels: Sequence[str],
    parents: Sequence[int | None],
    *,
    prefix: str = "n",
    node_keys: Sequence[Hashable] | None = None,
) -> nx.DiGraph:
    assert len(labels) == len(parents)
    if node_keys is None:
        node_keys = tuple(range(len(labels)))
    assert len(node_keys) == len(labels)
    graph = nx.DiGraph()
    graph.graph["corpus"] = prefix
    graph.graph[17] = {"nested": [prefix]}
    for i, (node, label) in enumerate(zip(node_keys, labels, strict=True)):
        graph.add_node(
            node,
            label=label,
            time=float(i) / 10.0,
            uid=(prefix, i),
            payload={"index": i, "values": [label, i]},
        )
    for i, parent_i in enumerate(parents):
        if parent_i is not None:
            graph.add_edge(node_keys[parent_i], node_keys[i], ignored_raw_edge_attr=i)
    return graph


def raw_signature(graph: nx.DiGraph) -> tuple[Any, ...]:
    node_attrs = {data["uid"]: dict(data) for _node, data in graph.nodes(data=True)}
    edges = frozenset(
        (graph.nodes[parent]["uid"], graph.nodes[child]["uid"]) for parent, child in graph.edges
    )
    return node_attrs, edges, dict(graph.graph)


def encoded_signature(graph: nx.DiGraph) -> tuple[Any, ...]:
    order = tuple(nx.topological_sort(graph))
    position = {node: i for i, node in enumerate(order)}
    nodes = tuple(
        (
            graph.nodes[node]["label"],
            graph.nodes[node]["type"],
            graph.nodes[node]["size"],
            graph.nodes[node]["time"],
            graph.nodes[node]["super_uids"],
        )
        for node in order
    )
    edges = tuple(
        sorted(
            (
                position[parent],
                position[child],
                tuple(data["attach_map"]),
            )
            for parent, child, data in graph.edges(data=True)
        )
    )
    metadata = (
        graph.graph["tree_coarsening_schema"],
        graph.graph["tree_coarsening_fitting_sizes"],
        graph.graph["tree_coarsening_provenance"],
    )
    return nodes, edges, metadata


def encoded_occurrence_signature(graph: nx.DiGraph) -> tuple[Any, ...]:
    """Exact encoded semantics independent of package node numbering."""

    occurrence = {
        tuple(data["super_uids"]): (
            data["label"],
            data["type"],
            data["size"],
            data["time"],
        )
        for _node, data in graph.nodes(data=True)
    }
    edges = frozenset(
        (
            tuple(graph.nodes[parent]["super_uids"]),
            tuple(graph.nodes[child]["super_uids"]),
            tuple(data["attach_map"]),
        )
        for parent, child, data in graph.edges(data=True)
    )
    metadata = (
        graph.graph["tree_coarsening_schema"],
        graph.graph["tree_coarsening_fitting_sizes"],
        graph.graph["tree_coarsening_provenance"],
    )
    return occurrence, edges, metadata


def snapshot_graph(graph: nx.DiGraph) -> nx.DiGraph:
    return deepcopy(graph)


def assert_graph_unchanged(graph: nx.DiGraph, snapshot: nx.DiGraph) -> None:
    assert dict(graph.graph) == dict(snapshot.graph)
    assert dict(graph.nodes) == dict(snapshot.nodes)
    assert {(u, v): dict(data) for u, v, data in graph.edges(data=True)} == {
        (u, v): dict(data) for u, v, data in snapshot.edges(data=True)
    }


@pytest.fixture
def chain4() -> nx.DiGraph:
    return make_tree(["A", "B", "B", "C"], [None, 0, 1, 2], prefix="chain")


@pytest.fixture
def star4() -> nx.DiGraph:
    return make_tree(
        ["P", "C", "C", "C", "C"],
        [None, 0, 0, 0, 0],
        prefix="star",
    )
