"""Raw-node and graph provenance helpers for schema 1.0."""

from __future__ import annotations

from collections.abc import Hashable as HashableABC, Mapping
from typing import Any

import networkx as nx

from .exceptions import ProvenanceError

PROVENANCE_KEY = "tree_coarsening_provenance"
NODE_ATTRS_KEY = "node_attrs_by_uid"
GRAPH_ATTRS_KEY = "graph_attrs"
RESERVED_PREFIX = "tree_coarsening_"


def require_uid(uid: Any, *, context: str = "uid") -> Any:
    if not isinstance(uid, HashableABC):
        raise ProvenanceError(f"{context} must be hashable; got {uid!r}.")
    try:
        hash(uid)
    except Exception as exc:
        raise ProvenanceError(f"{context} cannot be hashed reliably: {uid!r}.") from exc
    return uid


def normalize_super_uids(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise ProvenanceError(f"super_uids must be a tuple; got {type(value).__name__}.")
    if not value:
        raise ProvenanceError("super_uids must not be empty.")
    for i, uid in enumerate(value):
        require_uid(uid, context=f"super_uids[{i}]")
    return value


def split_super_uids(
    super_uids: tuple[Any, ...],
    component_sizes: tuple[int, ...],
) -> tuple[tuple[Any, ...], ...]:
    flat = normalize_super_uids(super_uids)
    pieces: list[tuple[Any, ...]] = []
    cursor = 0
    for i, size in enumerate(component_sizes):
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ProvenanceError(f"component_sizes[{i}] must be positive; got {size!r}.")
        pieces.append(tuple(flat[cursor : cursor + size]))
        cursor += size
    if cursor != len(flat):
        raise ProvenanceError(
            f"component sizes total {cursor}, but super_uids contains {len(flat)} values."
        )
    return tuple(pieces)


def snapshot_raw_provenance(graph: nx.DiGraph) -> dict[str, Any]:
    node_attrs: dict[Any, dict[str, Any]] = {}
    for node, data in graph.nodes(data=True):
        uid = require_uid(data["uid"], context=f"uid on node {node!r}")
        if uid in node_attrs:
            raise ProvenanceError(f"duplicate raw UID {uid!r}.")
        node_attrs[uid] = dict(data)
    graph_attrs = {
        key: value
        for key, value in graph.graph.items()
        if not (isinstance(key, str) and key.startswith(RESERVED_PREFIX))
    }
    return {NODE_ATTRS_KEY: node_attrs, GRAPH_ATTRS_KEY: graph_attrs}


def normalize_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceError("tree_coarsening_provenance must be a mapping.")
    if set(value) != {NODE_ATTRS_KEY, GRAPH_ATTRS_KEY}:
        raise ProvenanceError(
            "provenance must contain exactly 'node_attrs_by_uid' and 'graph_attrs'."
        )
    raw_nodes = value[NODE_ATTRS_KEY]
    raw_graph = value[GRAPH_ATTRS_KEY]
    if not isinstance(raw_nodes, Mapping):
        raise ProvenanceError("node_attrs_by_uid must be a mapping.")
    if not isinstance(raw_graph, Mapping):
        raise ProvenanceError("graph_attrs must be a mapping.")

    nodes: dict[Any, dict[str, Any]] = {}
    for uid, attrs in raw_nodes.items():
        require_uid(uid, context="provenance UID")
        if not isinstance(attrs, Mapping):
            raise ProvenanceError(f"provenance attributes for UID {uid!r} must be a mapping.")
        attrs_copy = dict(attrs)
        if "uid" not in attrs_copy or attrs_copy["uid"] != uid:
            raise ProvenanceError(
                f"provenance attributes for UID {uid!r} do not store the same 'uid'."
            )
        nodes[uid] = attrs_copy

    graph_attrs = dict(raw_graph)
    bad = [key for key in graph_attrs if isinstance(key, str) and key.startswith(RESERVED_PREFIX)]
    if bad:
        raise ProvenanceError(f"provenance graph_attrs contains reserved keys: {bad!r}.")
    return {NODE_ATTRS_KEY: nodes, GRAPH_ATTRS_KEY: graph_attrs}


def get_node_attrs_by_uid(graph: nx.DiGraph) -> dict[Any, dict[str, Any]]:
    return normalize_provenance(graph.graph[PROVENANCE_KEY])[NODE_ATTRS_KEY]


def copy_graph_provenance(source: nx.DiGraph, target: nx.DiGraph) -> None:
    target.graph[PROVENANCE_KEY] = normalize_provenance(source.graph[PROVENANCE_KEY])
