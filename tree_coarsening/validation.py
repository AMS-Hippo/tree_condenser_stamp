"""Schema-1 raw and encoded graph validation."""

from __future__ import annotations

from collections.abc import Hashable as HashableABC, Mapping
from math import isfinite
from numbers import Real
from typing import Any, Hashable

import networkx as nx

from .exceptions import (
    AttachmentError,
    ExactTypeError,
    FittingSizeError,
    GraphSchemaError,
    LabelMetadataError,
    ProvenanceError,
    StageOrderError,
    TreeStructureError,
)
from .provenance import (
    NODE_ATTRS_KEY,
    RESERVED_PREFIX,
    normalize_provenance,
    normalize_super_uids,
    require_uid,
)
from .schema import (
    FITTING_SIZES_KEY,
    NODE_FIELDS,
    PROVENANCE_KEY,
    RESERVED_GRAPH_KEYS,
    SCHEMA_KEY,
    ValidationLevel,
    get_graph_fitting_sizes,
    normalize_attach_map,
    normalize_schema_record,
    normalize_validation_level,
    representative_time,
)
from .structural import (
    CompositeType,
    exact_root_count,
    exact_site_count,
    exact_type_label,
    is_exact_type,
    iter_base_labels,
    iter_exact_types,
)


def _is_finite_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))


def _validate_basic_tree_shape(graph: nx.DiGraph) -> Hashable:
    if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
        raise GraphSchemaError("graph must be a non-multigraph networkx.DiGraph.")
    if graph.number_of_nodes() == 0:
        raise TreeStructureError("tree must be nonempty.")
    if graph.number_of_edges() != graph.number_of_nodes() - 1:
        raise TreeStructureError("a tree must satisfy |E| = |V| - 1.")
    roots = [node for node, degree in graph.in_degree() if degree == 0]
    if len(roots) != 1:
        raise TreeStructureError(f"tree must have exactly one root; found {len(roots)}.")
    root = roots[0]
    bad = [node for node, degree in graph.in_degree() if node != root and degree != 1]
    if bad:
        raise TreeStructureError(f"every nonroot must have in-degree one; invalid nodes: {bad!r}.")
    if not nx.is_directed_acyclic_graph(graph):
        raise TreeStructureError("tree must be acyclic.")
    if not nx.is_weakly_connected(graph):
        raise TreeStructureError("the underlying undirected graph must be connected.")
    return root


def validate_raw_tree(graph: nx.DiGraph) -> None:
    """Validate the normative raw graph contract."""

    _validate_basic_tree_shape(graph)
    reserved_graph = [
        key for key in graph.graph if isinstance(key, str) and key.startswith(RESERVED_PREFIX)
    ]
    if reserved_graph:
        raise GraphSchemaError(f"raw graph contains reserved attributes: {reserved_graph!r}.")

    seen_uids: set[Any] = set()
    for node, data in graph.nodes(data=True):
        for field in ("label", "time", "uid"):
            if field not in data:
                raise GraphSchemaError(f"raw node {node!r} is missing {field!r}.")
        reserved = [field for field in ("type", "size", "super_uids") if field in data]
        if reserved:
            raise GraphSchemaError(
                f"raw node {node!r} supplies reserved encoded fields: {reserved!r}."
            )
        if not isinstance(data["label"], str):
            raise LabelMetadataError(
                f"raw node {node!r} label must be a string; got {data['label']!r}."
            )
        if not _is_finite_real(data["time"]):
            raise GraphSchemaError(
                f"raw node {node!r} time must be finite and non-Boolean; got {data['time']!r}."
            )
        uid = require_uid(data["uid"], context=f"uid on raw node {node!r}")
        if uid in seen_uids:
            raise ProvenanceError(f"raw UID {uid!r} is not unique.")
        seen_uids.add(uid)


def _validate_composite_stage_order(
    exact: CompositeType,
    stage_index: Mapping[str, int],
) -> None:
    stack: list[CompositeType] = [exact]
    while stack:
        current = stack.pop()
        if current.model_id not in stage_index:
            raise StageOrderError(f"exact type references inactive model_id {current.model_id!r}.")
        current_index = stage_index[current.model_id]
        for i, parent_i in enumerate(current.parent):
            if parent_i != -1 and parent_i >= i:
                raise ExactTypeError(
                    "package-produced CompositeType components must place parents before "
                    f"children; component {i} points to {parent_i}."
                )
        for component in current.components:
            if isinstance(component, CompositeType):
                if component.model_id not in stage_index:
                    raise StageOrderError(
                        f"nested exact type references inactive model_id {component.model_id!r}."
                    )
                if stage_index[component.model_id] > current_index:
                    raise StageOrderError(
                        f"stage {current.model_id!r} contains later-stage type "
                        f"{component.model_id!r}."
                    )
                stack.append(component)


def validate_encoded_tree(
    graph: nx.DiGraph,
    *,
    level: ValidationLevel = "full",
) -> None:
    """Validate a schema-1 encoded graph.

    ``structural`` and ``False`` retain all checks needed for safe rewiring and
    decoding. ``full`` additionally scans the global provenance partition and
    representative times.
    """

    normalized_level = normalize_validation_level(level)
    root = _validate_basic_tree_shape(graph)
    missing_metadata = [key for key in RESERVED_GRAPH_KEYS if key not in graph.graph]
    if missing_metadata:
        raise GraphSchemaError(f"encoded graph is missing metadata: {missing_metadata!r}.")
    unknown_reserved = [
        key
        for key in graph.graph
        if isinstance(key, str)
        and key.startswith(RESERVED_PREFIX)
        and key not in RESERVED_GRAPH_KEYS
    ]
    if unknown_reserved:
        raise GraphSchemaError(
            f"encoded graph contains unknown reserved attributes: {unknown_reserved!r}."
        )

    schema = normalize_schema_record(graph.graph[SCHEMA_KEY])
    sizes = get_graph_fitting_sizes(graph)
    provenance = normalize_provenance(graph.graph[PROVENANCE_KEY])
    stages = schema["stages"]
    stage_index = {record["model_id"]: i for i, record in enumerate(stages)}

    reachable_labels: set[Any] = set()
    visible_uids: list[Any] = []
    for node, data in graph.nodes(data=True):
        missing = [field for field in NODE_FIELDS if field not in data]
        if missing:
            raise GraphSchemaError(f"encoded node {node!r} is missing fields {missing!r}.")
        label = data["label"]
        if not isinstance(label, HashableABC):
            raise LabelMetadataError(f"encoded node {node!r} label is not hashable: {label!r}.")
        try:
            hash(label)
        except Exception as exc:
            raise LabelMetadataError(
                f"encoded node {node!r} label cannot be hashed reliably: {label!r}."
            ) from exc
        exact = data["type"]
        if not is_exact_type(exact):
            raise ExactTypeError(f"encoded node {node!r} has invalid exact type {exact!r}.")
        if label != exact_type_label(exact):
            raise ExactTypeError(
                f"encoded node {node!r} label {label!r} disagrees with exact type label "
                f"{exact_type_label(exact)!r}."
            )
        size = data["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ExactTypeError(f"encoded node {node!r} has invalid size {size!r}.")
        exact_size = exact_site_count(exact)
        if size != exact_size:
            raise ExactTypeError(
                f"encoded node {node!r} size is {size}, but exact type has {exact_size} sites."
            )
        if not _is_finite_real(data["time"]):
            raise ProvenanceError(f"encoded node {node!r} has invalid time {data['time']!r}.")
        uids = normalize_super_uids(data["super_uids"])
        if len(uids) != size:
            raise ProvenanceError(
                f"encoded node {node!r} has size {size}, but {len(uids)} provenance UIDs."
            )
        if len(uids) != len(set(uids)):
            raise ProvenanceError(
                f"encoded node {node!r} repeats a provenance UID within one occurrence."
            )
        if normalized_level == "full":
            base_labels = tuple(iter_base_labels(exact))
            for site, (base_label, uid) in enumerate(zip(base_labels, uids, strict=True)):
                attrs = provenance[NODE_ATTRS_KEY].get(uid)
                if attrs is None:
                    raise ProvenanceError(
                        f"encoded node {node!r} site {site} references missing UID {uid!r}."
                    )
                provenance_label = attrs.get("label")
                if provenance_label != base_label:
                    raise ProvenanceError(
                        f"encoded node {node!r} site {site} has exact base label "
                        f"{base_label!r}, but UID {uid!r} has provenance label "
                        f"{provenance_label!r}."
                    )
        visible_uids.extend(uids)
        for nested in iter_exact_types(exact):
            reachable_labels.add(exact_type_label(nested))
        if isinstance(exact, CompositeType):
            _validate_composite_stage_order(exact, stage_index)

    if exact_root_count(graph.nodes[root]["type"]) != 1:
        raise ExactTypeError("the encoded root exact type must expose exactly one root.")

    for parent, child, data in graph.edges(data=True):
        if "attach_map" not in data:
            raise AttachmentError(f"encoded edge {(parent, child)!r} is missing 'attach_map'.")
        attach = normalize_attach_map(
            data["attach_map"], context=f"attach_map on edge {(parent, child)!r}"
        )
        expected = exact_root_count(graph.nodes[child]["type"])
        if len(attach) != expected:
            raise AttachmentError(
                f"edge {(parent, child)!r} has {len(attach)} attachment values; "
                f"child exposes {expected} roots."
            )
        parent_size = graph.nodes[parent]["size"]
        bad = tuple(site for site in attach if site < 0 or site >= parent_size)
        if bad:
            raise AttachmentError(
                f"edge {(parent, child)!r} has sites {bad!r} outside parent size {parent_size}."
            )

    raw_attrs = provenance[NODE_ATTRS_KEY]
    for uid, attrs in raw_attrs.items():
        if not isinstance(attrs.get("label"), str):
            raise ProvenanceError(f"raw provenance label for UID {uid!r} must be a string.")
        if not _is_finite_real(attrs.get("time")):
            raise ProvenanceError(f"raw provenance time for UID {uid!r} is invalid.")
        reserved = [field for field in ("type", "size", "super_uids") if field in attrs]
        if reserved:
            raise ProvenanceError(
                f"raw provenance for UID {uid!r} contains reserved fields: {reserved!r}."
            )
        base_label = attrs["label"]
        if sizes.get(base_label) != 1:
            raise FittingSizeError(f"raw base label {base_label!r} must have fitting size 1.")

    introduced = {label for record in stages for label in record["introduced_labels"]}
    raw_base_labels = {attrs["label"] for attrs in raw_attrs.values()}
    invalid_introductions = introduced & raw_base_labels
    if invalid_introductions:
        raise StageOrderError(
            "stage records cannot introduce labels already present in raw-base metadata: "
            f"{invalid_introductions!r}."
        )
    missing_introduced = introduced - set(sizes)
    if missing_introduced:
        raise FittingSizeError(
            f"active stages introduce labels absent from fitting sizes: {missing_introduced!r}."
        )
    missing_reachable = reachable_labels - set(sizes)
    if missing_reachable:
        raise FittingSizeError(
            f"reachable exact-type labels lack fitting sizes: {missing_reachable!r}."
        )

    if normalized_level == "full":
        expected_uids = set(raw_attrs)
        if len(visible_uids) != len(set(visible_uids)):
            raise ProvenanceError("visible super_uids do not form a one-time partition.")
        if set(visible_uids) != expected_uids:
            missing = expected_uids - set(visible_uids)
            extra = set(visible_uids) - expected_uids
            raise ProvenanceError(
                f"visible provenance partition mismatch; missing={missing!r}, extra={extra!r}."
            )
        for node, data in graph.nodes(data=True):
            expected_time = representative_time(data["super_uids"], provenance)
            if float(data["time"]) != expected_time:
                raise ProvenanceError(
                    f"encoded node {node!r} time {data['time']!r} is not representative "
                    f"maximum {expected_time!r}."
                )


def node_order_key(
    graph: nx.DiGraph,
    node: Hashable,
) -> tuple[str, str, str, str, str]:
    """Return a semantic, stable sibling-order key.

    Time, matching label, and provenance mirror the ordering used before the
    schema-1 refactor while avoiding dependence on package node keys. Exact
    type is a final semantic discriminator for consumer-supplied graphs.
    """

    data = graph.nodes[node]
    return (
        repr(data.get("time")),
        repr(data.get("label")),
        repr(data.get("super_uids")),
        repr(data.get("type")),
        f"{type(node).__module__}.{type(node).__qualname__}:{node!r}",
    )


def deterministic_node_order(graph: nx.DiGraph) -> tuple[Hashable, ...]:
    root = _validate_basic_tree_shape(graph)
    out: list[Hashable] = []
    stack: list[Hashable] = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        children = sorted(graph.successors(node), key=lambda child: node_order_key(graph, child))
        stack.extend(reversed(children))
    return tuple(out)


def relabel_to_consecutive_parent_first(graph: nx.DiGraph) -> nx.DiGraph:
    order = deterministic_node_order(graph)
    mapping = {node: i for i, node in enumerate(order)}
    out = nx.DiGraph()
    out.graph[SCHEMA_KEY] = normalize_schema_record(graph.graph[SCHEMA_KEY])
    out.graph[FITTING_SIZES_KEY] = get_graph_fitting_sizes(graph)
    out.graph[PROVENANCE_KEY] = normalize_provenance(graph.graph[PROVENANCE_KEY])
    for node in order:
        data = graph.nodes[node]
        out.add_node(mapping[node], **{field: data[field] for field in NODE_FIELDS})
    for parent, child, data in graph.edges(data=True):
        out.add_edge(
            mapping[parent],
            mapping[child],
            attach_map=normalize_attach_map(data["attach_map"]),
        )
    return out
