"""Raw normalization and graph-level schema-1 metadata."""

from __future__ import annotations

from collections.abc import Hashable as HashableABC, Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any, Literal, TypeAlias

import networkx as nx

from .encoder import Vocabulary
from .exceptions import (
    AttachmentError,
    FittingSizeError,
    GraphSchemaError,
    LabelMetadataError,
    ProvenanceError,
    StageOrderError,
)
from .provenance import (
    NODE_ATTRS_KEY,
    PROVENANCE_KEY,
    normalize_provenance,
    normalize_super_uids,
    snapshot_raw_provenance,
)
from .structural import ExactType, MatchingLabel, base_type

SCHEMA_VERSION = "1.0"
SCHEMA_KEY = "tree_coarsening_schema"
FITTING_SIZES_KEY = "tree_coarsening_fitting_sizes"
NODE_FIELDS = ("label", "type", "size", "time", "super_uids")
EDGE_FIELDS = ("attach_map",)
RESERVED_GRAPH_KEYS = (SCHEMA_KEY, FITTING_SIZES_KEY, PROVENANCE_KEY)

ValidationLevel: TypeAlias = Literal["full", "structural", False]
StageRecord: TypeAlias = dict[str, Any]


def normalize_validation_level(value: Any) -> ValidationLevel:
    if value is False or value == "full" or value == "structural":
        return value
    raise ValueError("validate must be 'full', 'structural', or False.")


def normalize_attach_map(value: Any, *, context: str = "attach_map") -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise AttachmentError(f"{context} must be a tuple; got {type(value).__name__}.")
    for i, site in enumerate(value):
        if not isinstance(site, int) or isinstance(site, bool):
            raise AttachmentError(f"{context}[{i}] must be an integer; got {site!r}.")
    return value


def normalize_fitting_sizes(value: Any) -> dict[MatchingLabel, int]:
    if not isinstance(value, Mapping):
        raise FittingSizeError(f"{FITTING_SIZES_KEY!r} must be a mapping.")
    out: dict[MatchingLabel, int] = {}
    for label, size in value.items():
        if not isinstance(label, HashableABC):
            raise LabelMetadataError(f"fitting-size label must be hashable: {label!r}.")
        try:
            hash(label)
        except Exception as exc:
            raise LabelMetadataError(
                f"fitting-size label cannot be hashed reliably: {label!r}."
            ) from exc
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise FittingSizeError(
                f"fitting size for {label!r} must be a positive integer; got {size!r}."
            )
        out[label] = int(size)
    return out


def normalize_stage_records(value: Any) -> tuple[StageRecord, ...]:
    if not isinstance(value, tuple):
        raise GraphSchemaError("schema 'stages' must be a tuple.")
    out: list[StageRecord] = []
    seen: set[str] = set()
    introduced_by: dict[MatchingLabel, str] = {}
    for i, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise GraphSchemaError(f"stage {i} must be a mapping.")
        if set(record) != {"model_id", "introduced_labels"}:
            raise GraphSchemaError(
                f"stage {i} must contain exactly 'model_id' and 'introduced_labels'."
            )
        model_id = record["model_id"]
        labels = record["introduced_labels"]
        if not isinstance(model_id, str) or not model_id:
            raise GraphSchemaError(f"stage {i} has invalid model_id {model_id!r}.")
        if model_id in seen:
            raise StageOrderError(f"duplicate active model_id {model_id!r}.")
        seen.add(model_id)
        if not isinstance(labels, tuple):
            raise GraphSchemaError(f"stage {i} introduced_labels must be a tuple.")
        stage_labels: set[MatchingLabel] = set()
        for label in labels:
            if not isinstance(label, HashableABC):
                raise LabelMetadataError(f"stage {i} introduced label is not hashable: {label!r}.")
            try:
                hash(label)
            except Exception as exc:
                raise LabelMetadataError(
                    f"stage {i} introduced label cannot be hashed reliably: {label!r}."
                ) from exc
            if label in stage_labels:
                raise GraphSchemaError(f"stage {i} introduced_labels contains duplicates.")
            stage_labels.add(label)
            previous_owner = introduced_by.get(label)
            if previous_owner is not None:
                raise StageOrderError(
                    f"label {label!r} is introduced by more than one stage: "
                    f"{previous_owner!r} and {model_id!r}."
                )
            introduced_by[label] = model_id
        out.append({"model_id": model_id, "introduced_labels": tuple(labels)})
    return tuple(out)


def normalize_schema_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GraphSchemaError(f"{SCHEMA_KEY!r} must be a mapping.")
    if set(value) != {"version", "stages"}:
        raise GraphSchemaError("schema record must contain exactly 'version' and 'stages'.")
    version = value["version"]
    if version != SCHEMA_VERSION:
        raise GraphSchemaError(
            f"unsupported tree-coarsening schema {version!r}; expected {SCHEMA_VERSION!r}."
        )
    return {"version": SCHEMA_VERSION, "stages": normalize_stage_records(value["stages"])}


def is_encoded_graph(graph: nx.DiGraph) -> bool:
    return SCHEMA_KEY in graph.graph


def graph_stages(graph: nx.DiGraph) -> tuple[StageRecord, ...]:
    return normalize_schema_record(graph.graph[SCHEMA_KEY])["stages"]


def get_graph_fitting_sizes(graph: nx.DiGraph) -> dict[MatchingLabel, int]:
    return normalize_fitting_sizes(graph.graph[FITTING_SIZES_KEY])


def get_graph_provenance(graph: nx.DiGraph) -> dict[str, Any]:
    return normalize_provenance(graph.graph[PROVENANCE_KEY])


def representative_time(
    super_uids: tuple[Any, ...],
    provenance: Mapping[str, Any],
) -> float:
    flat = normalize_super_uids(super_uids)
    node_attrs = provenance[NODE_ATTRS_KEY]
    values: list[float] = []
    for uid in flat:
        try:
            raw_time = node_attrs[uid]["time"]
        except KeyError as exc:
            raise ProvenanceError(f"missing provenance time for UID {uid!r}.") from exc
        if (
            not isinstance(raw_time, Real)
            or isinstance(raw_time, bool)
            or not isfinite(float(raw_time))
        ):
            raise ProvenanceError(f"invalid provenance time for UID {uid!r}: {raw_time!r}.")
        values.append(float(raw_time))
    return max(values)


def encoded_node_attrs(
    *,
    label: MatchingLabel,
    type: ExactType,
    size: int,
    time: float,
    super_uids: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        "label": label,
        "type": type,
        "size": size,
        "time": float(time),
        "super_uids": tuple(super_uids),
    }


def _check_graph_type(graph: Any) -> None:
    if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
        raise GraphSchemaError("graph must be a non-multigraph networkx.DiGraph.")


def normalize_raw_graph(graph: nx.DiGraph) -> nx.DiGraph:
    from .validation import validate_raw_tree

    _check_graph_type(graph)
    validate_raw_tree(graph)
    provenance = snapshot_raw_provenance(graph)

    out = nx.DiGraph()
    out.graph[SCHEMA_KEY] = {"version": SCHEMA_VERSION, "stages": ()}
    out.graph[FITTING_SIZES_KEY] = {data["label"]: 1 for _node, data in graph.nodes(data=True)}
    out.graph[PROVENANCE_KEY] = provenance
    for node, data in graph.nodes(data=True):
        label = data["label"]
        out.add_node(
            node,
            **encoded_node_attrs(
                label=label,
                type=base_type(label),
                size=1,
                time=float(data["time"]),
                super_uids=(data["uid"],),
            ),
        )
    for parent, child in graph.edges:
        out.add_edge(parent, child, attach_map=(0,))
    return out


def copy_encoded_graph(graph: nx.DiGraph) -> nx.DiGraph:
    out = nx.DiGraph()
    schema = normalize_schema_record(graph.graph[SCHEMA_KEY])
    out.graph[SCHEMA_KEY] = {
        "version": schema["version"],
        "stages": tuple(dict(record) for record in schema["stages"]),
    }
    out.graph[FITTING_SIZES_KEY] = get_graph_fitting_sizes(graph)
    out.graph[PROVENANCE_KEY] = get_graph_provenance(graph)
    for node, data in graph.nodes(data=True):
        out.add_node(node, **dict(data))
    for parent, child, data in graph.edges(data=True):
        out.add_edge(parent, child, **dict(data))
    return out


def prepare_graph(
    graph: nx.DiGraph,
    *,
    validate: ValidationLevel = "full",
) -> nx.DiGraph:
    from .validation import validate_encoded_tree

    level = normalize_validation_level(validate)
    _check_graph_type(graph)
    reserved_present = [key for key in RESERVED_GRAPH_KEYS if key in graph.graph]
    if not reserved_present:
        return normalize_raw_graph(graph)
    if set(reserved_present) != set(RESERVED_GRAPH_KEYS):
        raise GraphSchemaError(
            "encoded metadata is incomplete; schema, fitting sizes, and provenance are all required."
        )
    out = copy_encoded_graph(graph)
    validate_encoded_tree(out, level=level)
    return out


def append_stage(
    graph: nx.DiGraph,
    *,
    model_id: str,
    vocab: Vocabulary,
) -> None:
    schema = normalize_schema_record(graph.graph[SCHEMA_KEY])
    stages = schema["stages"]
    active_ids = {record["model_id"] for record in stages}
    if model_id in active_ids:
        raise StageOrderError(f"model_id {model_id!r} is already active in this lineage.")

    sizes = get_graph_fitting_sizes(graph)
    introduced: list[MatchingLabel] = []
    for label in vocab.labels:
        size = vocab.fitting_size(label)
        previous = sizes.get(label)
        if previous is None:
            sizes[label] = size
            introduced.append(label)
        elif previous != size:
            raise FittingSizeError(
                f"stage label {label!r} has fitting size {size}, but input metadata has {previous}."
            )
    graph.graph[FITTING_SIZES_KEY] = sizes
    graph.graph[SCHEMA_KEY] = {
        "version": SCHEMA_VERSION,
        "stages": stages + ({"model_id": model_id, "introduced_labels": tuple(introduced)},),
    }


def pop_latest_stage(graph: nx.DiGraph, *, model_id: str) -> None:
    schema = normalize_schema_record(graph.graph[SCHEMA_KEY])
    stages = schema["stages"]
    if not stages or stages[-1]["model_id"] != model_id:
        actual = None if not stages else stages[-1]["model_id"]
        raise StageOrderError(
            f"decoder {model_id!r} does not own the latest active stage {actual!r}."
        )
    latest = stages[-1]
    sizes = get_graph_fitting_sizes(graph)
    for label in latest["introduced_labels"]:
        sizes.pop(label, None)
    graph.graph[FITTING_SIZES_KEY] = sizes
    graph.graph[SCHEMA_KEY] = {"version": SCHEMA_VERSION, "stages": stages[:-1]}


def fit_corpus_fitting_sizes(graphs: Sequence[nx.DiGraph]) -> dict[MatchingLabel, int]:
    out: dict[MatchingLabel, int] = {}
    for graph in graphs:
        for label, size in get_graph_fitting_sizes(graph).items():
            previous = out.get(label)
            if previous is not None and previous != size:
                raise FittingSizeError(
                    f"fit corpus gives label {label!r} conflicting sizes {previous} and {size}."
                )
            out[label] = size
    return out
