"""Generic occurrence-driven decoder for one fitted schema-1 stage."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Hashable
from typing import Any

import networkx as nx

from .decoder import TreeDecoder
from .exceptions import (
    BoundaryExpansionError,
    DecodeSelectionError,
    ExactTypeError,
    FittingSizeError,
    ProvenanceError,
    StageOrderError,
    TargetNotFoundError,
    TypeOwnershipError,
)
from .provenance import GRAPH_ATTRS_KEY, NODE_ATTRS_KEY, normalize_super_uids, split_super_uids
from .schema import (
    PROVENANCE_KEY,
    encoded_node_attrs,
    get_graph_fitting_sizes,
    graph_stages,
    normalize_attach_map,
    pop_latest_stage,
    prepare_graph,
    representative_time,
)
from .structural import (
    CompositeType,
    exact_root_count,
    exact_site_count,
    exact_type_label,
    is_base_type,
    iter_exact_types,
)
from .validation import (
    deterministic_node_order,
    relabel_to_consecutive_parent_first,
    validate_encoded_tree,
    validate_raw_tree,
)


class _BoundaryRequired(Exception):
    def __init__(self, child: Hashable) -> None:
        self.child = child


class StructuralStageDecoder(TreeDecoder):
    """Decode composites owned by one ``model_id`` using exact occurrence data."""

    def _owns(self, value: Any) -> bool:
        return isinstance(value, CompositeType) and value.model_id == self.model_id

    def decode(
        self,
        graph: nx.DiGraph,
        *,
        target: Hashable | None = None,
        by: str = "node",
        recursive: bool = True,
        boundary_policy: str = "expand",
        validate: str | bool = "full",
    ) -> nx.DiGraph:
        if by not in {"node", "label", "type"}:
            raise DecodeSelectionError("by must be 'node', 'label', or 'type'.")
        if not isinstance(recursive, bool):
            raise DecodeSelectionError("recursive must be Boolean.")
        if boundary_policy not in {"expand", "raise"}:
            raise DecodeSelectionError("boundary_policy must be 'expand' or 'raise'.")

        current = prepare_graph(graph, validate=validate)
        stages = graph_stages(current)
        active_ids = {record["model_id"] for record in stages}
        if self.model_id not in active_ids:
            raise StageOrderError(f"model_id {self.model_id!r} is not active in this graph.")
        if stages[-1]["model_id"] != self.model_id:
            raise StageOrderError(
                f"decoder {self.model_id!r} does not own latest stage {stages[-1]['model_id']!r}."
            )
        self._validate_artifact_against_graph(current, stages[-1])

        if target is None:
            while True:
                selected = tuple(
                    node
                    for node in deterministic_node_order(current)
                    if self._owns(current.nodes[node]["type"])
                )
                if not selected:
                    break
                self._expand_selected(
                    current,
                    selected,
                    recursive=True,
                    boundary_policy="expand",
                )
            leftovers = [
                (node, nested)
                for node, data in current.nodes(data=True)
                for nested in iter_exact_types(data["type"])
                if self._owns(nested)
            ]
            if leftovers:
                raise TypeOwnershipError(
                    f"complete decode left types owned by {self.model_id!r}: {leftovers!r}."
                )
            pop_latest_stage(current, model_id=self.model_id)
            if not graph_stages(current):
                return self._materialize_raw(current)
            current = relabel_to_consecutive_parent_first(current)
            validate_encoded_tree(current, level=validate)
            return current

        selected = self._select(current, target=target, by=by)
        self._expand_selected(
            current,
            selected,
            recursive=recursive,
            boundary_policy=boundary_policy,
        )
        current = relabel_to_consecutive_parent_first(current)
        validate_encoded_tree(current, level=validate)
        return current

    def _validate_artifact_against_graph(
        self,
        graph: nx.DiGraph,
        stage_record: dict[str, Any],
    ) -> None:
        """Reject same-ID decoder artifacts that do not describe this stage.

        Schema 1 deliberately stores no fitted-rule fingerprint in graph
        metadata. The strongest check available without changing that frozen
        schema is therefore vocabulary-based: every stage-introduced label and
        every owned composite label must belong to this decoder's vocabulary,
        and all declared fitting sizes must agree with the graph.
        """

        graph_sizes = get_graph_fitting_sizes(graph)
        vocab_sizes = self.vocab.as_dict()
        introduced = tuple(stage_record["introduced_labels"])

        missing_from_artifact = tuple(label for label in introduced if label not in vocab_sizes)
        if missing_from_artifact:
            raise TypeOwnershipError(
                f"decoder {self.model_id!r} does not declare stage-introduced labels "
                f"{missing_from_artifact!r}."
            )
        vocab_positions = {label: i for i, label in enumerate(self.vocab.labels)}
        introduced_positions = tuple(vocab_positions[label] for label in introduced)
        if introduced_positions != tuple(sorted(introduced_positions)):
            raise TypeOwnershipError(
                f"stage {self.model_id!r} introduced-label order disagrees with the "
                "decoder vocabulary."
            )

        for label, expected_size in vocab_sizes.items():
            actual_size = graph_sizes.get(label)
            if actual_size is None:
                raise FittingSizeError(
                    f"decoder {self.model_id!r} declares label {label!r}, but the active "
                    "graph has no fitting size for it."
                )
            if actual_size != expected_size:
                raise FittingSizeError(
                    f"decoder {self.model_id!r} declares label {label!r} with fitting "
                    f"size {expected_size}, but the graph records {actual_size}."
                )

        owned_labels = {
            nested.label
            for _node, data in graph.nodes(data=True)
            for nested in iter_exact_types(data["type"])
            if self._owns(nested)
        }
        undeclared = tuple(sorted(owned_labels - set(vocab_sizes), key=repr))
        if undeclared:
            raise TypeOwnershipError(
                f"decoder {self.model_id!r} does not declare owned composite labels {undeclared!r}."
            )

    def _select(
        self,
        graph: nx.DiGraph,
        *,
        target: Hashable,
        by: str,
    ) -> tuple[Hashable, ...]:
        order = deterministic_node_order(graph)
        if by == "node":
            if target not in graph:
                raise TargetNotFoundError(f"target node {target!r} is absent.")
            if not self._owns(graph.nodes[target]["type"]):
                raise TypeOwnershipError(
                    f"target node {target!r} is not owned by decoder {self.model_id!r}."
                )
            return (target,)

        field = "label" if by == "label" else "type"
        matches = tuple(
            node
            for node in order
            if self._owns(graph.nodes[node]["type"]) and graph.nodes[node][field] == target
        )
        if not matches:
            raise TargetNotFoundError(
                f"no occurrences owned by {self.model_id!r} match {field} {target!r}."
            )
        return matches

    def _expand_selected(
        self,
        graph: nx.DiGraph,
        selected: tuple[Hashable, ...],
        *,
        recursive: bool,
        boundary_policy: str,
    ) -> None:
        stack: list[tuple[Hashable, bool]] = [(node, recursive) for node in reversed(selected)]
        while stack:
            node, recurse_descendants = stack.pop()
            if node not in graph or not self._owns(graph.nodes[node]["type"]):
                continue
            try:
                new_nodes = self._expand_one(graph, node)
            except _BoundaryRequired as exc:
                if boundary_policy == "raise":
                    raise BoundaryExpansionError(
                        f"expanding node {node!r} would give child {exc.child!r} "
                        "multiple current parents."
                    ) from None
                if exc.child not in graph or not self._owns(graph.nodes[exc.child]["type"]):
                    owner = None
                    if exc.child in graph and isinstance(
                        graph.nodes[exc.child]["type"], CompositeType
                    ):
                        owner = graph.nodes[exc.child]["type"].model_id
                    raise BoundaryExpansionError(
                        f"boundary child {exc.child!r} is owned by {owner!r}, not the "
                        f"current stage {self.model_id!r}."
                    ) from None
                stack.append((node, recurse_descendants))
                stack.append((exc.child, False))
                continue

            if recurse_descendants:
                for new_node in reversed(new_nodes):
                    if new_node in graph and self._owns(graph.nodes[new_node]["type"]):
                        stack.append((new_node, True))

    def _expand_one(
        self,
        graph: nx.DiGraph,
        node: Hashable,
    ) -> tuple[Hashable, ...]:
        data = graph.nodes[node]
        exact = data["type"]
        if not self._owns(exact):
            return ()
        assert isinstance(exact, CompositeType)

        flat_uids = normalize_super_uids(data["super_uids"])
        component_sizes = tuple(exact_site_count(component) for component in exact.components)
        uid_pieces = split_super_uids(flat_uids, component_sizes)
        provenance = graph.graph[PROVENANCE_KEY]

        offsets: list[int] = []
        cursor = 0
        for size in component_sizes:
            offsets.append(cursor)
            cursor += size
        total_sites = cursor

        def locate(site: int) -> tuple[int, int]:
            if site < 0 or site >= total_sites:
                raise ExactTypeError(
                    f"outgoing site {site} is outside composite size {total_sites}."
                )
            component_i = bisect_right(offsets, site) - 1
            return component_i, site - offsets[component_i]

        outgoing: list[tuple[Hashable, int, tuple[int, ...]]] = []
        for _parent, child, edge_data in list(graph.out_edges(node, data=True)):
            attach = normalize_attach_map(
                edge_data["attach_map"], context=f"edge {(node, child)!r}"
            )
            routed = tuple(locate(site) for site in attach)
            component_ids = {component_i for component_i, _local in routed}
            if len(component_ids) != 1:
                raise _BoundaryRequired(child)
            component_i = routed[0][0]
            outgoing.append((child, component_i, tuple(local for _component_i, local in routed)))

        component_nodes = tuple(object() for _ in exact.components)
        for i, component in enumerate(exact.components):
            uids = uid_pieces[i]
            graph.add_node(
                component_nodes[i],
                **encoded_node_attrs(
                    label=exact_type_label(component),
                    type=component,
                    size=component_sizes[i],
                    time=representative_time(uids, provenance),
                    super_uids=uids,
                ),
            )

        for i, parent_i in enumerate(exact.parent):
            if parent_i == -1:
                continue
            graph.add_edge(
                component_nodes[parent_i],
                component_nodes[i],
                attach_map=exact.attachment_slice(i),
            )

        predecessors = tuple(graph.predecessors(node))
        if len(predecessors) > 1:
            raise ExactTypeError(f"encoded node {node!r} has multiple parents.")
        if predecessors:
            outside_parent = predecessors[0]
            incoming = normalize_attach_map(
                graph.edges[outside_parent, node]["attach_map"],
                context=f"incoming edge to {node!r}",
            )
            if len(incoming) != exact.root_count:
                raise ExactTypeError(
                    f"incoming edge to {node!r} has {len(incoming)} roots; "
                    f"expected {exact.root_count}."
                )
            cursor = 0
            for i in exact.root_positions:
                width = exact_root_count(exact.components[i])
                graph.add_edge(
                    outside_parent,
                    component_nodes[i],
                    attach_map=tuple(incoming[cursor : cursor + width]),
                )
                cursor += width
        elif exact.root_count != 1:
            raise ExactTypeError(f"root occurrence {node!r} expands to {exact.root_count} roots.")

        for child, component_i, local_map in outgoing:
            graph.add_edge(component_nodes[component_i], child, attach_map=local_map)
        graph.remove_node(node)
        return component_nodes

    def _materialize_raw(self, graph: nx.DiGraph) -> nx.DiGraph:
        provenance = graph.graph[PROVENANCE_KEY]
        raw_nodes = provenance[NODE_ATTRS_KEY]
        visible_uids = [
            uid
            for _node, data in graph.nodes(data=True)
            for uid in normalize_super_uids(data["super_uids"])
        ]
        if len(visible_uids) != len(set(visible_uids)):
            raise ProvenanceError(
                "cannot materialize raw data because visible super_uids are not unique."
            )
        expected_uids = set(raw_nodes)
        actual_uids = set(visible_uids)
        if actual_uids != expected_uids:
            raise ProvenanceError(
                "cannot materialize raw data from an incomplete provenance partition; "
                f"missing={expected_uids - actual_uids!r}, "
                f"extra={actual_uids - expected_uids!r}."
            )
        out = nx.DiGraph()
        out.graph.update(dict(provenance[GRAPH_ATTRS_KEY]))
        uid_for_node: dict[Hashable, Any] = {}
        materialized_uids: set[Any] = set()

        for node, data in graph.nodes(data=True):
            exact = data["type"]
            if not is_base_type(exact):
                raise ExactTypeError(
                    f"final raw materialization encountered non-base type {exact!r}."
                )
            uids = normalize_super_uids(data["super_uids"])
            if len(uids) != 1:
                raise ProvenanceError(
                    f"base occurrence {node!r} contains {len(uids)} UIDs rather than one."
                )
            uid = uids[0]
            if uid not in raw_nodes:
                raise ProvenanceError(f"missing raw provenance for UID {uid!r}.")
            if uid in materialized_uids:
                raise ProvenanceError(f"UID {uid!r} occurs in more than one final base occurrence.")
            provenance_label = raw_nodes[uid].get("label")
            exact_label = exact_type_label(exact)
            if provenance_label != exact_label:
                raise ProvenanceError(
                    f"final base occurrence for UID {uid!r} has exact label "
                    f"{exact_label!r}, but provenance records {provenance_label!r}."
                )
            materialized_uids.add(uid)
            out.add_node(uid, **dict(raw_nodes[uid]))
            uid_for_node[node] = uid

        expected_uids = set(raw_nodes)
        if materialized_uids != expected_uids:
            missing = expected_uids - materialized_uids
            extra = materialized_uids - expected_uids
            raise ProvenanceError(
                f"final provenance partition mismatch; missing={missing!r}, extra={extra!r}."
            )

        for parent, child, edge_data in graph.edges(data=True):
            attach = normalize_attach_map(edge_data["attach_map"])
            if attach != (0,):
                raise ExactTypeError(
                    f"raw materialization encountered non-atomic attachment {attach!r}."
                )
            out.add_edge(uid_for_node[parent], uid_for_node[child])
        validate_raw_tree(out)
        return out
