"""Generic schema-1 contraction machinery.

Ordinary coarseners choose current nodes for a rule; the shared engine builds
exact types, rewires attachments, updates lineage, and supplies a generic
occurrence-driven decoder.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .encoder import EncodingRule, TreeEncoder
from .exceptions import AttachmentError, InternalInvariantError, ValidationError
from .schema import (
    FITTING_SIZES_KEY,
    PROVENANCE_KEY,
    SCHEMA_KEY,
    append_stage,
    encoded_node_attrs,
    normalize_attach_map,
    prepare_graph,
)
from .structural import CompositeType, exact_root_count
from .validation import (
    deterministic_node_order,
    relabel_to_consecutive_parent_first,
    validate_encoded_tree,
)


@dataclass(frozen=True, slots=True)
class Contraction:
    """One concrete occurrence selected for a fitted rule."""

    rule_index: int
    nodes: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rule_index, int)
            or isinstance(self.rule_index, bool)
            or self.rule_index < 0
        ):
            raise ValidationError(f"invalid contraction rule_index {self.rule_index!r}.")
        if not isinstance(self.nodes, tuple) or not self.nodes:
            raise ValidationError("a contraction must contain a nonempty node tuple.")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValidationError("a contraction cannot contain one node more than once.")


class RuleBasedEncoder(TreeEncoder):
    """Shared encoder for methods expressible as ordered contraction batches."""

    @abstractmethod
    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[Hashable]]:
        """Return vertex-disjoint current-node selections for *rule*."""

    def transform(
        self,
        graph: nx.DiGraph,
        *,
        validate: str | bool = "full",
    ) -> nx.DiGraph:
        current = prepare_graph(graph, validate=validate)
        append_stage(current, model_id=self.model_id, vocab=self.vocab)
        for rule in self.rules:
            raw_groups = self.select_contractions(current, rule)
            contractions = tuple(
                Contraction(rule_index=rule.rule_index, nodes=tuple(group)) for group in raw_groups
            )
            if contractions:
                current = apply_contraction_batch(
                    current,
                    model_id=self.model_id,
                    rule=rule,
                    contractions=contractions,
                    _validate_result=False,
                )
        current = relabel_to_consecutive_parent_first(current)
        validate_encoded_tree(current, level=validate)
        return current


def _canonical_component_order(
    graph: nx.DiGraph,
    selected: set[Hashable],
    global_position: dict[Hashable, int],
) -> tuple[Hashable, ...]:
    missing = selected - graph.nodes
    if missing:
        raise ValidationError(f"contraction references nodes absent from graph: {missing!r}.")
    return tuple(sorted(selected, key=global_position.__getitem__))


def _component_geometry(
    graph: nx.DiGraph,
    component: tuple[Hashable, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    position = {node: i for i, node in enumerate(component)}
    parent_vector: list[int] = []
    site_offsets: list[int] = []
    root_offsets: list[int] = []
    site_cursor = 0
    root_cursor = 0
    for node in component:
        predecessors = tuple(graph.predecessors(node))
        if len(predecessors) > 1:
            raise InternalInvariantError(f"current tree node {node!r} has multiple parents.")
        parent = predecessors[0] if predecessors else None
        parent_i = position.get(parent, -1)
        parent_vector.append(parent_i)
        site_offsets.append(site_cursor)
        site_cursor += graph.nodes[node]["size"]
        root_offsets.append(root_cursor if parent_i == -1 else 0)
        if parent_i == -1:
            root_cursor += exact_root_count(graph.nodes[node]["type"])
    for i, parent_i in enumerate(parent_vector):
        if parent_i >= i:
            raise InternalInvariantError(
                f"canonical contraction order placed parent {parent_i} after child {i}."
            )
    return tuple(parent_vector), tuple(site_offsets), tuple(root_offsets)


def _validate_contractible(
    graph: nx.DiGraph,
    component: tuple[Hashable, ...],
    parent_vector: tuple[int, ...],
) -> None:
    selected = set(component)
    root_positions = [i for i, parent_i in enumerate(parent_vector) if parent_i == -1]
    graph_root = next(node for node, degree in graph.in_degree() if degree == 0)
    if graph_root in selected:
        if len(root_positions) != 1 or component[root_positions[0]] != graph_root:
            raise ValidationError(
                "a contraction containing the graph root must have exactly one selected root."
            )
        return

    outside_parents: list[Hashable] = []
    for i in root_positions:
        predecessors = tuple(graph.predecessors(component[i]))
        if len(predecessors) != 1:
            raise ValidationError(
                f"nonroot selected root {component[i]!r} does not have one current parent."
            )
        outside_parents.append(predecessors[0])
    if not outside_parents or len(set(outside_parents)) != 1:
        raise ValidationError(
            "all roots of a nonroot selected forest must share one current parent."
        )


def apply_mixed_contraction_batch(
    graph: nx.DiGraph,
    *,
    model_id: str,
    planned: Sequence[tuple[EncodingRule, Contraction]],
    _validate_result: bool = True,
    _global_order: tuple[Hashable, ...] | None = None,
) -> nx.DiGraph:
    """Apply one deterministic vertex-disjoint batch, possibly with different rules."""

    global_order = tuple(
        deterministic_node_order(graph) if _global_order is None else _global_order
    )
    if len(global_order) != graph.number_of_nodes() or set(global_order) != set(graph.nodes):
        raise InternalInvariantError("supplied global order does not cover the current graph once.")
    global_position = {node: i for i, node in enumerate(global_order)}
    canonical: list[tuple[EncodingRule, tuple[Hashable, ...]]] = []
    used: set[Hashable] = set()
    for rule, item in planned:
        if item.rule_index != rule.rule_index:
            raise ValidationError("contraction rule index does not match its supplied rule.")
        selected = set(item.nodes)
        overlap = selected & used
        if overlap:
            raise ValidationError(f"simultaneous contractions overlap at {overlap!r}.")
        component = _canonical_component_order(graph, selected, global_position)
        canonical.append((rule, component))
        used.update(selected)
    canonical.sort(key=lambda item: global_position[item[1][0]])

    replacement_for: dict[Hashable, object] = {}
    site_offset_for: dict[Hashable, int] = {}
    root_offset_for: dict[Hashable, int] = {}
    replacement_data: dict[object, dict[str, Any]] = {}

    for rule, component in canonical:
        parent_vector, site_offsets, root_offsets = _component_geometry(graph, component)
        _validate_contractible(graph, component, parent_vector)
        replacement = object()
        attach_values: list[int] = []
        for i, parent_i in enumerate(parent_vector):
            if parent_i == -1:
                continue
            parent_node = component[parent_i]
            child_node = component[i]
            attach_values.extend(
                normalize_attach_map(
                    graph.edges[parent_node, child_node]["attach_map"],
                    context=f"internal edge {(parent_node, child_node)!r}",
                )
            )
        exact = CompositeType(
            model_id=model_id,
            label=rule.output_label,
            parent=parent_vector,
            components=tuple(graph.nodes[node]["type"] for node in component),
            attach=tuple(attach_values),
        )
        super_uids = tuple(uid for node in component for uid in graph.nodes[node]["super_uids"])
        replacement_data[replacement] = encoded_node_attrs(
            label=rule.output_label,
            type=exact,
            size=sum(graph.nodes[node]["size"] for node in component),
            time=max(float(graph.nodes[node]["time"]) for node in component),
            super_uids=super_uids,
        )
        for i, node in enumerate(component):
            replacement_for[node] = replacement
            site_offset_for[node] = site_offsets[i]
            root_offset_for[node] = root_offsets[i]

    owner: dict[Hashable, Hashable | object] = {
        node: replacement_for.get(node, node) for node in graph.nodes
    }
    out = nx.DiGraph()
    out.graph[SCHEMA_KEY] = graph.graph[SCHEMA_KEY]
    out.graph[FITTING_SIZES_KEY] = graph.graph[FITTING_SIZES_KEY]
    out.graph[PROVENANCE_KEY] = graph.graph[PROVENANCE_KEY]
    for node, data in graph.nodes(data=True):
        if node not in used:
            out.add_node(
                node,
                **{field: data[field] for field in ("label", "type", "size", "time", "super_uids")},
            )
    for replacement, data in replacement_data.items():
        out.add_node(replacement, **data)

    edge_buffers: dict[tuple[Hashable | object, Hashable | object], list[int | None]] = {}
    for parent, child, data in graph.edges(data=True):
        new_parent = owner[parent]
        new_child = owner[child]
        if new_parent is new_child:
            continue
        old_map = normalize_attach_map(data["attach_map"], context=f"edge {(parent, child)!r}")
        parent_offset = site_offset_for.get(parent, 0)
        mapped = tuple(parent_offset + site for site in old_map)
        child_offset = root_offset_for.get(child, 0)
        total_roots = exact_root_count(out.nodes[new_child]["type"])
        key = (new_parent, new_child)
        buffer = edge_buffers.setdefault(key, [None] * total_roots)
        stop = child_offset + len(mapped)
        if stop > total_roots:
            raise AttachmentError("collapsed incoming attachment slice exceeds child roots.")
        for i, site in enumerate(mapped, start=child_offset):
            previous = buffer[i]
            if previous is not None and previous != site:
                raise AttachmentError(
                    f"contraction produces conflicting attachment values {previous} and {site}."
                )
            buffer[i] = site

    parents_by_child: dict[Hashable | object, set[Hashable | object]] = {}
    for (parent, child), buffer in edge_buffers.items():
        if any(value is None for value in buffer):
            raise ValidationError(
                "simultaneous contractions would split one collapsed child across parents."
            )
        parents_by_child.setdefault(child, set()).add(parent)
        out.add_edge(parent, child, attach_map=tuple(int(value) for value in buffer))
    bad_children = [child for child, parents in parents_by_child.items() if len(parents) > 1]
    if bad_children:
        raise ValidationError(
            "simultaneous contractions would give collapsed nodes multiple parents: "
            f"{bad_children!r}."
        )

    if _validate_result:
        out = relabel_to_consecutive_parent_first(out)
        validate_encoded_tree(out, level="structural")
    return out


def apply_contraction_batch(
    graph: nx.DiGraph,
    *,
    model_id: str,
    rule: EncodingRule,
    contractions: Sequence[Contraction],
    _validate_result: bool = True,
) -> nx.DiGraph:
    return apply_mixed_contraction_batch(
        graph,
        model_id=model_id,
        planned=tuple((rule, contraction) for contraction in contractions),
        _validate_result=_validate_result,
    )
