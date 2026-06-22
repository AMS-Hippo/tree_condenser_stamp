"""Attachment-independent edge BPE for schema-1 directed labeled trees.

The learning and compact contraction core is preserved from v0.12.1. Schema-1
adaptation is isolated to graph normalization, exact occurrence capture, stage
metadata, and fitted artifact construction.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal

import networkx as nx

from ..coarsener import TreeCoarsener
from ..decoder import TreeDecoder
from ..encoder import EncodingRule, TreeEncoder
from ..exceptions import (
    ConfigurationError,
    FittingSizeError,
    LabelMetadataError,
    ValidationError,
)
from ..provenance import PROVENANCE_KEY, normalize_provenance
from ..schema import (
    FITTING_SIZES_KEY,
    SCHEMA_KEY,
    append_stage,
    encoded_node_attrs,
    fit_corpus_fitting_sizes,
    get_graph_fitting_sizes,
    normalize_attach_map,
    normalize_schema_record,
    prepare_graph,
)
from ..stage_decoder import StructuralStageDecoder
from ..structural import CompositeType, ExactType, is_base_type
from ..validation import validate_encoded_tree

Token = Hashable
EdgeKey = tuple[int, int]
PairScoreFunction = Callable[[int, int, int, int, int], float]
PairScoreName = Literal["count", "normalized", "size_weighted"]
PairScore = PairScoreName | PairScoreFunction


def max_component_time(*values: float) -> float:
    return max(float(value) for value in values)


def _has_raw_occurrence_geometry(graph: nx.DiGraph) -> bool:
    """Return whether visible occurrences are unchanged one-site raw nodes.

    A no-op stage still appends lineage metadata under schema 1.0.  That metadata
    must not by itself switch BPE to the historical encoded-node ordering and
    thereby change overlap selection.  The check is deliberately confined to
    the graph adapter; the compact BPE algorithm remains unchanged.
    """

    return all(
        is_base_type(data.get("type"))
        and data.get("size") == 1
        and isinstance(data.get("super_uids"), tuple)
        and len(data["super_uids"]) == 1
        for _, data in graph.nodes(data=True)
    )


def _stable_uid_order_key(value: Any) -> tuple[Any, ...]:
    """Order recommended primitive/tuple UIDs without comparing unlike types."""

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", int(value))
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_stable_uid_order_key(item) for item in value))
    if isinstance(value, frozenset):
        return (
            "frozenset",
            tuple(sorted(_stable_uid_order_key(item) for item in value)),
        )
    return (
        f"object:{type(value).__module__}.{type(value).__qualname__}",
        repr(value),
    )


def count_pair_score(
    n_ab: int,
    n_a: int,
    n_b: int,
    s_a: int,
    s_b: int,
) -> float:
    """Return the ordinary unweighted BPE score ``N(A, B)``."""

    del n_a, n_b, s_a, s_b
    return float(n_ab)


def normalized_pair_score(
    n_ab: int,
    n_a: int,
    n_b: int,
    s_a: int,
    s_b: int,
) -> float:
    """Return ``N(A,B) / sqrt(N(A) N(B))``.

    The pair count is positive for every eligible bucket, so both endpoint
    counts are positive as well.  The explicit guard produces a useful error
    if an incremental label-count invariant is ever violated.
    """

    del s_a, s_b
    if n_a <= 0 or n_b <= 0:
        raise ValidationError("normalized pair score requires positive endpoint occurrence counts.")
    return float(n_ab) / math.sqrt(float(n_a) * float(n_b))


def size_weighted_pair_score(
    n_ab: int,
    n_a: int,
    n_b: int,
    s_a: int,
    s_b: int,
) -> float:
    """Return ``N(A,B) * (S(A) + S(B))`` using label fitting sizes."""

    del n_a, n_b
    return float(n_ab) * float(s_a + s_b)


_BUILTIN_PAIR_SCORES: dict[PairScoreName, PairScoreFunction] = {
    "count": count_pair_score,
    "normalized": normalized_pair_score,
    "size_weighted": size_weighted_pair_score,
}

_NUMBA_PAIR_SCORE_MODES: dict[PairScoreName, int] = {
    "count": 0,
    "normalized": 1,
    "size_weighted": 2,
}


@dataclass(frozen=True, slots=True)
class _PairSelection:
    key: EdgeKey
    count: int
    parent_count: int
    child_count: int
    parent_size: int
    child_size: int
    score: float


def edge_bpe_token(model_id: str, rank: int) -> tuple[str, str, int]:
    """Return the stage-namespaced matching label for one BPE rule."""

    if not isinstance(model_id, str) or not model_id:
        raise ValidationError("edge-BPE model_id must be a nonempty string.")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValidationError("edge-BPE rank must be a nonnegative integer.")
    return ("edge_bpe", model_id, int(rank))


@dataclass(frozen=True, slots=True)
class EdgeBPERule:
    """One learned label-pair contraction rule."""

    rank: int
    token: Token
    parent_label: Token
    child_label: Token
    count: int
    score: float | None = None
    parent_count: int | None = None
    child_count: int | None = None
    parent_size: int | None = None
    child_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 0:
            raise ConfigurationError("edge-BPE rank must be a nonnegative integer.")
        for name, value in (
            ("token", self.token),
            ("parent_label", self.parent_label),
            ("child_label", self.child_label),
        ):
            try:
                hash(value)
            except Exception as exc:
                raise ConfigurationError(
                    f"edge-BPE {name} must be hashable; got {value!r}."
                ) from exc
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count <= 0:
            raise ConfigurationError("edge-BPE count must be a positive integer.")
        if self.score is not None:
            if not isinstance(self.score, Real) or isinstance(self.score, bool):
                raise ConfigurationError("edge-BPE score must be a finite real or None.")
            if not math.isfinite(float(self.score)):
                raise ConfigurationError("edge-BPE score must be a finite real or None.")
        for name, value in (
            ("parent_count", self.parent_count),
            ("child_count", self.child_count),
            ("parent_size", self.parent_size),
            ("child_size", self.child_size),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ConfigurationError(f"edge-BPE {name} must be a positive integer or None.")

    @property
    def parent_token(self) -> Token:
        return self.parent_label

    @property
    def child_token(self) -> Token:
        return self.child_label


@dataclass(slots=True)
class _TokenCodec:
    token_to_id: dict[Token, int] = field(default_factory=dict)
    id_to_token: list[Token] = field(default_factory=list)
    sort_key_by_id: list[str] = field(default_factory=list)

    def intern(self, token: Token) -> int:
        existing = self.token_to_id.get(token)
        if existing is not None:
            return existing
        token_id = len(self.id_to_token)
        self.token_to_id[token] = token_id
        self.id_to_token.append(token)
        self.sort_key_by_id.append(repr(token))
        return token_id

    def decode(self, token_id: int) -> Token:
        return self.id_to_token[token_id]

    def sort_key(self, token_id: int) -> str:
        return self.sort_key_by_id[token_id]


@dataclass(slots=True)
class _UidRope:
    """Append-only flat UID provenance rope used only while transforming."""

    leaves: list[tuple[Any, ...]]
    left: list[int] = field(default_factory=list)
    right: list[int] = field(default_factory=list)

    def merge(self, left_ref: int, right_ref: int) -> int:
        ref = len(self.leaves) + len(self.left)
        self.left.append(left_ref)
        self.right.append(right_ref)
        return ref

    def flatten(self, ref: int) -> tuple[Any, ...]:
        leaf_count = len(self.leaves)
        out: list[Any] = []
        stack = [ref]
        while stack:
            current = stack.pop()
            if current < leaf_count:
                out.extend(self.leaves[current])
                continue
            merge_index = current - leaf_count
            if merge_index < 0 or merge_index >= len(self.left):
                raise RuntimeError(f"invalid provenance-rope reference {current!r}.")
            stack.append(self.right[merge_index])
            stack.append(self.left[merge_index])
        return tuple(out)


@dataclass(slots=True)
class _BPEVocabulary:
    fitting_sizes: dict[Token, int]

    def fitting_size(self, label: Token) -> int:
        try:
            return self.fitting_sizes[label]
        except KeyError as exc:
            raise LabelMetadataError(f"no fitting size is registered for label {label!r}.") from exc

    def add_fitting_size(self, label: Token, size: int) -> None:
        previous = self.fitting_sizes.get(label)
        if previous is not None and previous != size:
            raise ValidationError(
                f"label {label!r} has conflicting fitting sizes {previous} and {size}."
            )
        self.fitting_sizes[label] = size


@dataclass(slots=True)
class _OutputContext:
    model_id: str
    schema: dict[str, Any]
    fitting_sizes: dict[Token, int]
    provenance: dict[str, Any]
    uid_rope: _UidRope


def _bump_count(counts: Counter[EdgeKey], key: EdgeKey, delta: int) -> None:
    value = counts.get(key, 0) + delta
    if value < 0:
        raise RuntimeError(f"edge-pair count for {key!r} became negative.")
    if value == 0:
        counts.pop(key, None)
    else:
        counts[key] = value


def _initial_label_statistics(
    states: Sequence["_CompactEdgeTree"],
    codec: _TokenCodec,
    vocab: _BPEVocabulary,
) -> tuple[list[int], list[int]]:
    """Return dense occurrence counts and fitting sizes by label id."""

    label_counts = [0] * len(codec.id_to_token)
    for state in states:
        for node, is_alive in enumerate(state.alive):
            if is_alive:
                label_counts[state.label[node]] += 1
    label_sizes = [vocab.fitting_size(codec.decode(i)) for i in range(len(codec.id_to_token))]
    return label_counts, label_sizes


def _set_new_label_statistics(
    label_counts: list[int],
    label_sizes: list[int],
    *,
    label_id: int,
    size: int,
) -> None:
    """Ensure dense arrays contain ``label_id`` and register its fixed size."""

    while len(label_counts) <= label_id:
        label_counts.append(0)
        label_sizes.append(0)
    label_counts[label_id] = 0
    label_sizes[label_id] = size


def _update_label_counts_after_merge(
    label_counts: list[int],
    *,
    parent_id: int,
    child_id: int,
    new_id: int,
    events: int,
) -> None:
    """Apply the occurrence-count change from ``events`` actual contractions."""

    if events < 0:
        raise RuntimeError("actual contraction count cannot be negative.")
    if parent_id == child_id:
        label_counts[parent_id] -= 2 * events
    else:
        label_counts[parent_id] -= events
        label_counts[child_id] -= events
    label_counts[new_id] += events
    if label_counts[parent_id] < 0 or label_counts[child_id] < 0:
        raise RuntimeError("incremental label occurrence count became negative.")


@dataclass(slots=True)
class _CompactEdgeTree:
    """Mutable array-backed state shared by fitting and transformation."""

    parent: list[int]
    children: list[list[int]]
    label: list[int]
    size: list[int]
    time: list[float]
    alive: list[bool]
    codec: _TokenCodec
    vocab: _BPEVocabulary
    edge_index: dict[EdgeKey, set[int]] = field(default_factory=lambda: defaultdict(set))

    output: _OutputContext | None = None
    type_token: list[ExactType] | None = None
    uid_ref: list[int] | None = None
    attach_to_parent: list[tuple[int, ...]] | None = None

    @classmethod
    def from_graph(
        cls,
        graph: nx.DiGraph,
        *,
        codec: _TokenCodec,
        vocab: _BPEVocabulary,
        model_id: str = "",
        pair_counts: Counter[EdgeKey] | None = None,
        capture_output: bool = False,
        build_edge_index: bool = True,
        input_is_normalized_raw: bool | None = None,
    ) -> "_CompactEdgeTree":
        roots = [node for node in graph if graph.in_degree(node) == 0]
        if len(roots) != 1:
            raise ValidationError(f"expected exactly one root; found {len(roots)}.")
        if input_is_normalized_raw is None:
            input_is_normalized_raw = _has_raw_occurrence_geometry(graph)

        root = roots[0]
        order: list[Any] = [root]
        seen = {root}
        parent = [-1]
        children: list[list[int]] = [[]]

        def child_key(child: Any) -> tuple[Any, ...]:
            data = graph.nodes[child]
            uids = data["super_uids"]
            semantic_tie: Any
            if input_is_normalized_raw:
                # Preserve the v0.12.1 raw-input ordering exactly.
                semantic_tie = repr(uids[0])
            else:
                # Earlier encoded graphs used synthetic integer node IDs here.
                # A structural UID key preserves their usual numeric ordering
                # without making schema-1 correctness depend on node keys.
                semantic_tie = _stable_uid_order_key(uids)
            return (
                float(data["time"]),
                repr(data["label"]),
                semantic_tie,
                repr(child),
            )

        cursor = 0
        while cursor < len(order):
            node = order[cursor]
            for child in sorted(graph.successors(node), key=child_key):
                if child in seen:
                    raise ValidationError(f"node {child!r} is reachable more than once.")
                child_i = len(order)
                seen.add(child)
                order.append(child)
                parent.append(cursor)
                children.append([])
                children[cursor].append(child_i)
            cursor += 1
        if len(order) != graph.number_of_nodes():
            raise ValidationError("not every node is reachable from the directed root.")

        labels = [codec.intern(graph.nodes[node]["label"]) for node in order]
        sizes = [int(graph.nodes[node]["size"]) for node in order]
        times = [float(graph.nodes[node]["time"]) for node in order]
        alive = [True] * len(order)

        output: _OutputContext | None = None
        types: list[ExactType] | None = None
        uid_ref: list[int] | None = None
        attachments: list[tuple[int, ...]] | None = None
        if capture_output:
            types = [graph.nodes[node]["type"] for node in order]
            uid_leaves = [tuple(graph.nodes[node]["super_uids"]) for node in order]
            uid_rope = _UidRope(uid_leaves)
            uid_ref = list(range(len(order)))
            attachments = []
            for i, node in enumerate(order):
                p = parent[i]
                if p == -1:
                    attachments.append(())
                else:
                    attachments.append(
                        normalize_attach_map(
                            graph.edges[order[p], node]["attach_map"],
                            context=f"edge {(order[p], node)!r}",
                        )
                    )
            output = _OutputContext(
                model_id=model_id,
                schema=normalize_schema_record(graph.graph[SCHEMA_KEY]),
                fitting_sizes=get_graph_fitting_sizes(graph),
                provenance=normalize_provenance(graph.graph[PROVENANCE_KEY]),
                uid_rope=uid_rope,
            )

        state = cls(
            parent=parent,
            children=children,
            label=labels,
            size=sizes,
            time=times,
            alive=alive,
            codec=codec,
            vocab=vocab,
            output=output,
            type_token=types,
            uid_ref=uid_ref,
            attach_to_parent=attachments,
        )
        if build_edge_index:
            state.rebuild_edge_index(pair_counts=pair_counts)
        return state

    @classmethod
    def from_raw_graph(cls, G: nx.DiGraph, **kwargs: Any) -> "_CompactEdgeTree":
        """Backward-compatible alias for the now-general ``from_graph``."""

        return cls.from_graph(G, **kwargs)

    def rebuild_edge_index(self, *, pair_counts: Counter[EdgeKey] | None = None) -> None:
        self.edge_index = defaultdict(set)
        for child in range(len(self.parent)):
            if self._edge_is_live(child):
                self._add_edge(child, pair_counts=pair_counts)

    def _edge_is_live(self, child: int) -> bool:
        if child < 0 or child >= len(self.alive) or not self.alive[child]:
            return False
        p = self.parent[child]
        return p >= 0 and p < len(self.alive) and self.alive[p]

    def _edge_key_unchecked(self, child: int) -> EdgeKey:
        p = self.parent[child]
        return (self.label[p], self.label[child])

    def _edge_key(self, child: int) -> EdgeKey:
        if not self._edge_is_live(child):
            raise ValidationError(f"node {child!r} does not have a live incoming edge.")
        return self._edge_key_unchecked(child)

    def _add_edge(self, child: int, *, pair_counts: Counter[EdgeKey] | None) -> None:
        key = self._edge_key_unchecked(child)
        self.edge_index[key].add(child)
        if pair_counts is not None:
            _bump_count(pair_counts, key, 1)

    def _remove_edge(self, child: int, *, pair_counts: Counter[EdgeKey] | None) -> None:
        key = self._edge_key_unchecked(child)
        bucket = self.edge_index.get(key)
        if bucket is None or child not in bucket:
            raise RuntimeError(f"live edge for child {child!r} is missing from its bucket.")
        bucket.remove(child)
        if not bucket:
            self.edge_index.pop(key, None)
        if pair_counts is not None:
            _bump_count(pair_counts, key, -1)

    def _edge_sort_key(self, child: int) -> tuple[float, float, int, int]:
        p = self.parent[child]
        return (self.time[child], self.time[p], p, child)

    def contract_pair(
        self,
        key: EdgeKey,
        *,
        new_label: int,
        pair_counts: Counter[EdgeKey] | None = None,
        rule_token: Token | None = None,
    ) -> int:
        bucket = self.edge_index.get(key)
        if not bucket:
            return 0
        candidates = sorted(bucket, key=self._edge_sort_key)
        used: set[int] = set()
        events = 0
        for child in candidates:
            if not self._edge_is_live(child):
                continue
            p = self.parent[child]
            if p in used or child in used or self._edge_key(child) != key:
                continue
            self._contract_edge(
                p,
                child,
                new_label=new_label,
                pair_counts=pair_counts,
                rule_token=rule_token,
            )
            used.add(p)
            used.add(child)
            events += 1
        return events

    def _contract_edge(
        self,
        parent_node: int,
        child_node: int,
        *,
        new_label: int,
        pair_counts: Counter[EdgeKey] | None,
        rule_token: Token | None,
    ) -> None:
        if not self._edge_is_live(child_node) or self.parent[child_node] != parent_node:
            raise ValidationError("attempted to contract a non-live edge occurrence.")

        grandparent = self.parent[parent_node]
        old_parent_children = self.children[parent_node]
        child_children = self.children[child_node]

        if grandparent != -1:
            self._remove_edge(parent_node, pair_counts=pair_counts)
        remaining: list[int] = []
        found = False
        for current in old_parent_children:
            self._remove_edge(current, pair_counts=pair_counts)
            if current == child_node:
                found = True
            else:
                remaining.append(current)
        if not found:
            raise RuntimeError("contracted child is missing from parent child list.")
        for current in child_children:
            self._remove_edge(current, pair_counts=pair_counts)

        parent_size = self.size[parent_node]
        self.label[parent_node] = new_label
        self.size[parent_node] += self.size[child_node]
        self.time[parent_node] = max_component_time(self.time[parent_node], self.time[child_node])

        if self.output is not None:
            if (
                self.type_token is None
                or self.uid_ref is None
                or self.attach_to_parent is None
                or rule_token is None
            ):
                raise RuntimeError("output-enabled compact state is incomplete.")
            parent_type = self.type_token[parent_node]
            child_type = self.type_token[child_node]
            contracted_map = self.attach_to_parent[child_node]
            self.type_token[parent_node] = CompositeType(
                model_id=self.output.model_id,
                label=rule_token,
                parent=(-1, 0),
                components=(parent_type, child_type),
                attach=contracted_map,
            )
            self.uid_ref[parent_node] = self.output.uid_rope.merge(
                self.uid_ref[parent_node], self.uid_ref[child_node]
            )
            self.uid_ref[child_node] = -1

        for current in child_children:
            self.parent[current] = parent_node
            if self.attach_to_parent is not None:
                self.attach_to_parent[current] = tuple(
                    parent_size + q for q in self.attach_to_parent[current]
                )
        remaining.extend(child_children)
        self.children[parent_node] = remaining

        self.alive[child_node] = False
        self.parent[child_node] = -1
        self.children[child_node] = []
        if self.attach_to_parent is not None:
            self.attach_to_parent[child_node] = ()

        if grandparent != -1:
            self._add_edge(parent_node, pair_counts=pair_counts)
        for current in remaining:
            self._add_edge(current, pair_counts=pair_counts)

    def to_networkx(self, *, validate: str | bool = "full") -> nx.DiGraph:
        if (
            self.output is None
            or self.type_token is None
            or self.uid_ref is None
            or self.attach_to_parent is None
        ):
            raise RuntimeError("fit-time compact states cannot be emitted as NetworkX.")
        context = self.output
        live = [node for node, keep in enumerate(self.alive) if keep]
        mapping = {old: new for new, old in enumerate(live)}
        out = nx.DiGraph()
        out.graph[SCHEMA_KEY] = context.schema
        out.graph[FITTING_SIZES_KEY] = dict(context.fitting_sizes)
        out.graph[PROVENANCE_KEY] = context.provenance
        for old in live:
            out.add_node(
                mapping[old],
                **encoded_node_attrs(
                    label=self.codec.decode(self.label[old]),
                    type=self.type_token[old],
                    size=self.size[old],
                    time=self.time[old],
                    super_uids=context.uid_rope.flatten(self.uid_ref[old]),
                ),
            )
        for old_child in live:
            old_parent = self.parent[old_child]
            if old_parent == -1:
                continue
            out.add_edge(
                mapping[old_parent],
                mapping[old_child],
                attach_map=self.attach_to_parent[old_child],
            )
        validate_encoded_tree(out, level=validate)
        return out


class EdgeBPEEncoder(TreeEncoder):
    """Apply fitted label-pair rules with occurrence-specific exact types."""

    def __init__(
        self,
        *,
        model_id: str,
        rules: Sequence[EncodingRule],
        edge_rules: Sequence[EdgeBPERule],
        input_labels: Sequence[Token],
    ) -> None:
        super().__init__(model_id=model_id, rules=rules)
        edge_rules = tuple(edge_rules)
        input_labels = tuple(input_labels)
        if len(edge_rules) != len(self.rules):
            raise ConfigurationError(
                "EdgeBPEEncoder generic rules and edge rules must have equal lengths."
            )
        if len(set(input_labels)) != len(input_labels):
            raise ConfigurationError("EdgeBPEEncoder input_labels must be unique.")
        available_labels = set(input_labels)
        for expected_rank, (rule, edge_rule) in enumerate(zip(self.rules, edge_rules, strict=True)):
            if not isinstance(edge_rule, EdgeBPERule):
                raise ConfigurationError("edge_rules must contain EdgeBPERule values.")
            if edge_rule.rank != expected_rank:
                raise ConfigurationError(
                    f"edge-BPE rule ranks must be consecutive; expected {expected_rank}, "
                    f"got {edge_rule.rank}."
                )
            expected_token = edge_bpe_token(model_id, expected_rank)
            if edge_rule.token != expected_token or rule.output_label != expected_token:
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} output label is not its stage-namespaced token."
                )
            if (
                edge_rule.parent_label not in available_labels
                or edge_rule.child_label not in available_labels
            ):
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} references a label unavailable at "
                    "that point in the ordered program."
                )
            if rule.operation != "edge":
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} must use operation='edge'."
                )
            expected_pattern_keys = {
                "parent_label",
                "child_label",
                "count_semantics",
                "raw_count",
                "actual_events",
                "pair_score",
                "parent_count",
                "child_count",
            }
            if set(rule.pattern) != expected_pattern_keys:
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} generic pattern disagrees with the "
                    "schema-1 artifact contract."
                )
            if rule.parameter_names:
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} must not declare parametric fields."
                )
            if (
                rule.pattern["parent_label"] != edge_rule.parent_label
                or rule.pattern["child_label"] != edge_rule.child_label
                or rule.pattern["count_semantics"] != "raw_matching_edges"
                or rule.pattern["raw_count"] != edge_rule.count
                or rule.pattern["parent_count"] != edge_rule.parent_count
                or rule.pattern["child_count"] != edge_rule.child_count
            ):
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} generic pattern disagrees with "
                    "its optimized rule record."
                )
            actual_events = rule.pattern["actual_events"]
            if (
                not isinstance(actual_events, int)
                or isinstance(actual_events, bool)
                or actual_events <= 0
                or actual_events > edge_rule.count
            ):
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} has invalid actual event count "
                    f"{actual_events!r}."
                )
            pair_score = rule.pattern["pair_score"]
            if not isinstance(pair_score, str) or not pair_score:
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} pair-score name must be nonempty."
                )
            if (
                edge_rule.parent_count is None
                or edge_rule.child_count is None
                or edge_rule.parent_size is None
                or edge_rule.child_size is None
            ):
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} is missing endpoint counts or fitting sizes."
                )
            expected_size = edge_rule.parent_size + edge_rule.child_size
            if rule.output_fitting_size != expected_size:
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} output fitting size disagrees with "
                    "its endpoint sizes."
                )
            if rule.score != edge_rule.score:
                raise ConfigurationError(
                    f"edge-BPE rule {expected_rank} score disagrees with its optimized rule record."
                )
            available_labels.add(expected_token)
        self.edge_rules = edge_rules
        self.input_labels = input_labels

    def transform(
        self,
        graph: nx.DiGraph,
        *,
        validate: str | bool = "full",
    ) -> nx.DiGraph:
        current = prepare_graph(graph, validate=validate)
        input_is_normalized_raw = _has_raw_occurrence_geometry(current)
        current_sizes = get_graph_fitting_sizes(current)
        expected_sizes = dict(current_sizes)
        for rule in self.edge_rules:
            for label, expected in (
                (rule.parent_label, rule.parent_size),
                (rule.child_label, rule.child_size),
            ):
                actual = expected_sizes.get(label)
                if actual is not None and actual != expected:
                    raise FittingSizeError(
                        f"edge-BPE rule {rule.rank} was fitted with label {label!r} "
                        f"at size {expected}, but the transform graph records {actual}."
                    )
            assert rule.parent_size is not None and rule.child_size is not None
            expected_sizes[rule.token] = rule.parent_size + rule.child_size

        append_stage(current, model_id=self.model_id, vocab=self.vocab)
        fit_vocab = _BPEVocabulary(get_graph_fitting_sizes(current))

        codec = _TokenCodec()
        for label in self.input_labels:
            codec.intern(label)
        for rule in self.edge_rules:
            codec.intern(rule.parent_label)
            codec.intern(rule.child_label)
            codec.intern(rule.token)

        state = _CompactEdgeTree.from_graph(
            current,
            codec=codec,
            vocab=fit_vocab,
            model_id=self.model_id,
            capture_output=True,
            input_is_normalized_raw=input_is_normalized_raw,
        )
        for rule in self.edge_rules:
            state.contract_pair(
                (codec.intern(rule.parent_label), codec.intern(rule.child_label)),
                new_label=codec.intern(rule.token),
                pair_counts=None,
                rule_token=rule.token,
            )
        return state.to_networkx(validate=validate)


class EdgeBPECoarsener(TreeCoarsener):
    """Learn ordinary BPE rules from parent/child matching labels only."""

    def __init__(
        self,
        *,
        num_merges: int | None = None,
        min_pair_count: int = 2,
        pair_score: PairScore = "count",
        backend: Literal["python", "numba"] = "python",
        model_id: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id)
        if num_merges is not None and (
            not isinstance(num_merges, int) or isinstance(num_merges, bool) or num_merges < 0
        ):
            raise ConfigurationError("num_merges must be None or a nonnegative integer.")
        if (
            not isinstance(min_pair_count, int)
            or isinstance(min_pair_count, bool)
            or min_pair_count < 1
        ):
            raise ConfigurationError("min_pair_count must be a positive integer.")
        if backend not in {"python", "numba"}:
            raise ConfigurationError("backend must be 'python' or 'numba'.")
        if isinstance(pair_score, str):
            if pair_score not in _BUILTIN_PAIR_SCORES:
                allowed = ", ".join(sorted(_BUILTIN_PAIR_SCORES))
                raise ConfigurationError(f"pair_score must be one of {allowed}, or a callable.")
            pair_score_name: PairScoreName | None = pair_score
            pair_score_function = _BUILTIN_PAIR_SCORES[pair_score]
        elif callable(pair_score):
            pair_score_name = None
            pair_score_function = pair_score
        else:
            raise ConfigurationError("pair_score must be a built-in score name or a callable.")
        if backend == "numba" and pair_score_name is None:
            raise ConfigurationError(
                "backend='numba' supports built-in pair_score values only; "
                "use backend='python' for a custom callable."
            )
        self.num_merges = num_merges
        self.min_pair_count = min_pair_count
        self.pair_score = pair_score
        self.pair_score_name_: PairScoreName | None = pair_score_name
        self._pair_score_function: PairScoreFunction = pair_score_function
        self.pair_score_display_name_ = (
            pair_score_name
            if pair_score_name is not None
            else getattr(pair_score_function, "__name__", "custom")
        )
        self.backend = backend
        self.backend_requested_ = backend
        self.backend_used_: Literal["python", "numba"] = "python"
        self.history_: list[dict[str, Any]] = []

    def _fit(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        use_numba = self.backend == "numba"
        numba_forest: Any | None = None
        backend_used: Literal["python", "numba"] = "python"
        if use_numba:
            from .edge_bpe_numba import NumbaTrainingForest, require_numba

            require_numba()
            backend_used = "numba"

        fitting_sizes = fit_corpus_fitting_sizes(graphs)
        input_labels = tuple(sorted(fitting_sizes, key=repr))
        vocab = _BPEVocabulary(dict(fitting_sizes))
        codec = _TokenCodec()
        counts: Counter[EdgeKey] = Counter()
        states: list[_CompactEdgeTree] = []
        for graph in graphs:
            states.append(
                _CompactEdgeTree.from_graph(
                    graph,
                    codec=codec,
                    vocab=vocab,
                    pair_counts=None if use_numba else counts,
                    capture_output=False,
                    build_edge_index=not use_numba,
                )
            )

        if use_numba:
            max_possible_merges = sum(len(state.parent) - 1 for state in states)
            if self.num_merges is not None:
                max_possible_merges = min(max_possible_merges, self.num_merges)
            initial_label_sizes = [
                vocab.fitting_size(codec.decode(label_id))
                for label_id in range(len(codec.id_to_token))
            ]
            numba_forest = NumbaTrainingForest.from_compact_states(
                states,
                label_capacity=len(codec.id_to_token) + max_possible_merges,
                initial_label_sizes=initial_label_sizes,
            )
            states.clear()
            label_counts: list[int] | None = None
            label_sizes: list[int] | None = None
        else:
            label_counts, label_sizes = _initial_label_statistics(states, codec, vocab)

        learned: list[EdgeBPERule] = []
        encoding_rules: list[EncodingRule] = []
        history: list[dict[str, Any]] = []
        rank = 0
        while self.num_merges is None or rank < self.num_merges:
            if numba_forest is None:
                if label_counts is None or label_sizes is None:
                    raise RuntimeError("Python BPE fitting is missing label statistics.")
                best = self._select_best_pair(counts, codec, label_counts, label_sizes)
            else:
                if self.pair_score_name_ is None:
                    raise RuntimeError("custom pair scorer reached Numba selection.")
                best = numba_forest.select_best_pair(
                    self.min_pair_count,
                    codec,
                    score_mode=_NUMBA_PAIR_SCORE_MODES[self.pair_score_name_],
                )
            if best is None:
                break
            key = best.key
            raw_count = best.count
            parent_id, child_id = key
            parent_label = codec.decode(parent_id)
            child_label = codec.decode(child_id)
            token = edge_bpe_token(self.model_id, rank)

            vocab.add_fitting_size(token, best.parent_size + best.child_size)
            new_id = codec.intern(token)
            if numba_forest is None:
                if label_counts is None or label_sizes is None:
                    raise RuntimeError("Python BPE fitting is missing label statistics.")
                _set_new_label_statistics(
                    label_counts,
                    label_sizes,
                    label_id=new_id,
                    size=best.parent_size + best.child_size,
                )
                actual_events = sum(
                    state.contract_pair(key, new_label=new_id, pair_counts=counts)
                    for state in states
                )
                _update_label_counts_after_merge(
                    label_counts,
                    parent_id=parent_id,
                    child_id=child_id,
                    new_id=new_id,
                    events=actual_events,
                )
            else:
                numba_forest.register_label(new_id, size=best.parent_size + best.child_size)
                actual_events = numba_forest.contract_pair(key, new_label=new_id)
            if actual_events == 0:
                vocab.fitting_sizes.pop(token, None)
                if numba_forest is None:
                    counts.pop(key, None)
                    continue
                raise RuntimeError(
                    "Numba pair index selected a positive-count label pair with no "
                    "contractible occurrence."
                )

            learned.append(
                EdgeBPERule(
                    rank=rank,
                    token=token,
                    parent_label=parent_label,
                    child_label=child_label,
                    count=raw_count,
                    score=best.score,
                    parent_count=best.parent_count,
                    child_count=best.child_count,
                    parent_size=best.parent_size,
                    child_size=best.child_size,
                )
            )
            encoding_rules.append(
                EncodingRule(
                    rule_index=rank,
                    operation="edge",
                    output_label=token,
                    output_fitting_size=best.parent_size + best.child_size,
                    pattern={
                        "parent_label": parent_label,
                        "child_label": child_label,
                        "count_semantics": "raw_matching_edges",
                        "raw_count": raw_count,
                        "actual_events": actual_events,
                        "pair_score": self.pair_score_display_name_,
                        "parent_count": best.parent_count,
                        "child_count": best.child_count,
                    },
                    score=best.score,
                )
            )
            history.append(
                {
                    "rank": rank,
                    "token": token,
                    "parent_label": parent_label,
                    "child_label": child_label,
                    "parent_token": parent_label,
                    "child_token": child_label,
                    "count": raw_count,
                    "count_semantics": "raw_matching_edges",
                    "parent_count": best.parent_count,
                    "child_count": best.child_count,
                    "parent_size": best.parent_size,
                    "child_size": best.child_size,
                    "score": best.score,
                    "pair_score": self.pair_score_display_name_,
                    "actual_events": actual_events,
                }
            )
            rank += 1

        rules = tuple(encoding_rules)
        encoder = EdgeBPEEncoder(
            model_id=self.model_id,
            rules=rules,
            edge_rules=tuple(learned),
            input_labels=input_labels,
        )
        decoder = StructuralStageDecoder(model_id=self.model_id, rules=rules)
        self.backend_used_ = backend_used
        self.history_ = history
        return encoder, decoder

    def _select_best_pair(
        self,
        counts: Counter[EdgeKey],
        codec: _TokenCodec,
        label_counts: Sequence[int],
        label_sizes: Sequence[int],
    ) -> _PairSelection | None:
        best: _PairSelection | None = None
        best_priority: tuple[float, int, str, str] | None = None
        for key, count in counts.items():
            if count < self.min_pair_count:
                continue
            parent_id, child_id = key
            parent_count = int(label_counts[parent_id])
            child_count = int(label_counts[child_id])
            parent_size = int(label_sizes[parent_id])
            child_size = int(label_sizes[child_id])
            try:
                score = float(
                    self._pair_score_function(
                        int(count),
                        parent_count,
                        child_count,
                        parent_size,
                        child_size,
                    )
                )
            except Exception as exc:
                raise ValidationError(
                    f"pair_score failed for pair "
                    f"({codec.decode(parent_id)!r}, {codec.decode(child_id)!r})."
                ) from exc
            if not math.isfinite(score):
                raise ValidationError(
                    f"pair_score returned non-finite value {score!r} for "
                    f"N(A,B)={count}, N(A)={parent_count}, N(B)={child_count}, "
                    f"S(A)={parent_size}, S(B)={child_size}."
                )
            priority = (
                score,
                count,
                codec.sort_key(parent_id),
                codec.sort_key(child_id),
            )
            if best_priority is None or priority > best_priority:
                best_priority = priority
                best = _PairSelection(
                    key=key,
                    count=int(count),
                    parent_count=parent_count,
                    child_count=child_count,
                    parent_size=parent_size,
                    child_size=child_size,
                    score=score,
                )
        return best
