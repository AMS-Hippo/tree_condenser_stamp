"""Parametric sibling-star coarsener for schema 1.0."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Any

import networkx as nx

from ..coarsener import TreeCoarsener
from ..contraction import Contraction, apply_mixed_contraction_batch
from ..decoder import TreeDecoder
from ..encoder import EncodingRule, TreeEncoder
from ..exceptions import ConfigurationError
from ..schema import append_stage, prepare_graph
from ..stage_decoder import StructuralStageDecoder
from ..validation import (
    deterministic_node_order,
    relabel_to_consecutive_parent_first,
    validate_encoded_tree,
)


def parametric_star_label(
    parent_label: Hashable,
    child_label: Hashable,
) -> tuple[Any, ...]:
    return ("star", parent_label, child_label)


class ParametricStarEncoder(TreeEncoder):
    """Apply every learned family once to the current input-tree view."""

    def __init__(
        self,
        *,
        model_id: str,
        rules: Sequence[EncodingRule],
        contract_d: int,
    ) -> None:
        super().__init__(model_id=model_id, rules=rules)
        if not isinstance(contract_d, int) or isinstance(contract_d, bool) or contract_d < 2:
            raise ConfigurationError("contract_d must be an integer of at least 2.")
        seen_pairs: set[tuple[Hashable, Hashable]] = set()
        required_pattern_keys = {
            "parent_label",
            "child_label",
            "witness_min_children",
            "contract_min_children",
        }
        for rule in self.rules:
            if set(rule.pattern) != required_pattern_keys:
                raise ConfigurationError(
                    f"parametric-star rule {rule.rule_index} must contain exactly "
                    f"pattern keys {sorted(required_pattern_keys)!r}."
                )
            parent_label = rule.pattern["parent_label"]
            child_label = rule.pattern["child_label"]
            pair = (parent_label, child_label)
            try:
                hash(pair)
            except Exception as exc:
                raise ConfigurationError(
                    f"parametric-star rule {rule.rule_index} labels must be hashable."
                ) from exc
            if pair in seen_pairs:
                raise ConfigurationError(f"parametric-star rules repeat label pair {pair!r}.")
            seen_pairs.add(pair)
            if rule.operation != "siblings":
                raise ConfigurationError(
                    f"parametric-star rule {rule.rule_index} must use operation='siblings'."
                )
            if rule.output_label != parametric_star_label(parent_label, child_label):
                raise ConfigurationError(
                    f"parametric-star rule {rule.rule_index} output label disagrees with "
                    "its matching pair."
                )
            if rule.output_fitting_size != 2:
                raise ConfigurationError(
                    f"parametric-star rule {rule.rule_index} must have fitting size 2."
                )
            if rule.parameter_names != ("arity",):
                raise ConfigurationError(
                    f"parametric-star rule {rule.rule_index} must omit only 'arity'."
                )
            witness_d = rule.pattern["witness_min_children"]
            rule_contract_d = rule.pattern["contract_min_children"]
            for name, threshold in (
                ("witness_min_children", witness_d),
                ("contract_min_children", rule_contract_d),
            ):
                if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 2:
                    raise ConfigurationError(
                        f"parametric-star rule {rule.rule_index} {name} must be an "
                        "integer of at least 2."
                    )
            if rule_contract_d != contract_d:
                raise ConfigurationError(
                    f"parametric-star rule {rule.rule_index} contract threshold disagrees "
                    "with encoder state."
                )
        self.contract_d = contract_d
        self._rule_waves = self._build_independent_rule_waves(self.rules)

    @classmethod
    def _build_independent_rule_waves(
        cls,
        rules: Sequence[EncodingRule],
    ) -> tuple[tuple[EncodingRule, ...], ...]:
        """Partition ordered rules into maximal semantics-preserving batches.

        A rule removes occurrences of its child label and creates its output
        label. A later rule therefore needs a new wave when either match label
        uses one of those changed labels. We also separate the conservative
        adjacent case in which the later child label is an earlier parent label.
        The accumulated label sets make this partition linear in rule count.
        """
        waves: list[tuple[EncodingRule, ...]] = []
        current: list[EncodingRule] = []
        changed_labels: set[Hashable] = set()
        parent_labels: set[Hashable] = set()
        for rule in rules:
            parent_label = rule.pattern["parent_label"]
            child_label = rule.pattern["child_label"]
            if current and (
                parent_label in changed_labels
                or child_label in changed_labels
                or child_label in parent_labels
            ):
                waves.append(tuple(current))
                current = []
                changed_labels.clear()
                parent_labels.clear()
            current.append(rule)
            changed_labels.update((child_label, rule.output_label))
            parent_labels.add(parent_label)
        if current:
            waves.append(tuple(current))
        return tuple(waves)

    def _independent_rule_waves(self) -> tuple[tuple[EncodingRule, ...], ...]:
        return self._rule_waves

    def transform(
        self,
        graph: nx.DiGraph,
        *,
        validate: str | bool = "full",
    ) -> nx.DiGraph:
        current = prepare_graph(graph, validate=validate)
        append_stage(current, model_id=self.model_id, vocab=self.vocab)

        # Rule indices define temporal order. Rules that cannot affect one
        # another's matching view share a wave; interacting rules begin a new
        # wave. Within each rule, a parent-first greedy scan preserves the
        # historical self-overlap policy. Every planned occurrence in a wave is
        # vertex-disjoint, so one graph rebuild is sufficient for the wave.
        for wave in self._independent_rule_waves():
            planned: list[tuple[EncodingRule, Contraction]] = []
            rules_by_parent: dict[Hashable, list[EncodingRule]] = {}
            selected_by_rule: dict[int, set[Hashable]] = {}
            for rule in wave:
                rules_by_parent.setdefault(rule.pattern["parent_label"], []).append(rule)
                selected_by_rule[rule.rule_index] = set()

            global_order = deterministic_node_order(current)
            global_position = {node: i for i, node in enumerate(global_order)}
            for parent in global_order:
                matching_rules = rules_by_parent.get(current.nodes[parent]["label"], ())
                if not matching_rules:
                    continue
                children_by_label: dict[Hashable, list[Hashable]] = {}
                for child in sorted(
                    current.successors(parent),
                    key=global_position.__getitem__,
                ):
                    children_by_label.setdefault(current.nodes[child]["label"], []).append(child)
                for rule in matching_rules:
                    selected_nodes = selected_by_rule[rule.rule_index]
                    if parent in selected_nodes:
                        continue
                    members = tuple(children_by_label.get(rule.pattern["child_label"], ()))
                    if len(members) < self.contract_d:
                        continue
                    planned.append(
                        (
                            rule,
                            Contraction(
                                rule_index=rule.rule_index,
                                nodes=members,
                            ),
                        )
                    )
                    selected_nodes.update(members)

            if planned:
                current = apply_mixed_contraction_batch(
                    current,
                    model_id=self.model_id,
                    planned=tuple(planned),
                    _validate_result=False,
                    _global_order=global_order,
                )
        current = relabel_to_consecutive_parent_first(current)
        validate_encoded_tree(current, level=validate)
        return current


class ParametricStarDecoder(StructuralStageDecoder):
    """Named artifact type for the generic stage decoder."""


class ParametricStarCoarsener(TreeCoarsener):
    """Learn label-pair star families while keeping arity occurrence-specific."""

    def __init__(
        self,
        d: int,
        m: int,
        *,
        contract_d: int | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id)
        if not isinstance(d, int) or isinstance(d, bool) or d < 2:
            raise ConfigurationError("d must be an integer of at least 2.")
        if not isinstance(m, int) or isinstance(m, bool) or m < 1:
            raise ConfigurationError("m must be a positive integer.")
        if contract_d is None:
            contract_d = d
        if not isinstance(contract_d, int) or isinstance(contract_d, bool) or contract_d < 2:
            raise ConfigurationError("contract_d must be an integer of at least 2.")
        self.d = d
        self.m = m
        self.contract_d = contract_d

    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        support: Counter[tuple[Hashable, Hashable]] = Counter()
        for graph in graphs:
            for parent, data in graph.nodes(data=True):
                child_counts = Counter(
                    graph.nodes[child]["label"] for child in graph.successors(parent)
                )
                for child_label, count in child_counts.items():
                    if count >= self.d:
                        support[(data["label"], child_label)] += 1

        learned = [
            (pair, count)
            for pair, count in sorted(
                support.items(),
                key=lambda item: (repr(item[0][0]), repr(item[0][1])),
            )
            if count >= self.m
        ]
        rules = tuple(
            EncodingRule(
                rule_index=index,
                operation="siblings",
                output_label=parametric_star_label(parent_label, child_label),
                output_fitting_size=2,
                pattern={
                    "parent_label": parent_label,
                    "child_label": child_label,
                    "witness_min_children": self.d,
                    "contract_min_children": self.contract_d,
                },
                parameter_names=("arity",),
                score=float(count),
            )
            for index, ((parent_label, child_label), count) in enumerate(learned)
        )
        return (
            ParametricStarEncoder(
                model_id=self.model_id,
                rules=rules,
                contract_d=self.contract_d,
            ),
            ParametricStarDecoder(model_id=self.model_id, rules=rules),
        )
