"""A deliberately small schema-1 coarsener for configured vertices."""

from __future__ import annotations

from collections.abc import Collection, Hashable as HashableABC, Iterable, Sequence
from typing import Any, Hashable, Literal

import networkx as nx

from ..coarsener import TreeCoarsener
from ..contraction import RuleBasedEncoder
from ..decoder import TreeDecoder
from ..encoder import EncodingRule, TreeEncoder
from ..exceptions import ConfigurationError, ProvenanceError
from ..stage_decoder import StructuralStageDecoder
from ..validation import deterministic_node_order

SelectorKind = Literal["uid", "label"]
ComponentPolicy = Literal["all", "largest"]


def named_component_label(model_id: str) -> tuple[str, str]:
    return ("named_component", model_id)


class NamedVertexEncoder(RuleBasedEncoder):
    """Select configured occurrences; shared machinery performs all rewiring."""

    def __init__(
        self,
        *,
        model_id: str,
        rules: Sequence[EncodingRule],
        selector: SelectorKind,
        selected_values: frozenset[Hashable],
        component_policy: ComponentPolicy,
    ) -> None:
        super().__init__(model_id=model_id, rules=rules)
        if selector not in {"uid", "label"}:
            raise ConfigurationError("selector must be 'uid' or 'label'.")
        selected_values = frozenset(selected_values)
        if not selected_values:
            raise ConfigurationError("selected_values must not be empty.")
        if component_policy not in {"all", "largest"}:
            raise ConfigurationError("component_policy must be 'all' or 'largest'.")
        if len(self.rules) != 1:
            raise ConfigurationError("NamedVertexEncoder requires exactly one rule.")
        rule = self.rules[0]
        required_pattern_keys = {"selector", "values", "component_policy"}
        if set(rule.pattern) != required_pattern_keys:
            raise ConfigurationError(
                "named-component rule must contain exactly selector, values, and "
                "component_policy pattern keys."
            )
        expected_values = tuple(sorted(selected_values, key=repr))
        try:
            pattern_values = tuple(rule.pattern["values"])
        except TypeError as exc:
            raise ConfigurationError(
                "named-component rule pattern values must be an iterable collection."
            ) from exc
        if rule.operation != "component":
            raise ConfigurationError("named-component rule must use operation='component'.")
        if rule.output_label != named_component_label(model_id):
            raise ConfigurationError("named-component output label disagrees with model_id.")
        if rule.parameter_names != ("topology",):
            raise ConfigurationError("named-component rule must omit only 'topology'.")
        if (
            rule.pattern["selector"] != selector
            or pattern_values != expected_values
            or rule.pattern["component_policy"] != component_policy
        ):
            raise ConfigurationError(
                "named-component rule pattern disagrees with encoder selection state."
            )
        self.selector = selector
        self.selected_values = selected_values
        self.component_policy = component_policy

    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[Hashable]]:
        del rule
        selected: set[Hashable] = set()
        if self.selector == "label":
            selected = {
                node
                for node, data in graph.nodes(data=True)
                if data["label"] in self.selected_values
            }
        else:
            for node, data in graph.nodes(data=True):
                uids = set(data["super_uids"])
                overlap = uids & self.selected_values
                if overlap and overlap != uids:
                    raise ProvenanceError(
                        f"UID selection partially overlaps occurrence {node!r}: "
                        f"selected {overlap!r} of {uids!r}."
                    )
                if uids and uids <= self.selected_values:
                    selected.add(node)

        if not selected:
            return ()
        order = deterministic_node_order(graph)
        position = {node: i for i, node in enumerate(order)}
        components = [
            tuple(sorted(component, key=position.__getitem__))
            for component in nx.connected_components(
                graph.subgraph(selected).to_undirected(as_view=True)
            )
            if len(component) >= 2
        ]
        components.sort(key=lambda component: position[component[0]])
        if self.component_policy == "largest" and components:
            # Size is primary. Equal-size components are resolved by occurrence
            # semantics rather than current NetworkX keys, so relabeling an
            # otherwise identical input cannot change the selected component.
            def semantic_key(component: tuple[Hashable, ...]) -> tuple[Any, ...]:
                return tuple(
                    (
                        repr(graph.nodes[node]["label"]),
                        repr(graph.nodes[node]["type"]),
                        repr(graph.nodes[node]["super_uids"]),
                        repr(graph.nodes[node]["time"]),
                    )
                    for node in component
                )

            components = [max(components, key=lambda item: (len(item), semantic_key(item)))]
        return tuple(components)


class NamedVertexCoarsener(TreeCoarsener):
    """Contract configured maximal connected current-tree components."""

    def __init__(
        self,
        *,
        uids: Collection[Hashable] | None = None,
        labels: Collection[Hashable] | None = None,
        component_policy: ComponentPolicy = "all",
        fitting_size: int = 1,
        model_id: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id)
        if (uids is None) == (labels is None):
            raise ConfigurationError("exactly one of uids= and labels= must be supplied.")
        values: Any = uids if uids is not None else labels
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Collection):
            raise ConfigurationError("uids= and labels= must be explicit nonempty collections.")
        if not values:
            raise ConfigurationError("uids= and labels= must not be empty.")
        for value in values:
            if not isinstance(value, HashableABC):
                raise ConfigurationError(f"selection value must be hashable: {value!r}.")
            try:
                hash(value)
            except Exception as exc:
                raise ConfigurationError(
                    f"selection value cannot be hashed reliably: {value!r}."
                ) from exc
        if component_policy not in {"all", "largest"}:
            raise ConfigurationError("component_policy must be 'all' or 'largest'.")
        if not isinstance(fitting_size, int) or isinstance(fitting_size, bool) or fitting_size <= 0:
            raise ConfigurationError("fitting_size must be a positive integer.")

        self.selector: SelectorKind = "uid" if uids is not None else "label"
        self.selected_values = frozenset(values)
        self.component_policy = component_policy
        self.fitting_size = fitting_size

    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        del graphs
        rule = EncodingRule(
            rule_index=0,
            operation="component",
            output_label=named_component_label(self.model_id),
            output_fitting_size=self.fitting_size,
            pattern={
                "selector": self.selector,
                "values": tuple(sorted(self.selected_values, key=repr)),
                "component_policy": self.component_policy,
            },
            parameter_names=("topology",),
        )
        encoder = NamedVertexEncoder(
            model_id=self.model_id,
            rules=(rule,),
            selector=self.selector,
            selected_values=self.selected_values,
            component_policy=self.component_policy,
        )
        decoder = StructuralStageDecoder(model_id=self.model_id, rules=(rule,))
        return encoder, decoder
