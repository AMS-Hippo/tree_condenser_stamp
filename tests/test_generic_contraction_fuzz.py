from __future__ import annotations

import random
from collections.abc import Hashable, Iterable, Sequence

import networkx as nx

from tree_coarsening import (
    EncodingRule,
    RuleBasedEncoder,
    StructuralStageDecoder,
    TreeCoarsener,
    TreeDecoder,
    TreeEncoder,
    validate_encoded_tree,
)

from conftest import make_tree, raw_signature


class _SelectedForestEncoder(RuleBasedEncoder):
    def __init__(
        self,
        *,
        model_id: str,
        rules: Sequence[EncodingRule],
        groups: Sequence[Sequence[Hashable]],
    ) -> None:
        super().__init__(model_id=model_id, rules=rules)
        self.groups = tuple(tuple(group) for group in groups)

    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[Hashable]]:
        del graph, rule
        return self.groups


class _SelectedForestCoarsener(TreeCoarsener):
    def __init__(
        self,
        *,
        groups: Sequence[Sequence[Hashable]],
        model_id: str,
    ) -> None:
        super().__init__(model_id=model_id)
        self.groups = tuple(tuple(group) for group in groups)

    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        del graphs
        rule = EncodingRule(
            0,
            "component",
            ("random-forest", self.model_id),
            1,
            {"selection": "configured-test-forest"},
            ("topology",),
        )
        rules = (rule,)
        return (
            _SelectedForestEncoder(
                model_id=self.model_id,
                rules=rules,
                groups=self.groups,
            ),
            StructuralStageDecoder(model_id=self.model_id, rules=rules),
        )


def _random_tree(seed: int, n: int = 24) -> nx.DiGraph:
    rng = random.Random(seed)
    parents: list[int | None] = [None]
    for node in range(1, n):
        # Favor recent parents but retain enough branching for multi-root forests.
        if rng.random() < 0.55:
            parent = rng.randrange(max(0, node - 5), node)
        else:
            parent = rng.randrange(node)
        parents.append(parent)
    labels = [chr(ord("A") + rng.randrange(4)) for _ in range(n)]
    return make_tree(labels, parents, prefix=f"forest-{seed}")


def _connected_downward_selection(
    graph: nx.DiGraph,
    root: int,
    rng: random.Random,
) -> set[int]:
    selected = {root}
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in graph.successors(parent):
            if rng.random() < 0.55:
                selected.add(child)
                stack.append(child)
    return selected


def _one_contractible_forest(graph: nx.DiGraph, seed: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    branching = [node for node in graph if graph.out_degree(node) >= 2]
    if branching and seed % 2:
        outside_parent = rng.choice(branching)
        roots = list(graph.successors(outside_parent))
        rng.shuffle(roots)
        roots = roots[: rng.randint(2, len(roots))]
        selected: set[int] = set()
        for root in roots:
            selected.update(_connected_downward_selection(graph, root, rng))
        return tuple(selected)

    selected = _connected_downward_selection(graph, 0, rng)
    if len(selected) == 1:
        selected.add(next(iter(graph.successors(0))))
    return tuple(selected)


def _disjoint_edge_groups(graph: nx.DiGraph) -> tuple[tuple[int, int], ...]:
    used: set[int] = set()
    groups: list[tuple[int, int]] = []
    for parent, child in graph.edges:
        if parent in used or child in used:
            continue
        groups.append((parent, child))
        used.update((parent, child))
        if len(groups) == 4:
            break
    return tuple(groups)


def test_random_contractible_forests_round_trip_exactly() -> None:
    for seed in range(80):
        raw = _random_tree(seed)
        group = _one_contractible_forest(raw, seed)
        model = _SelectedForestCoarsener(
            groups=(group,),
            model_id=f"random-forest-{seed}",
        ).fit([raw])
        encoded = model.transform(raw, validate=("full", "structural", False)[seed % 3])
        validate_encoded_tree(encoded, level="full")
        assert raw_signature(model.decode(encoded, validate=False)) == raw_signature(raw)


def test_random_simultaneous_disjoint_edge_batches_round_trip_exactly() -> None:
    for seed in range(40):
        raw = _random_tree(1000 + seed)
        groups = _disjoint_edge_groups(raw)
        model = _SelectedForestCoarsener(
            groups=groups,
            model_id=f"random-batch-{seed}",
        ).fit([raw])
        encoded = model.transform(raw, validate=False)
        validate_encoded_tree(encoded, level="full")
        assert raw_signature(model.decode(encoded, validate="structural")) == raw_signature(raw)
