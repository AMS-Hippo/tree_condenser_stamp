from __future__ import annotations

import random
from collections.abc import Hashable

import networkx as nx

from tree_coarsening import ParametricStarCoarsener
from tree_coarsening.coarseners.parametric_star import ParametricStarEncoder
from tree_coarsening.contraction import Contraction, apply_mixed_contraction_batch
from tree_coarsening.schema import append_stage, prepare_graph
from tree_coarsening.validation import (
    node_order_key,
    relabel_to_consecutive_parent_first,
    validate_encoded_tree,
)


def _legacy_sequential_transform(
    encoder: ParametricStarEncoder,
    graph: nx.DiGraph,
) -> nx.DiGraph:
    """Reference the pre-batching parent-first application exactly."""

    current = prepare_graph(graph, validate=False)
    append_stage(current, model_id=encoder.model_id, vocab=encoder.vocab)
    for rule in encoder.rules:
        parent_label = rule.pattern["parent_label"]
        child_label = rule.pattern["child_label"]
        root = next(node for node, degree in current.in_degree() if degree == 0)
        stack: list[Hashable] = [root]
        while stack:
            parent = stack.pop()
            if parent not in current:
                continue
            if current.nodes[parent]["label"] == parent_label:
                members = tuple(
                    child
                    for child in sorted(
                        current.successors(parent),
                        key=lambda child: node_order_key(current, child),
                    )
                    if current.nodes[child]["label"] == child_label
                )
                if len(members) >= encoder.contract_d:
                    current = apply_mixed_contraction_batch(
                        current,
                        model_id=encoder.model_id,
                        planned=((rule, Contraction(rule.rule_index, members)),),
                        _validate_result=False,
                    )
            children_after = sorted(
                current.successors(parent),
                key=lambda child: node_order_key(current, child),
            )
            stack.extend(reversed(children_after))
    current = relabel_to_consecutive_parent_first(current)
    validate_encoded_tree(current, level=False)
    return current


def _random_tree(seed: int, node_count: int) -> nx.DiGraph:
    rng = random.Random(seed)
    graph = nx.DiGraph()
    for node in range(node_count):
        graph.add_node(
            node,
            label=rng.choice(("A", "B", "C")),
            time=float(rng.randrange(11)),
            uid=(seed, node),
        )
        if node:
            graph.add_edge(rng.randrange(node), node)
    return graph


def test_parametric_star_batching_is_differentially_identical_to_sequential() -> None:
    for seed in range(40):
        training = (
            _random_tree(10_000 + seed, 75),
            _random_tree(20_000 + seed, 60),
        )
        model = ParametricStarCoarsener(
            2,
            1,
            model_id=f"star-batch-differential-{seed}",
        ).fit(training, validate=False)
        graph = _random_tree(seed, 100)

        batched = model.encoder_.transform(graph, validate=False)
        sequential = _legacy_sequential_transform(model.encoder_, graph)

        assert nx.utils.graphs_equal(batched, sequential), f"batch drift at seed {seed}"


def _many_independent_star_rules(rule_count: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, label="ROOT", time=0.0, uid=("many-star", 0))
    next_node = 1
    for rule_i in range(rule_count):
        parent = next_node
        next_node += 1
        graph.add_node(
            parent,
            label=f"P{rule_i}",
            time=float(rule_i + 1),
            uid=("many-star", parent),
        )
        graph.add_edge(0, parent)
        for child_i in range(3):
            child = next_node
            next_node += 1
            graph.add_node(
                child,
                label=f"C{rule_i}",
                time=float(rule_i + 1) + 0.1 * (child_i + 1),
                uid=("many-star", child),
            )
            graph.add_edge(parent, child)
    return graph


def _raw_uid_signature(graph: nx.DiGraph) -> tuple[object, object]:
    nodes = {data["uid"]: (data["label"], data["time"]) for _node, data in graph.nodes(data=True)}
    edges = frozenset(
        (graph.nodes[parent]["uid"], graph.nodes[child]["uid"]) for parent, child in graph.edges
    )
    return nodes, edges


def test_wave_batching_matches_sequential_on_encoded_inputs() -> None:
    for seed in range(12):
        training = (
            _random_tree(30_000 + seed, 70),
            _random_tree(40_000 + seed, 65),
        )
        upstream = ParametricStarCoarsener(
            2,
            1,
            model_id=f"star-wave-upstream-{seed}",
        ).fit(training, validate=False)
        encoded_training = tuple(upstream.transform(graph, validate=False) for graph in training)
        downstream = ParametricStarCoarsener(
            2,
            1,
            model_id=f"star-wave-downstream-{seed}",
        ).fit(encoded_training, validate=False)
        graph = upstream.transform(
            _random_tree(50_000 + seed, 90),
            validate=False,
        )

        batched = downstream.encoder_.transform(graph, validate=False)
        sequential = _legacy_sequential_transform(downstream.encoder_, graph)

        assert nx.utils.graphs_equal(batched, sequential), (
            f"encoded-input wave drift at seed {seed}"
        )


def test_independent_star_rules_share_one_graph_batch(monkeypatch) -> None:
    import tree_coarsening.coarseners.parametric_star as star_module

    graph = _many_independent_star_rules(40)
    model = ParametricStarCoarsener(
        2,
        1,
        model_id="star-independent-wave",
    ).fit([graph], validate=False)
    assert len(model.encoder_.rules) == 40

    calls: list[int] = []
    original = star_module.apply_mixed_contraction_batch

    def recording_batch(*args, **kwargs):
        calls.append(len(kwargs["planned"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(star_module, "apply_mixed_contraction_batch", recording_batch)
    encoded = model.transform(graph, validate=False)

    assert calls == [40]
    assert _raw_uid_signature(model.decode(encoded, validate=False)) == _raw_uid_signature(graph)


def test_interacting_star_rules_remain_in_temporal_waves(monkeypatch) -> None:
    import tree_coarsening.coarseners.parametric_star as star_module

    graph = nx.DiGraph()
    labels = ("R", "P", "P", "C", "C", "C", "C")
    parents = (None, 0, 0, 1, 1, 2, 2)
    for node, label in enumerate(labels):
        graph.add_node(node, label=label, time=float(node), uid=("interacting", node))
        if parents[node] is not None:
            graph.add_edge(parents[node], node)

    model = ParametricStarCoarsener(
        2,
        1,
        model_id="star-interacting-waves",
    ).fit([graph], validate=False)
    calls: list[int] = []
    original = star_module.apply_mixed_contraction_batch

    def recording_batch(*args, **kwargs):
        calls.append(len(kwargs["planned"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(star_module, "apply_mixed_contraction_batch", recording_batch)
    encoded = model.transform(graph, validate=False)

    assert calls == [2, 1]
    assert _raw_uid_signature(model.decode(encoded, validate=False)) == _raw_uid_signature(graph)
