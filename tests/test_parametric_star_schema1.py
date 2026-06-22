from __future__ import annotations

import networkx as nx

import tree_coarsening.coarseners.parametric_star as parametric_star_module
from tree_coarsening import CompositeType, ParametricStarCoarsener, parametric_star_label

from conftest import make_tree, raw_signature


def star(arity: int, *, prefix: str) -> nx.DiGraph:
    return make_tree(
        ["P"] + ["C"] * arity,
        [None] + [0] * arity,
        prefix=prefix,
    )


def test_parametric_star_same_label_has_variable_exact_geometry() -> None:
    training = [star(3, prefix="fit3"), star(5, prefix="fit5")]
    model = ParametricStarCoarsener(3, 2, model_id="star-family").fit(training)
    expected_label = parametric_star_label("P", "C")
    assert len(model.encoder_.rules) == 1
    assert model.encoder_.rules[0].output_label == expected_label
    assert model.encoder_.rules[0].output_fitting_size == 2
    assert model.encoder_.rules[0].parameter_names == ("arity",)

    encoded3 = model.transform(star(3, prefix="out3"))
    encoded5 = model.transform(star(5, prefix="out5"))
    occurrence3 = next(
        data for _, data in encoded3.nodes(data=True) if data["label"] == expected_label
    )
    occurrence5 = next(
        data for _, data in encoded5.nodes(data=True) if data["label"] == expected_label
    )
    assert isinstance(occurrence3["type"], CompositeType)
    assert isinstance(occurrence5["type"], CompositeType)
    assert occurrence3["size"] == 3
    assert occurrence5["size"] == 5
    assert occurrence3["type"] != occurrence5["type"]
    assert encoded3.graph["tree_coarsening_fitting_sizes"][expected_label] == 2
    assert encoded5.graph["tree_coarsening_fitting_sizes"][expected_label] == 2
    assert raw_signature(model.decode(encoded3)) == raw_signature(star(3, prefix="out3"))
    assert raw_signature(model.decode(encoded5)) == raw_signature(star(5, prefix="out5"))


def test_parametric_star_contract_threshold_is_transform_time_only() -> None:
    fit_graph = star(4, prefix="fit")
    model = ParametricStarCoarsener(4, 1, contract_d=2, model_id="threshold").fit([fit_graph])
    smaller = star(2, prefix="small")
    encoded = model.transform(smaller)
    expected_label = parametric_star_label("P", "C")
    assert any(data["label"] == expected_label for _, data in encoded.nodes(data=True))
    assert raw_signature(model.decode(encoded)) == raw_signature(smaller)


def test_parametric_star_support_counts_witness_parents() -> None:
    graph = make_tree(
        ["R", "P", "C", "C", "C", "P", "C", "C", "C"],
        [None, 0, 1, 1, 1, 0, 5, 5, 5],
        prefix="support",
    )
    learned = ParametricStarCoarsener(3, 2, model_id="learned").fit([graph])
    absent = ParametricStarCoarsener(3, 3, model_id="absent").fit([graph])
    assert len(learned.encoder_.rules) == 1
    assert learned.encoder_.rules[0].score == 2.0
    assert absent.encoder_.rules == ()


def test_parametric_star_rule_indices_define_temporal_order() -> None:
    # The deterministic rule order is P->C followed by R->P. The descendant
    # contractions therefore occur before the ancestor sibling contraction,
    # and both exact families remain visible in the transformed tree.
    graph = make_tree(
        ["R", "P", "P", "C", "C", "C", "C"],
        [None, 0, 0, 1, 1, 2, 2],
        prefix="temporal-rules",
    )
    model = ParametricStarCoarsener(2, 1, model_id="temporal-star").fit([graph])
    assert [
        (rule.pattern["parent_label"], rule.pattern["child_label"]) for rule in model.encoder_.rules
    ] == [("P", "C"), ("R", "P")]

    encoded = model.transform(graph)
    visible_labels = {data["label"] for _, data in encoded.nodes(data=True)}
    assert parametric_star_label("P", "C") in visible_labels
    assert parametric_star_label("R", "P") in visible_labels
    assert raw_signature(model.decode(encoded)) == raw_signature(graph)


def test_parametric_star_parent_before_child_avoids_self_overlap() -> None:
    graph = make_tree(
        ["A", "A", "A", "A", "A", "A", "A"],
        [None, 0, 0, 1, 1, 2, 2],
        prefix="self-overlap",
    )
    model = ParametricStarCoarsener(2, 1, model_id="self-overlap-star").fit([graph])
    encoded = model.transform(graph)
    family = parametric_star_label("A", "A")
    occurrences = [data for _, data in encoded.nodes(data=True) if data["label"] == family]
    assert len(occurrences) == 1
    assert occurrences[0]["size"] == 2
    assert raw_signature(model.decode(encoded)) == raw_signature(graph)


def test_parametric_star_visits_new_current_parents_top_down() -> None:
    # The second fitted program learns both P->S and star(P,S)->T. Applying
    # P->S creates a new current parent occurrence; parent-before-child
    # traversal must subsequently visit it and apply the second family.
    graph = make_tree(
        ["P", "S", "T", "T", "S", "T", "T"],
        [None, 0, 1, 1, 0, 4, 4],
        prefix="dynamic-parent",
    )
    upstream = ParametricStarCoarsener(
        2,
        1,
        model_id="dynamic-upstream",
    ).fit([graph])
    upstream_encoded = upstream.transform(graph)

    model = ParametricStarCoarsener(
        2,
        1,
        model_id="dynamic-current",
    ).fit([graph, upstream_encoded])
    first_label = parametric_star_label("P", "S")
    second_label = parametric_star_label(first_label, "T")
    assert {
        (rule.pattern["parent_label"], rule.pattern["child_label"]) for rule in model.encoder_.rules
    } >= {("P", "S"), (first_label, "T")}

    encoded = model.transform(graph)
    assert any(data["label"] == first_label for _, data in encoded.nodes(data=True))
    assert any(data["label"] == second_label for _, data in encoded.nodes(data=True))
    assert raw_signature(model.decode(encoded)) == raw_signature(graph)


def test_parametric_star_batches_disjoint_occurrences_once_per_rule(monkeypatch) -> None:
    training = star(2, prefix="batch-fit")
    model = ParametricStarCoarsener(2, 1, model_id="batch-star").fit([training])

    groups = 80
    labels = ["R"]
    parents: list[int | None] = [None]
    for _ in range(groups):
        parent = len(labels)
        labels.extend(["P", "C", "C"])
        parents.extend([0, parent, parent])
    graph = make_tree(labels, parents, prefix="batch-transform")

    calls: list[int] = []
    real_apply = parametric_star_module.apply_mixed_contraction_batch

    def recording_apply(*args, **kwargs):
        calls.append(len(kwargs["planned"]))
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(parametric_star_module, "apply_mixed_contraction_batch", recording_apply)
    encoded = model.transform(graph)

    assert calls == [groups]
    family = parametric_star_label("P", "C")
    assert sum(data["label"] == family for _, data in encoded.nodes(data=True)) == groups
    assert raw_signature(model.decode(encoded)) == raw_signature(graph)


def test_parametric_star_batched_greedy_selection_handles_deep_self_overlap() -> None:
    labels = ["A"] * 31
    parents: list[int | None] = [None] + [(index - 1) // 2 for index in range(1, 31)]
    graph = make_tree(labels, parents, prefix="deep-self-overlap")
    model = ParametricStarCoarsener(2, 1, model_id="deep-self-overlap-star").fit([graph])

    encoded = model.transform(graph)
    family = parametric_star_label("A", "A")
    occurrences = [data for _, data in encoded.nodes(data=True) if data["label"] == family]

    # Parent-first greedy selection contracts the root's children, skips those
    # removed centers, then contracts the four qualifying depth-two families.
    assert len(occurrences) == 5
    assert all(data["size"] == 2 for data in occurrences)
    assert raw_signature(model.decode(encoded)) == raw_signature(graph)
