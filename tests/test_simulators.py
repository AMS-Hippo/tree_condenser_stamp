from __future__ import annotations

from collections import Counter

import networkx as nx

from tree_coarsening import (
    EdgeBPECoarsener,
    NamedVertexCoarsener,
    ParametricStarCoarsener,
    validate_raw_tree,
)
from tree_coarsening.utils import (
    make_edge_bpe_dataset,
    make_named_component_tree,
    make_repeated_edge_tree,
    make_starburst_dataset,
)

from conftest import raw_signature


def _uid_signature(graph: nx.DiGraph) -> tuple[object, object]:
    nodes = sorted(
        (repr(data["uid"]), data["label"], data["time"]) for _, data in graph.nodes(data=True)
    )
    edges = sorted(
        (repr(graph.nodes[parent]["uid"]), repr(graph.nodes[child]["uid"]))
        for parent, child in graph.edges
    )
    return nodes, edges


def test_repeated_edge_tree_has_expected_motif_counts() -> None:
    graph = make_repeated_edge_tree(
        n_repeats=7,
        motif_labels=("A", "B", "C", "D"),
        seed=2,
    )
    validate_raw_tree(graph)

    counts = Counter(
        (graph.nodes[parent]["label"], graph.nodes[child]["label"]) for parent, child in graph.edges
    )
    assert counts[("A", "B")] == 7
    assert counts[("B", "C")] == 7
    assert counts[("C", "D")] == 7


def test_edge_bpe_dataset_supports_hierarchical_roundtrip() -> None:
    graphs = make_edge_bpe_dataset(n_graphs=2, n_repeats=5, seed=3)
    coarsener = EdgeBPECoarsener(
        num_merges=3,
        min_pair_count=4,
        model_id="sim-bpe",
    ).fit(graphs)

    assert len(coarsener.history_) == 3
    encoded = coarsener.transform(graphs[0])
    decoded = coarsener.decode(encoded)
    assert encoded.number_of_nodes() < graphs[0].number_of_nodes()
    assert raw_signature(decoded) == raw_signature(graphs[0])


def test_named_component_tree_has_separated_component_sizes() -> None:
    graph = make_named_component_tree(
        component_sizes=(5, 3),
        selected_labels=("A", "B"),
        include_singleton=True,
        seed=4,
    )
    validate_raw_tree(graph)

    selected = [node for node, data in graph.nodes(data=True) if data["label"] in {"A", "B"}]
    sizes = sorted(
        (len(component) for component in nx.weakly_connected_components(graph.subgraph(selected))),
        reverse=True,
    )
    assert sizes == [5, 3, 1]


def test_named_component_simulator_supports_label_and_uid_examples() -> None:
    graph = make_named_component_tree(seed=5)

    by_label = NamedVertexCoarsener(
        labels={"A", "B"},
        component_policy="all",
        model_id="sim-named-label",
    ).fit([graph])
    label_encoded = by_label.transform(graph)
    assert raw_signature(by_label.decode(label_encoded)) == raw_signature(graph)

    selected_uids = {
        data["uid"]
        for _, data in graph.nodes(data=True)
        if data["uid"].startswith("named_component_0_")
    }
    by_uid = NamedVertexCoarsener(
        uids=selected_uids,
        model_id="sim-named-uid",
    ).fit([graph])
    uid_encoded = by_uid.transform(graph)
    assert raw_signature(by_uid.decode(uid_encoded)) == raw_signature(graph)


def test_seeded_simulators_are_reproducible_valid_and_round_trip() -> None:
    star_first = make_starburst_dataset(
        n_graphs=2,
        max_nodes=24,
        n_bursts=2,
        burst_size_range=(3, 4),
        seed=101,
    )
    star_second = make_starburst_dataset(
        n_graphs=2,
        max_nodes=24,
        n_bursts=2,
        burst_size_range=(3, 4),
        seed=101,
    )
    assert [_uid_signature(graph) for graph in star_first] == [
        _uid_signature(graph) for graph in star_second
    ]

    bpe_first = make_edge_bpe_dataset(n_graphs=2, n_repeats=4, seed=202)
    bpe_second = make_edge_bpe_dataset(n_graphs=2, n_repeats=4, seed=202)
    assert [_uid_signature(graph) for graph in bpe_first] == [
        _uid_signature(graph) for graph in bpe_second
    ]

    named_first = make_named_component_tree(seed=303)
    named_second = make_named_component_tree(seed=303)
    assert _uid_signature(named_first) == _uid_signature(named_second)

    for graph in (*star_first, *bpe_first, named_first):
        validate_raw_tree(graph)

    star = ParametricStarCoarsener(
        2,
        1,
        model_id="sim-roundtrip-star",
    ).fit(star_first)
    bpe = EdgeBPECoarsener(
        num_merges=2,
        min_pair_count=2,
        model_id="sim-roundtrip-bpe",
    ).fit(bpe_first)
    named = NamedVertexCoarsener(
        labels={"A", "B"},
        model_id="sim-roundtrip-named",
    ).fit([named_first])

    for model, graph in (
        (star, star_first[0]),
        (bpe, bpe_first[0]),
        (named, named_first),
    ):
        encoded = model.transform(graph)
        assert raw_signature(model.decode(encoded)) == raw_signature(graph)
