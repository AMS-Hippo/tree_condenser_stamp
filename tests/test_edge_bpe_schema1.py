from __future__ import annotations

import networkx as nx
import pytest

from tree_coarsening import (
    FITTING_SIZES_KEY,
    EdgeBPECoarsener,
    FittingSizeError,
    NamedVertexCoarsener,
    edge_bpe_token,
    size_weighted_pair_score,
    validate_encoded_tree,
)

from conftest import (
    encoded_signature,
    make_tree,
    raw_signature,
    snapshot_graph,
    assert_graph_unchanged,
)


def test_bpe_overlap_count_and_disjoint_event_count_are_distinct() -> None:
    graph = make_tree(["A", "A", "A"], [None, 0, 1], prefix="overlap")
    model = EdgeBPECoarsener(
        num_merges=1,
        min_pair_count=1,
        model_id="bpe-overlap",
    ).fit([graph])
    assert len(model.history_) == 1
    event = model.history_[0]
    assert event["parent_label"] == "A"
    assert event["child_label"] == "A"
    assert event["count"] == 2
    assert event["actual_events"] == 1
    rule = model.encoder_.rules[0]
    assert rule.output_label == edge_bpe_token("bpe-overlap", 0)
    assert rule.pattern["raw_count"] == 2
    assert rule.pattern["actual_events"] == 1


def test_bpe_nested_merge_round_trip_and_exact_offsets(chain4: nx.DiGraph) -> None:
    before = snapshot_graph(chain4)
    model = EdgeBPECoarsener(
        num_merges=2,
        min_pair_count=1,
        model_id="bpe-nested",
    ).fit([chain4])
    encoded = model.transform(chain4)
    assert_graph_unchanged(chain4, before)
    validate_encoded_tree(encoded)
    assert encoded.number_of_nodes() == 2
    assert sorted(data["size"] for _, data in encoded.nodes(data=True)) == [1, 3]
    assert raw_signature(model.decode(encoded)) == raw_signature(chain4)


def test_bpe_size_weighted_formula_uses_label_sizes() -> None:
    assert size_weighted_pair_score(3, 100, 200, 2, 7) == 27.0


def test_bpe_transform_is_independent_of_encoded_node_keys(chain4: nx.DiGraph) -> None:
    model = EdgeBPECoarsener(
        num_merges=2,
        min_pair_count=1,
        model_id="bpe-keys",
    ).fit([chain4])
    baseline = model.transform(chain4)

    normalized = (
        EdgeBPECoarsener(
            num_merges=0,
            min_pair_count=1,
            model_id="normalizer",
        )
        .fit([chain4])
        .transform(chain4)
    )
    mapping = {node: ("opaque", 100 - i) for i, node in enumerate(normalized.nodes)}
    relabeled = nx.relabel_nodes(normalized, mapping, copy=True)
    # A fresh model is needed because the no-op normalizer stage is active in this input.
    on_encoded = EdgeBPECoarsener(
        num_merges=2,
        min_pair_count=1,
        model_id="bpe-encoded-keys",
    ).fit([normalized])
    first = on_encoded.transform(normalized)
    second = on_encoded.transform(relabeled)
    assert encoded_signature(first) == encoded_signature(second)
    assert baseline.number_of_nodes() == first.number_of_nodes()


def test_numba_backend_rejects_custom_scorer() -> None:
    with pytest.raises(Exception, match="custom"):
        EdgeBPECoarsener(pair_score=lambda *args: 1.0, backend="numba")


def test_bpe_transform_rejects_changed_input_label_fitting_size() -> None:
    raw = make_tree(["A", "A", "B"], [None, 0, 1], prefix="input-size")
    upstream = NamedVertexCoarsener(labels={"A"}, fitting_size=2, model_id="size-producer").fit(
        [raw]
    )
    encoded = upstream.transform(raw)
    bpe = EdgeBPECoarsener(num_merges=1, min_pair_count=1, model_id="size-consumer").fit([encoded])

    malformed = encoded.copy()
    malformed.graph.update(encoded.graph)
    malformed.graph[FITTING_SIZES_KEY] = dict(encoded.graph[FITTING_SIZES_KEY])
    producer_label = ("named_component", "size-producer")
    malformed.graph[FITTING_SIZES_KEY][producer_label] = 3
    # The graph schema alone cannot know the producer artifact's intended size;
    # the fitted BPE program can and must reject reinterpreting the label.
    validate_encoded_tree(malformed)
    with pytest.raises(FittingSizeError, match="was fitted with label"):
        bpe.transform(malformed)
