from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Hashable

import networkx as nx
import pytest

from tree_coarsening import (
    ConfigurationError,
    EdgeBPECoarsener,
    EncodingRule,
    FittingSizeError,
    NamedVertexCoarsener,
    NotFittedError,
    RuleBasedEncoder,
    StageOrderError,
    StructuralStageDecoder,
    TreeCoarsener,
    TreeDecoder,
    TreeEncoder,
    ValidationError,
)

from conftest import assert_graph_unchanged, make_tree, raw_signature, snapshot_graph


class _FirstEdgeEncoder(RuleBasedEncoder):
    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[Hashable]]:
        del rule
        edge = next(iter(graph.edges), None)
        return () if edge is None else (edge,)


class _FixedOutputCoarsener(TreeCoarsener):
    def __init__(
        self,
        *,
        output_label: Hashable,
        fitting_size: int,
        model_id: str,
    ) -> None:
        super().__init__(model_id=model_id)
        self.output_label = output_label
        self.fitting_size = fitting_size

    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        del graphs
        rule = EncodingRule(
            0,
            "edge",
            self.output_label,
            self.fitting_size,
            {"kind": "fixed-output-test"},
        )
        return (
            _FirstEdgeEncoder(model_id=self.model_id, rules=(rule,)),
            StructuralStageDecoder(model_id=self.model_id, rules=(rule,)),
        )


class _BadPairCoarsener(TreeCoarsener):
    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        del graphs
        encoder_rule = EncodingRule(0, "edge", "encoder-output", 1, {})
        decoder_rule = EncodingRule(0, "edge", "decoder-output", 1, {})
        return (
            _FirstEdgeEncoder(model_id=self.model_id, rules=(encoder_rule,)),
            StructuralStageDecoder(model_id=self.model_id, rules=(decoder_rule,)),
        )


@pytest.mark.parametrize("operation", ["transform", "decode"])
def test_public_fitted_operations_require_fit(operation: str, chain4: nx.DiGraph) -> None:
    model = NamedVertexCoarsener(labels={"B"}, model_id="not-fitted")
    with pytest.raises(NotFittedError):
        getattr(model, operation)(chain4)


@pytest.mark.parametrize(
    "graphs",
    [
        pytest.param([], id="empty"),
        pytest.param("not-a-sequence-of-graphs", id="string"),
        pytest.param(nx.DiGraph(), id="single-graph"),
    ],
)
def test_fit_rejects_invalid_corpus_container(graphs: object) -> None:
    model = NamedVertexCoarsener(labels={"A"}, model_id="bad-corpus")
    with pytest.raises((TypeError, ValueError)):
        model.fit(graphs)  # type: ignore[arg-type]


def test_fit_transform_returns_one_output_per_input_without_mutation() -> None:
    graphs = [
        make_tree(["A", "A", "B"], [None, 0, 1], prefix="ft0"),
        make_tree(["A", "A", "C"], [None, 0, 1], prefix="ft1"),
    ]
    snapshots = [snapshot_graph(graph) for graph in graphs]
    model = NamedVertexCoarsener(labels={"A"}, model_id="fit-transform")
    outputs = model.fit_transform(graphs)
    assert len(outputs) == len(graphs)
    for graph, snapshot in zip(graphs, snapshots, strict=True):
        assert_graph_unchanged(graph, snapshot)
    for original, encoded in zip(graphs, outputs, strict=True):
        assert raw_signature(model.decode(encoded)) == raw_signature(original)


def test_failed_refit_preserves_prior_artifacts_and_diagnostics() -> None:
    graph = make_tree(["A", "B", "A", "B"], [None, 0, 1, 2], prefix="atomic")
    should_fail = False

    def scorer(n_ab: int, n_a: int, n_b: int, s_a: int, s_b: int) -> float:
        del n_a, n_b, s_a, s_b
        if should_fail:
            raise RuntimeError("deliberate scorer failure")
        return float(n_ab)

    model = EdgeBPECoarsener(
        num_merges=2,
        min_pair_count=1,
        pair_score=scorer,
        model_id="atomic-bpe",
    ).fit([graph])
    old_encoder = model.encoder_
    old_decoder = model.decoder_
    old_history = deepcopy(model.history_)
    old_backend = model.backend_used_

    should_fail = True
    with pytest.raises(ValidationError, match="pair_score failed"):
        model.fit([graph])

    assert model.encoder_ is old_encoder
    assert model.decoder_ is old_decoder
    assert model.history_ == old_history
    assert model.backend_used_ == old_backend
    encoded = model.transform(graph)
    assert raw_signature(model.decode(encoded)) == raw_signature(graph)


def test_failed_initial_fit_leaves_artifacts_unset() -> None:
    graph = make_tree(["A", "B"], [None, 0], prefix="bad-pair")
    model = _BadPairCoarsener(model_id="bad-pair")
    with pytest.raises(ConfigurationError, match="rules must be identical"):
        model.fit([graph])
    assert model.encoder_ is None
    assert model.decoder_ is None


def test_fit_corpus_rejects_conflicting_fitting_sizes_for_every_coarsener() -> None:
    first_raw = make_tree(["A", "B"], [None, 0], prefix="size-two")
    second_raw = make_tree(["A", "B"], [None, 0], prefix="size-three")
    shared_label = ("shared-fit-label",)
    first = (
        _FixedOutputCoarsener(
            output_label=shared_label,
            fitting_size=2,
            model_id="producer-two",
        )
        .fit([first_raw])
        .transform(first_raw)
    )
    second = (
        _FixedOutputCoarsener(
            output_label=shared_label,
            fitting_size=3,
            model_id="producer-three",
        )
        .fit([second_raw])
        .transform(second_raw)
    )

    consumer = NamedVertexCoarsener(labels={"A"}, model_id="consumer")
    with pytest.raises(FittingSizeError, match="conflicting sizes"):
        consumer.fit([first, second])
    assert consumer.encoder_ is None
    assert consumer.decoder_ is None


def test_transform_rejects_reusing_an_active_model_id(chain4: nx.DiGraph) -> None:
    model = NamedVertexCoarsener(labels={"B"}, model_id="collision").fit([chain4])
    encoded = model.transform(chain4)
    before = snapshot_graph(encoded)
    with pytest.raises(StageOrderError, match="already active"):
        model.transform(encoded)
    assert_graph_unchanged(encoded, before)


def test_raw_node_and_graph_attributes_round_trip_but_edge_attributes_do_not() -> None:
    graph = make_tree(["A", "A", "B"], [None, 0, 1], prefix="attrs")
    model = NamedVertexCoarsener(labels={"A"}, model_id="attrs-stage").fit([graph])
    decoded = model.decode(model.transform(graph))
    assert raw_signature(decoded) == raw_signature(graph)
    assert all(not data for *_edge, data in decoded.edges(data=True))


def test_invalid_validation_level_is_rejected_without_mutation(chain4: nx.DiGraph) -> None:
    model = NamedVertexCoarsener(labels={"B"}, model_id="validation-level").fit([chain4])
    before = snapshot_graph(chain4)
    with pytest.raises(ValueError, match="validate"):
        model.transform(chain4, validate="fast")  # type: ignore[arg-type]
    assert_graph_unchanged(chain4, before)


def test_fit_transform_failure_restores_previous_fitted_state() -> None:
    raw = make_tree(["A", "A", "B"], [None, 0, 1], prefix="ft-atomic")
    model = NamedVertexCoarsener(labels={"A"}, model_id="ft-atomic-stage").fit([raw])
    old_encoder = model.encoder_
    old_decoder = model.decoder_
    already_active = model.transform(raw)

    # Fitting on an active same-ID graph is permitted, but transforming that
    # graph must reject the stage collision. fit_transform is one public
    # operation, so the successful refit preceding that failure is rolled back.
    with pytest.raises(StageOrderError, match="already active"):
        model.fit_transform([raw, already_active])

    assert model.encoder_ is old_encoder
    assert model.decoder_ is old_decoder
    round_trip = model.decode(model.transform(raw))
    assert raw_signature(round_trip) == raw_signature(raw)
