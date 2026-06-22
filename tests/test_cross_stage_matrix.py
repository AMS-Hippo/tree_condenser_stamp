from __future__ import annotations

from itertools import permutations

import pytest

from tree_coarsening import (
    CompositionError,
    DecodeSelectionError,
    EdgeBPECoarsener,
    NamedVertexCoarsener,
    ParametricStarCoarsener,
    SCHEMA_KEY,
    StageOrderError,
    combine,
)

from conftest import (
    assert_graph_unchanged,
    encoded_occurrence_signature,
    encoded_signature,
    make_tree,
    raw_signature,
    snapshot_graph,
)


def _pipeline_tree():
    return make_tree(
        ["R", "P", "C", "C", "C", "B", "B", "D", "P", "C", "C"],
        [None, 0, 1, 1, 1, 0, 5, 6, 0, 8, 8],
        prefix="pipeline",
    )


def _new_stage(name: str, index: int):
    model_id = f"{name}-{index}"
    if name == "star":
        return ParametricStarCoarsener(2, 1, model_id=model_id)
    if name == "bpe":
        return EdgeBPECoarsener(
            num_merges=3,
            min_pair_count=1,
            model_id=model_id,
        )
    if name == "named":
        return NamedVertexCoarsener(labels={"B", "C"}, model_id=model_id)
    raise AssertionError(name)


@pytest.mark.parametrize("order", tuple(permutations(("star", "bpe", "named"))))
def test_all_three_stage_orders_round_trip(order: tuple[str, ...]) -> None:
    raw = _pipeline_tree()
    current = raw
    models = []
    for index, name in enumerate(order):
        model = _new_stage(name, index).fit([current])
        current = model.transform(current)
        models.append(model)

    assert tuple(record["model_id"] for record in current.graph[SCHEMA_KEY]["stages"]) == tuple(
        model.model_id for model in models
    )
    for model in reversed(models):
        current = model.decode(current)
    assert raw_signature(current) == raw_signature(raw)


def test_reverse_stage_order_is_enforced_without_mutation() -> None:
    raw = _pipeline_tree()
    star = ParametricStarCoarsener(2, 1, model_id="first").fit([raw])
    first = star.transform(raw)
    bpe = EdgeBPECoarsener(
        num_merges=2,
        min_pair_count=1,
        model_id="second",
    ).fit([first])
    second = bpe.transform(first)
    before = snapshot_graph(second)

    with pytest.raises(StageOrderError, match="latest stage"):
        star.decode(second)
    assert_graph_unchanged(second, before)


def test_stage_decode_restores_preceding_metadata_exactly() -> None:
    raw = _pipeline_tree()
    star = ParametricStarCoarsener(2, 1, model_id="metadata-star").fit([raw])
    first = star.transform(raw)
    bpe = EdgeBPECoarsener(
        num_merges=3,
        min_pair_count=1,
        model_id="metadata-bpe",
    ).fit([first])
    second = bpe.transform(first)

    restored = bpe.decode(second)
    assert encoded_occurrence_signature(restored) == encoded_occurrence_signature(first)
    assert raw_signature(star.decode(restored)) == raw_signature(raw)


def test_noop_stage_still_has_explicit_lineage_and_round_trips() -> None:
    raw = make_tree(["A", "B"], [None, 0], prefix="noop")
    model = EdgeBPECoarsener(
        num_merges=0,
        min_pair_count=1,
        model_id="noop-stage",
    ).fit([raw])
    encoded = model.transform(raw)
    assert encoded.graph[SCHEMA_KEY]["stages"] == (
        {"model_id": "noop-stage", "introduced_labels": ()},
    )
    assert raw_signature(model.decode(encoded)) == raw_signature(raw)


def test_reusing_active_model_id_is_rejected() -> None:
    raw = make_tree(["A", "B"], [None, 0], prefix="collision")
    model = EdgeBPECoarsener(
        num_merges=0,
        min_pair_count=1,
        model_id="collision-stage",
    ).fit([raw])
    encoded = model.transform(raw)

    with pytest.raises(StageOrderError, match="already active"):
        model.transform(encoded)


def test_lazy_composition_matches_manual_pipeline_and_decodes() -> None:
    raw = _pipeline_tree()
    star = ParametricStarCoarsener(2, 1, model_id="combined-star").fit([raw])
    first = star.transform(raw)
    bpe = EdgeBPECoarsener(
        num_merges=2,
        min_pair_count=1,
        model_id="combined-bpe",
    ).fit([first])

    manual = bpe.transform(first)
    encoder, decoder = combine(
        (star.encoder_, bpe.encoder_),
        (star.decoder_, bpe.decoder_),
    )
    combined = encoder.transform(raw)
    assert encoded_signature(combined) == encoded_signature(manual)
    assert raw_signature(decoder.decode(combined)) == raw_signature(raw)

    with pytest.raises(DecodeSelectionError, match="targeted"):
        decoder.decode(combined, target=0)


def test_combine_rejects_duplicate_atomic_stage_ids_even_when_nested() -> None:
    raw = make_tree(["A"], [None], prefix="duplicates")
    model = EdgeBPECoarsener(
        num_merges=0,
        min_pair_count=1,
        model_id="duplicate",
    ).fit([raw])
    first_encoder, first_decoder = combine((model.encoder_,), (model.decoder_,))

    with pytest.raises(CompositionError, match="duplicate stage model ID"):
        combine(
            (first_encoder, model.encoder_),
            (first_decoder, model.decoder_),
        )
