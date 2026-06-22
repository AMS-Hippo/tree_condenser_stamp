from __future__ import annotations

import pytest

from tree_coarsening import (
    CompositionError,
    ConfigurationError,
    EdgeBPEEncoder,
    EdgeBPERule,
    EncodingRule,
    FittingSizeError,
    NamedVertexCoarsener,
    NamedVertexEncoder,
    ParametricStarEncoder,
    StructuralStageDecoder,
    TypeOwnershipError,
    combine,
    edge_bpe_token,
    named_component_label,
    parametric_star_label,
)

from conftest import make_tree


def test_same_id_decoder_with_different_vocabulary_is_rejected() -> None:
    raw = make_tree(["A", "A"], [None, 0], prefix="rogue")
    model = NamedVertexCoarsener(labels={"A"}, model_id="same-id").fit([raw])
    encoded = model.transform(raw)
    rogue_rule = EncodingRule(
        0,
        "component",
        ("different", "same-id"),
        1,
        {"selector": "label", "values": ("A",), "component_policy": "all"},
        ("topology",),
    )
    rogue = StructuralStageDecoder(model_id="same-id", rules=(rogue_rule,))

    with pytest.raises(TypeOwnershipError, match="stage-introduced"):
        rogue.decode(encoded)


def test_same_id_decoder_with_wrong_fitting_size_is_rejected() -> None:
    raw = make_tree(["A", "A"], [None, 0], prefix="size")
    model = NamedVertexCoarsener(labels={"A"}, model_id="size-id").fit([raw])
    encoded = model.transform(raw)
    label = named_component_label("size-id")
    rogue_rule = EncodingRule(
        0,
        "component",
        label,
        7,
        {"selector": "label", "values": ("A",), "component_policy": "all"},
        ("topology",),
    )
    rogue = StructuralStageDecoder(model_id="size-id", rules=(rogue_rule,))

    with pytest.raises(FittingSizeError, match="records 1"):
        rogue.decode(encoded)


def test_combine_rejects_same_vocab_but_different_rules() -> None:
    raw = make_tree(["A", "A"], [None, 0], prefix="combine")
    model = NamedVertexCoarsener(labels={"A"}, model_id="combine-id").fit([raw])
    encoder = model.encoder_
    assert encoder is not None
    original = encoder.rules[0]
    different = EncodingRule(
        original.rule_index,
        original.operation,
        original.output_label,
        original.output_fitting_size,
        {"selector": "label", "values": ("B",), "component_policy": "all"},
        original.parameter_names,
    )
    decoder = StructuralStageDecoder(model_id="combine-id", rules=(different,))

    with pytest.raises(CompositionError, match="rules"):
        combine((encoder,), (decoder,))


def test_parametric_encoder_rejects_rule_state_mismatch() -> None:
    rule = EncodingRule(
        0,
        "siblings",
        parametric_star_label("P", "C"),
        2,
        {
            "parent_label": "P",
            "child_label": "C",
            "witness_min_children": 3,
            "contract_min_children": 3,
        },
        ("arity",),
    )
    with pytest.raises(ConfigurationError, match="threshold"):
        ParametricStarEncoder(model_id="star", rules=(rule,), contract_d=2)


def test_named_encoder_rejects_rule_state_mismatch() -> None:
    rule = EncodingRule(
        0,
        "component",
        named_component_label("named"),
        1,
        {"selector": "label", "values": ("A",), "component_policy": "all"},
        ("topology",),
    )
    with pytest.raises(ConfigurationError, match="selection state"):
        NamedVertexEncoder(
            model_id="named",
            rules=(rule,),
            selector="label",
            selected_values=frozenset({"B"}),
            component_policy="all",
        )


def test_bpe_encoder_rejects_rule_state_mismatch() -> None:
    token = edge_bpe_token("bpe", 0)
    edge_rule = EdgeBPERule(
        rank=0,
        token=token,
        parent_label="A",
        child_label="B",
        count=2,
        score=2.0,
        parent_count=2,
        child_count=2,
        parent_size=1,
        child_size=1,
    )
    generic = EncodingRule(
        0,
        "edge",
        token,
        2,
        {"parent_label": "X", "child_label": "B", "raw_count": 2},
        score=2.0,
    )
    with pytest.raises(ConfigurationError, match="pattern disagrees"):
        EdgeBPEEncoder(
            model_id="bpe",
            rules=(generic,),
            edge_rules=(edge_rule,),
            input_labels=("A", "B"),
        )


def test_fitted_artifact_semantic_state_is_write_once() -> None:
    raw = make_tree(["A", "A"], [None, 0], prefix="immutable")
    model = NamedVertexCoarsener(labels={"A"}, model_id="immutable-stage").fit([raw])
    encoder = model.encoder_
    decoder = model.decoder_
    assert encoder is not None and decoder is not None

    with pytest.raises(AttributeError, match="semantically immutable"):
        encoder.selector = "uid"
    with pytest.raises(AttributeError, match="semantically immutable"):
        encoder._model_id = "changed"
    with pytest.raises(AttributeError, match="semantically immutable"):
        decoder._rules = ()
    with pytest.raises(AttributeError, match="semantically immutable"):
        del encoder.selected_values
    with pytest.raises(AttributeError, match="read-only"):
        encoder.vocab._sizes = {}
    with pytest.raises(TypeError, match="does not support item assignment"):
        encoder.vocab._sizes["mutated"] = 99

    copied_sizes = encoder.vocab.as_dict()
    copied_sizes["mutated"] = 99
    assert "mutated" not in encoder.vocab


@pytest.mark.parametrize("score", [True, "2.0", float("inf"), float("nan")])
def test_edge_bpe_rule_rejects_invalid_score_metadata(score: object) -> None:
    with pytest.raises(ConfigurationError, match="score"):
        EdgeBPERule(
            rank=0,
            token=edge_bpe_token("bad-score", 0),
            parent_label="A",
            child_label="B",
            count=1,
            score=score,  # type: ignore[arg-type]
        )


def test_bpe_encoder_requires_complete_endpoint_statistics() -> None:
    token = edge_bpe_token("missing-statistics", 0)
    generic = EncodingRule(
        0,
        "edge",
        token,
        2,
        {
            "parent_label": "A",
            "child_label": "B",
            "count_semantics": "raw_matching_edges",
            "raw_count": 1,
            "actual_events": 1,
            "pair_score": "count",
            "parent_count": None,
            "child_count": 1,
        },
        score=1.0,
    )
    optimized = EdgeBPERule(
        rank=0,
        token=token,
        parent_label="A",
        child_label="B",
        count=1,
        score=1.0,
        parent_count=None,
        child_count=1,
        parent_size=1,
        child_size=1,
    )
    with pytest.raises(ConfigurationError, match="missing endpoint"):
        EdgeBPEEncoder(
            model_id="missing-statistics",
            rules=(generic,),
            edge_rules=(optimized,),
            input_labels=("A", "B"),
        )


@pytest.mark.parametrize("actual_events", [0, 3, True])
def test_bpe_encoder_rejects_invalid_actual_event_metadata(actual_events: object) -> None:
    token = edge_bpe_token("bad-events", 0)
    generic = EncodingRule(
        0,
        "edge",
        token,
        2,
        {
            "parent_label": "A",
            "child_label": "B",
            "count_semantics": "raw_matching_edges",
            "raw_count": 2,
            "actual_events": actual_events,
            "pair_score": "count",
            "parent_count": 2,
            "child_count": 2,
        },
        score=2.0,
    )
    optimized = EdgeBPERule(
        rank=0,
        token=token,
        parent_label="A",
        child_label="B",
        count=2,
        score=2.0,
        parent_count=2,
        child_count=2,
        parent_size=1,
        child_size=1,
    )
    with pytest.raises(ConfigurationError, match="event count"):
        EdgeBPEEncoder(
            model_id="bad-events",
            rules=(generic,),
            edge_rules=(optimized,),
            input_labels=("A", "B"),
        )
