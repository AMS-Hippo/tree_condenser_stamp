from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Hashable

import networkx as nx
import pytest

from tree_coarsening import (
    CompositionError,
    DecodeSelectionError,
    EdgeBPECoarsener,
    EncodingRule,
    NamedVertexCoarsener,
    ParametricStarCoarsener,
    RuleBasedEncoder,
    StructuralStageDecoder,
    TreeDecoder,
    TreeEncoder,
    combine,
)

from conftest import (
    assert_graph_unchanged,
    encoded_signature,
    make_tree,
    raw_signature,
    snapshot_graph,
)


class _NoOpEncoder(RuleBasedEncoder):
    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[Hashable]]:
        del graph, rule
        return ()


def _artifact_pair(
    *, model_id: str, output_label: Hashable, fitting_size: int
) -> tuple[TreeEncoder, TreeDecoder]:
    rule = EncodingRule(0, "noop", output_label, fitting_size)
    return (
        _NoOpEncoder(model_id=model_id, rules=(rule,)),
        StructuralStageDecoder(model_id=model_id, rules=(rule,)),
    )


def _fit_pipeline() -> tuple[
    nx.DiGraph,
    tuple[TreeEncoder, ...],
    tuple[TreeDecoder, ...],
    nx.DiGraph,
]:
    raw = make_tree(
        ["R", "P", "C", "C", "C", "P", "C", "C", "C", "Z"],
        [None, 0, 1, 1, 1, 0, 5, 5, 5, 0],
        prefix="compose",
    )
    star = ParametricStarCoarsener(3, 2, model_id="compose-star").fit([raw])
    first = star.transform(raw)
    bpe = EdgeBPECoarsener(
        num_merges=2,
        min_pair_count=1,
        model_id="compose-bpe",
    ).fit([first])
    second = bpe.transform(first)
    named = NamedVertexCoarsener(
        labels={"Z"},
        model_id="compose-named",
    ).fit([second])
    third = named.transform(second)
    return (
        raw,
        (star.encoder_, bpe.encoder_, named.encoder_),
        (star.decoder_, bpe.decoder_, named.decoder_),
        third,
    )


def test_combined_pipeline_matches_sequential_transform_and_decode() -> None:
    raw, encoders, decoders, sequential = _fit_pipeline()
    combined_encoder, combined_decoder = combine(encoders, decoders)
    before = snapshot_graph(raw)

    encoded = combined_encoder.transform(raw)
    assert_graph_unchanged(raw, before)
    assert encoded_signature(encoded) == encoded_signature(sequential)
    assert raw_signature(combined_decoder.decode(encoded)) == raw_signature(raw)


def test_nested_combination_flattens_without_changing_stage_order() -> None:
    raw, encoders, decoders, sequential = _fit_pipeline()
    left_encoder, left_decoder = combine(encoders[:2], decoders[:2])
    nested_encoder, nested_decoder = combine(
        (left_encoder, encoders[2]),
        (left_decoder, decoders[2]),
    )
    encoded = nested_encoder.transform(raw)
    assert encoded_signature(encoded) == encoded_signature(sequential)
    assert nested_encoder.stage_model_ids == (
        "compose-star",
        "compose-bpe",
        "compose-named",
    )
    assert raw_signature(nested_decoder.decode(encoded)) == raw_signature(raw)


def test_combined_targeted_decode_is_explicitly_deferred() -> None:
    raw, encoders, decoders, _sequential = _fit_pipeline()
    combined_encoder, combined_decoder = combine(encoders, decoders)
    encoded = combined_encoder.transform(raw)
    with pytest.raises(DecodeSelectionError, match="deferred"):
        combined_decoder.decode(encoded, target=0)


@pytest.mark.parametrize(
    ("encoders", "decoders"),
    [
        ((), ()),
        ((_artifact_pair(model_id="a", output_label="A", fitting_size=1)[0],), ()),
        ("bad", "bad"),
    ],
)
def test_combine_rejects_invalid_container_shapes(
    encoders: object,
    decoders: object,
) -> None:
    with pytest.raises(CompositionError):
        combine(encoders, decoders)  # type: ignore[arg-type]


def test_combine_rejects_mismatched_pairs_and_duplicate_stage_ids() -> None:
    e1, d1 = _artifact_pair(model_id="one", output_label="A", fitting_size=1)
    e2, d2 = _artifact_pair(model_id="two", output_label="B", fitting_size=1)
    with pytest.raises(CompositionError, match="model IDs"):
        combine((e1,), (d2,))

    duplicate_encoder, duplicate_decoder = _artifact_pair(
        model_id="one", output_label="C", fitting_size=1
    )
    with pytest.raises(CompositionError, match="duplicate stage model ID"):
        combine((e1, duplicate_encoder), (d1, duplicate_decoder))


def test_combine_rejects_conflicting_label_fitting_sizes() -> None:
    e1, d1 = _artifact_pair(model_id="one", output_label=("shared",), fitting_size=2)
    e2, d2 = _artifact_pair(model_id="two", output_label=("shared",), fitting_size=3)
    with pytest.raises(CompositionError, match="fitting sizes"):
        combine((e1, e2), (d1, d2))
