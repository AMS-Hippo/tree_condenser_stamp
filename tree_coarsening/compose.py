"""Lazy composition of fitted schema-1 encoder/decoder stages."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from uuid import uuid4

import networkx as nx

from .decoder import TreeDecoder
from .encoder import EncodingRule, TreeEncoder
from .exceptions import CompositionError, DecodeSelectionError


def _inspection_rules(encoders: Sequence[TreeEncoder]) -> tuple[EncodingRule, ...]:
    flattened: list[EncodingRule] = []
    for encoder in encoders:
        for rule in encoder.rules:
            flattened.append(
                EncodingRule(
                    rule_index=len(flattened),
                    operation=rule.operation,
                    output_label=rule.output_label,
                    output_fitting_size=rule.output_fitting_size,
                    pattern=dict(rule.pattern),
                    parameter_names=rule.parameter_names,
                    score=rule.score,
                )
            )
    return tuple(flattened)


class CombinedTreeEncoder(TreeEncoder):
    """Apply an immutable tuple of atomic stage encoders in order."""

    def __init__(self, *, model_id: str, encoders: Sequence[TreeEncoder]) -> None:
        self.encoders = tuple(encoders)
        self.stage_model_ids = tuple(encoder.model_id for encoder in self.encoders)
        super().__init__(model_id=model_id, rules=_inspection_rules(self.encoders))

    def transform(
        self,
        graph: nx.DiGraph,
        *,
        validate: str | bool = "full",
    ) -> nx.DiGraph:
        current = graph
        for encoder in self.encoders:
            current = encoder.transform(current, validate=validate)
        return current


class CombinedTreeDecoder(TreeDecoder):
    """Reverse an immutable tuple of atomic stage decoders in reverse order."""

    def __init__(
        self,
        *,
        model_id: str,
        decoders: Sequence[TreeDecoder],
        inspection_rules: Sequence[EncodingRule],
    ) -> None:
        self.decoders = tuple(decoders)
        self.stage_model_ids = tuple(decoder.model_id for decoder in self.decoders)
        super().__init__(model_id=model_id, rules=inspection_rules)

    def decode(
        self,
        graph: nx.DiGraph,
        *,
        target: Hashable | None = None,
        by: str = "node",
        recursive: bool = True,
        boundary_policy: str = "expand",
        validate: str | bool = "full",
    ) -> nx.DiGraph:
        if target is not None:
            raise DecodeSelectionError(
                "targeted decoding through a combined pipeline is deferred; "
                "use the current component decoder directly."
            )
        current = graph
        for decoder in reversed(self.decoders):
            current = decoder.decode(
                current,
                target=None,
                by=by,
                recursive=recursive,
                boundary_policy=boundary_policy,
                validate=validate,
            )
        return current


def _flatten_encoder(encoder: TreeEncoder) -> tuple[TreeEncoder, ...]:
    if isinstance(encoder, CombinedTreeEncoder):
        return encoder.encoders
    return (encoder,)


def _flatten_decoder(decoder: TreeDecoder) -> tuple[TreeDecoder, ...]:
    if isinstance(decoder, CombinedTreeDecoder):
        return decoder.decoders
    return (decoder,)


def _validate_pair(encoder: TreeEncoder, decoder: TreeDecoder, *, index: int) -> None:
    if encoder.model_id != decoder.model_id:
        raise CompositionError(
            f"encoder/decoder stage {index} has model IDs "
            f"{encoder.model_id!r} and {decoder.model_id!r}."
        )
    if encoder.rules != decoder.rules:
        raise CompositionError(f"encoder/decoder stage {index} disagrees on fitted rules.")
    if encoder.vocab != decoder.vocab:
        raise CompositionError(f"encoder/decoder stage {index} disagrees on stage vocabulary.")


def combine(
    encoders: Sequence[TreeEncoder],
    decoders: Sequence[TreeDecoder],
) -> tuple[TreeEncoder, TreeDecoder]:
    if isinstance(encoders, (str, bytes, bytearray)) or isinstance(
        decoders, (str, bytes, bytearray)
    ):
        raise CompositionError("encoders and decoders must be nonempty sequences.")
    try:
        encoder_inputs = tuple(encoders)
        decoder_inputs = tuple(decoders)
    except TypeError as exc:
        raise CompositionError("encoders and decoders must be nonempty sequences.") from exc
    if not encoder_inputs or not decoder_inputs:
        raise CompositionError("encoders and decoders must be nonempty sequences.")
    if len(encoder_inputs) != len(decoder_inputs):
        raise CompositionError("encoders and decoders must have equal lengths.")
    if any(not isinstance(encoder, TreeEncoder) for encoder in encoder_inputs):
        raise CompositionError("every encoder must be a fitted TreeEncoder artifact.")
    if any(not isinstance(decoder, TreeDecoder) for decoder in decoder_inputs):
        raise CompositionError("every decoder must be a fitted TreeDecoder artifact.")

    atomic_encoders = tuple(
        component for encoder in encoder_inputs for component in _flatten_encoder(encoder)
    )
    atomic_decoders = tuple(
        component for decoder in decoder_inputs for component in _flatten_decoder(decoder)
    )
    if len(atomic_encoders) != len(atomic_decoders):
        raise CompositionError(
            "nested combined encoder and decoder pipelines have different stage counts."
        )

    seen_ids: set[str] = set()
    fitting_sizes: dict[object, int] = {}
    for index, (encoder, decoder) in enumerate(zip(atomic_encoders, atomic_decoders, strict=True)):
        _validate_pair(encoder, decoder, index=index)
        if encoder.model_id in seen_ids:
            raise CompositionError(
                f"duplicate stage model ID {encoder.model_id!r} in combined pipeline."
            )
        seen_ids.add(encoder.model_id)
        for label in encoder.vocab.labels:
            size = encoder.vocab.fitting_size(label)
            previous = fitting_sizes.get(label)
            if previous is not None and previous != size:
                raise CompositionError(
                    f"combined stages assign label {label!r} fitting sizes {previous} and {size}."
                )
            fitting_sizes[label] = size

    combined_id = f"combined:{uuid4().hex}"
    encoder = CombinedTreeEncoder(model_id=combined_id, encoders=atomic_encoders)
    decoder = CombinedTreeDecoder(
        model_id=combined_id,
        decoders=atomic_decoders,
        inspection_rules=encoder.rules,
    )
    if encoder.vocab != decoder.vocab:
        raise CompositionError("combined encoder and decoder vocabularies disagree.")
    return encoder, decoder


__all__ = ["CombinedTreeDecoder", "CombinedTreeEncoder", "combine"]
