from __future__ import annotations

import inspect
from dataclasses import MISSING, fields

from tree_coarsening import (
    EDGE_FIELDS,
    NODE_FIELDS,
    SCHEMA_VERSION,
    CompositeType,
    EncodingRule,
    StructuralStageDecoder,
    TreeCoarsener,
    TreeDecoder,
    TreeEncoder,
    combine,
)


def _signature_shape(
    callable_object: object,
) -> tuple[tuple[str, inspect._ParameterKind, object], ...]:
    return tuple(
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(callable_object).parameters.items()
    )


def test_frozen_occurrence_and_rule_field_names() -> None:
    assert SCHEMA_VERSION == "1.0"
    assert NODE_FIELDS == ("label", "type", "size", "time", "super_uids")
    assert EDGE_FIELDS == ("attach_map",)
    assert tuple(field.name for field in fields(CompositeType)) == (
        "model_id",
        "label",
        "parent",
        "components",
        "attach",
    )

    rule_fields = fields(EncodingRule)
    assert tuple(field.name for field in rule_fields) == (
        "rule_index",
        "operation",
        "output_label",
        "output_fitting_size",
        "pattern",
        "parameter_names",
        "score",
    )
    assert rule_fields[4].default is MISSING
    assert rule_fields[4].default_factory is dict
    assert rule_fields[5].default == ()
    assert rule_fields[6].default is None


def test_frozen_tree_coarsener_method_signatures() -> None:
    keyword_only = inspect.Parameter.KEYWORD_ONLY
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    empty = inspect.Parameter.empty

    assert _signature_shape(TreeCoarsener.fit) == (
        ("self", positional, empty),
        ("graphs", positional, empty),
        ("validate", keyword_only, "full"),
    )
    assert _signature_shape(TreeCoarsener.transform) == (
        ("self", positional, empty),
        ("graph", positional, empty),
        ("validate", keyword_only, "full"),
    )
    assert _signature_shape(TreeCoarsener.fit_transform) == (
        ("self", positional, empty),
        ("graphs", positional, empty),
        ("validate", keyword_only, "full"),
    )
    decode_shape = (
        ("self", positional, empty),
        ("graph", positional, empty),
        ("target", keyword_only, None),
        ("by", keyword_only, "node"),
        ("recursive", keyword_only, True),
        ("boundary_policy", keyword_only, "expand"),
        ("validate", keyword_only, "full"),
    )
    assert _signature_shape(TreeCoarsener.decode) == decode_shape
    assert _signature_shape(TreeCoarsener.inverse_transform) == decode_shape


def test_frozen_artifact_and_composition_signatures() -> None:
    keyword_only = inspect.Parameter.KEYWORD_ONLY
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    empty = inspect.Parameter.empty

    transform_shape = (
        ("self", positional, empty),
        ("graph", positional, empty),
        ("validate", keyword_only, "full"),
    )
    decode_shape = (
        ("self", positional, empty),
        ("graph", positional, empty),
        ("target", keyword_only, None),
        ("by", keyword_only, "node"),
        ("recursive", keyword_only, True),
        ("boundary_policy", keyword_only, "expand"),
        ("validate", keyword_only, "full"),
    )
    assert _signature_shape(TreeEncoder.transform) == transform_shape
    assert _signature_shape(TreeDecoder.decode) == decode_shape
    assert _signature_shape(StructuralStageDecoder.decode) == decode_shape
    assert _signature_shape(combine) == (
        ("encoders", positional, empty),
        ("decoders", positional, empty),
    )
