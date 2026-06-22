from __future__ import annotations

from copy import deepcopy

import pytest

from tree_coarsening import (
    AttachmentError,
    CompositeType,
    EncodingRule,
    ExactTypeError,
    FittingSizeError,
    GraphSchemaError,
    NamedVertexCoarsener,
    ProvenanceError,
    SCHEMA_KEY,
    StageOrderError,
    StructuralStageDecoder,
    TypeOwnershipError,
    base_type,
    validate_encoded_tree,
)

from conftest import assert_graph_unchanged, make_tree, snapshot_graph


def _no_op_encoded(*, prefix: str = "noop"):
    graph = make_tree(["A", "B"], [None, 0], prefix=prefix)
    model = NamedVertexCoarsener(labels={"absent"}, model_id=f"{prefix}-stage").fit([graph])
    return graph, model, model.transform(graph)


def test_encoded_graph_rejects_unsupported_schema_version() -> None:
    _raw, _model, encoded = _no_op_encoded(prefix="version")
    encoded.graph[SCHEMA_KEY]["version"] = "9.9"
    with pytest.raises(GraphSchemaError, match="unsupported"):
        validate_encoded_tree(encoded)


def test_prepare_rejects_incomplete_encoded_metadata() -> None:
    raw, model, encoded = _no_op_encoded(prefix="metadata")
    del encoded.graph["tree_coarsening_provenance"]
    before = snapshot_graph(encoded)
    with pytest.raises(GraphSchemaError, match="incomplete"):
        model.transform(encoded)
    assert_graph_unchanged(encoded, before)
    # The raw source remains independently usable.
    model.transform(raw)


def test_encoded_graph_rejects_unknown_reserved_graph_attribute() -> None:
    _raw, _model, encoded = _no_op_encoded(prefix="reserved")
    encoded.graph["tree_coarsening_future"] = 1
    with pytest.raises(GraphSchemaError, match="unknown reserved"):
        validate_encoded_tree(encoded)


@pytest.mark.parametrize("missing_field", ["label", "type", "size", "time", "super_uids"])
def test_encoded_graph_rejects_missing_node_fields(missing_field: str) -> None:
    _raw, _model, encoded = _no_op_encoded(prefix=f"missing-{missing_field}")
    del encoded.nodes[0][missing_field]
    with pytest.raises(GraphSchemaError, match="missing fields"):
        validate_encoded_tree(encoded)


def test_encoded_graph_rejects_label_type_disagreement() -> None:
    _raw, _model, encoded = _no_op_encoded(prefix="type-disagreement")
    encoded.nodes[0]["type"] = base_type("B")
    with pytest.raises(ExactTypeError, match="disagrees"):
        validate_encoded_tree(encoded, level="structural")


def test_full_validation_checks_exact_base_labels_against_uid_provenance() -> None:
    _raw, model, encoded = _no_op_encoded(prefix="base-label")
    encoded.nodes[0]["label"] = "B"
    encoded.nodes[0]["type"] = base_type("B")
    # Geometry is still safe, so structural validation can avoid the expensive
    # leaf/provenance scan. Full validation must reject the semantic mismatch.
    validate_encoded_tree(encoded, level="structural")
    with pytest.raises(ProvenanceError, match="exact base label"):
        validate_encoded_tree(encoded, level="full")
    with pytest.raises(ProvenanceError, match="exact label"):
        model.decode(encoded, validate=False)


def test_validate_false_cannot_discard_extra_provenance_uid_on_decode() -> None:
    _raw, model, encoded = _no_op_encoded(prefix="extra-provenance")
    provenance = encoded.graph["tree_coarsening_provenance"]["node_attrs_by_uid"]
    provenance[("extra-provenance", 99)] = {
        "uid": ("extra-provenance", 99),
        "label": "A",
        "time": 99.0,
    }
    before = snapshot_graph(encoded)
    with pytest.raises(ProvenanceError, match="incomplete provenance partition"):
        model.decode(encoded, validate=False)
    assert_graph_unchanged(encoded, before)


def test_validate_false_cannot_collapse_duplicate_visible_uids_on_decode() -> None:
    _raw, model, encoded = _no_op_encoded(prefix="duplicate-visible")
    encoded.nodes[1]["super_uids"] = encoded.nodes[0]["super_uids"]
    before = snapshot_graph(encoded)
    with pytest.raises(ProvenanceError, match="not unique"):
        model.decode(encoded, validate=False)
    assert_graph_unchanged(encoded, before)


def test_validate_false_cannot_invent_missing_visible_uid_on_decode() -> None:
    _raw, model, encoded = _no_op_encoded(prefix="missing-visible")
    encoded.nodes[1]["super_uids"] = (("missing-visible", 999),)
    with pytest.raises(ProvenanceError, match="incomplete provenance partition"):
        model.decode(encoded, validate=False)


def test_encoded_edge_requires_tuple_attach_map() -> None:
    _raw, _model, encoded = _no_op_encoded(prefix="scalar-map")
    encoded.edges[0, 1]["attach_map"] = 0
    with pytest.raises(AttachmentError, match="must be a tuple"):
        validate_encoded_tree(encoded)


@pytest.mark.parametrize("attach_map", [(), (0, 0), (1,)])
def test_encoded_edge_rejects_wrong_attachment_geometry(attach_map: tuple[int, ...]) -> None:
    _raw, _model, encoded = _no_op_encoded(prefix=f"map-{attach_map}")
    encoded.edges[0, 1]["attach_map"] = attach_map
    with pytest.raises(AttachmentError):
        validate_encoded_tree(encoded)


def test_stage_records_reject_duplicate_active_model_ids() -> None:
    _raw, _model, encoded = _no_op_encoded(prefix="duplicate-stage")
    stage = encoded.graph[SCHEMA_KEY]["stages"][0]
    encoded.graph[SCHEMA_KEY]["stages"] = (stage, deepcopy(stage))
    with pytest.raises(StageOrderError, match="duplicate active model_id"):
        validate_encoded_tree(encoded)


def test_stage_cannot_introduce_a_raw_base_label() -> None:
    _raw, _model, encoded = _no_op_encoded(prefix="raw-introduction")
    stage = encoded.graph[SCHEMA_KEY]["stages"][0]
    encoded.graph[SCHEMA_KEY]["stages"] = (
        {"model_id": stage["model_id"], "introduced_labels": ("A",)},
    )
    with pytest.raises(StageOrderError, match="raw-base"):
        validate_encoded_tree(encoded)


def test_exact_type_cannot_reference_inactive_model_id() -> None:
    _raw, _model, encoded = _no_op_encoded(prefix="inactive-type")
    exact = CompositeType(
        model_id="inactive",
        label=("inactive",),
        parent=(-1,),
        components=(base_type("A"),),
        attach=(),
    )
    encoded.nodes[0].update(label=("inactive",), type=exact)
    encoded.graph["tree_coarsening_fitting_sizes"][("inactive",)] = 1
    with pytest.raises(StageOrderError, match="inactive model_id"):
        validate_encoded_tree(encoded)


def test_nested_exact_type_cannot_point_to_a_later_stage() -> None:
    raw = make_tree(["A", "A", "A", "B"], [None, 0, 1, 2], prefix="later-nesting")
    first = NamedVertexCoarsener(
        uids={("later-nesting", 0), ("later-nesting", 1)},
        model_id="earlier",
    ).fit([raw])
    encoded_first = first.transform(raw)
    second = NamedVertexCoarsener(
        labels={("named_component", "earlier"), "A"},
        model_id="later",
    ).fit([encoded_first])
    encoded = second.transform(encoded_first)
    later_node = next(
        node
        for node, data in encoded.nodes(data=True)
        if isinstance(data["type"], CompositeType) and data["type"].model_id == "later"
    )
    later_type = encoded.nodes[later_node]["type"]
    invalid = CompositeType(
        model_id="earlier",
        label=("named_component", "earlier"),
        parent=(-1,),
        components=(later_type,),
        attach=(),
    )
    encoded.nodes[later_node].update(
        label=("named_component", "earlier"),
        type=invalid,
        size=invalid.site_count,
    )
    with pytest.raises(StageOrderError, match="later-stage type"):
        validate_encoded_tree(encoded)


def test_same_id_decoder_with_incompatible_vocabulary_is_rejected() -> None:
    raw = make_tree(["A", "A", "B"], [None, 0, 1], prefix="wrong-decoder")
    model = NamedVertexCoarsener(labels={"A"}, model_id="shared-id").fit([raw])
    encoded = model.transform(raw)
    wrong_rule = EncodingRule(0, "component", ("wrong-output",), 1, {})
    wrong_decoder = StructuralStageDecoder(model_id="shared-id", rules=(wrong_rule,))
    with pytest.raises(TypeOwnershipError, match="does not declare"):
        wrong_decoder.decode(encoded)


def test_same_id_decoder_with_wrong_fitting_size_is_rejected() -> None:
    raw = make_tree(["A", "A", "B"], [None, 0, 1], prefix="wrong-size")
    model = NamedVertexCoarsener(labels={"A"}, fitting_size=2, model_id="same-id-size").fit([raw])
    encoded = model.transform(raw)
    wrong_rule = EncodingRule(
        0,
        "component",
        ("named_component", "same-id-size"),
        3,
        {},
    )
    wrong_decoder = StructuralStageDecoder(model_id="same-id-size", rules=(wrong_rule,))
    with pytest.raises(FittingSizeError, match="records 2"):
        wrong_decoder.decode(encoded)
