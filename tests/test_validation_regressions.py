from __future__ import annotations

from copy import deepcopy

import pytest

from tree_coarsening import (
    FITTING_SIZES_KEY,
    PROVENANCE_KEY,
    EdgeBPECoarsener,
    ProvenanceError,
    validate_encoded_tree,
)

from conftest import assert_graph_unchanged, make_tree, snapshot_graph


def test_full_validation_matches_exact_base_labels_to_uid_provenance() -> None:
    raw = make_tree(["A", "B"], [None, 0], prefix="labels")
    model = EdgeBPECoarsener(
        num_merges=0,
        min_pair_count=1,
        model_id="label-check",
    ).fit([raw])
    encoded = model.transform(raw)
    encoded.graph[PROVENANCE_KEY]["node_attrs_by_uid"][("labels", 1)]["label"] = "A"

    # Structural/disabled validation avoids the global exact-leaf/provenance
    # scan. The malformed state is still rejected when complete decoding uses
    # that information, even with eager validation disabled.
    validate_encoded_tree(encoded, level="structural")
    validate_encoded_tree(encoded, level=False)
    with pytest.raises(ProvenanceError, match="exact base label"):
        validate_encoded_tree(encoded, level="full")
    with pytest.raises(ProvenanceError, match="exact label"):
        model.decode(encoded, validate=False)


@pytest.mark.parametrize("level", ["structural", False])
def test_local_duplicate_uid_is_rejected_even_without_full_scan(level: str | bool) -> None:
    raw = make_tree(["A", "B"], [None, 0], prefix="dupe")
    model = EdgeBPECoarsener(
        num_merges=1,
        min_pair_count=1,
        model_id="dupe-stage",
    ).fit([raw])
    encoded = model.transform(raw)
    composite = next(node for node, data in encoded.nodes(data=True) if data["size"] == 2)
    encoded.nodes[composite]["super_uids"] = (("dupe", 0), ("dupe", 0))

    with pytest.raises(ProvenanceError, match="repeats a provenance UID"):
        validate_encoded_tree(encoded, level=level)


def test_validate_false_complete_decode_cannot_drop_extra_provenance() -> None:
    raw = make_tree(["A"], [None], prefix="extra")
    model = EdgeBPECoarsener(
        num_merges=0,
        min_pair_count=1,
        model_id="extra-stage",
    ).fit([raw])
    encoded = model.transform(raw)
    encoded.graph[PROVENANCE_KEY]["node_attrs_by_uid"]["extra-uid"] = {
        "uid": "extra-uid",
        "label": "X",
        "time": 4.0,
    }
    encoded.graph[FITTING_SIZES_KEY]["X"] = 1
    before = snapshot_graph(encoded)

    with pytest.raises(ProvenanceError, match="incomplete provenance partition"):
        model.decode(encoded, validate=False)
    assert_graph_unchanged(encoded, before)


def test_full_validation_detects_cross_occurrence_duplicate_uids() -> None:
    raw = make_tree(["A", "B"], [None, 0], prefix="cross")
    model = EdgeBPECoarsener(
        num_merges=0,
        min_pair_count=1,
        model_id="cross-stage",
    ).fit([raw])
    encoded = model.transform(raw)
    nodes = tuple(encoded.nodes)
    encoded.nodes[nodes[1]]["super_uids"] = encoded.nodes[nodes[0]]["super_uids"]
    encoded.nodes[nodes[1]]["type"] = encoded.nodes[nodes[0]]["type"]
    encoded.nodes[nodes[1]]["label"] = encoded.nodes[nodes[0]]["label"]

    with pytest.raises(ProvenanceError, match="one-time partition"):
        validate_encoded_tree(encoded, level="full")


def test_failed_decode_does_not_mutate_nested_metadata() -> None:
    raw = make_tree(["A"], [None], prefix="metadata")
    model = EdgeBPECoarsener(
        num_merges=0,
        min_pair_count=1,
        model_id="metadata-stage",
    ).fit([raw])
    encoded = model.transform(raw)
    malformed = deepcopy(encoded)
    malformed.graph[PROVENANCE_KEY]["node_attrs_by_uid"].clear()
    before = snapshot_graph(malformed)

    with pytest.raises(ProvenanceError):
        model.decode(malformed, validate=False)
    assert_graph_unchanged(malformed, before)
