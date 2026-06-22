from __future__ import annotations

from types import MappingProxyType

import networkx as nx
import pytest

from tree_coarsening import (
    CompositeType,
    ConfigurationError,
    EncodingRule,
    ExactTypeError,
    GraphSchemaError,
    LabelMetadataError,
    Vocabulary,
    base_type,
    exact_root_count,
    exact_site_count,
    exact_type_label,
    normalize_raw_graph,
    validate_encoded_tree,
    validate_raw_tree,
)

from conftest import assert_graph_unchanged, make_tree, snapshot_graph


def test_composite_type_keeps_geometry_out_of_matching_label() -> None:
    label = ("family", "A", "B")
    binary = CompositeType(
        model_id="stage",
        label=label,
        parent=(-1, -1),
        components=(base_type("B"), base_type("B")),
        attach=(),
    )
    ternary = CompositeType(
        model_id="stage",
        label=label,
        parent=(-1, -1, -1),
        components=(base_type("B"), base_type("B"), base_type("B")),
        attach=(),
    )
    assert exact_type_label(binary) == exact_type_label(ternary) == label
    assert exact_site_count(binary) == 2
    assert exact_site_count(ternary) == 3
    assert exact_root_count(binary) == 2
    assert exact_root_count(ternary) == 3
    assert binary != ternary


def test_composite_type_validates_geometry() -> None:
    with pytest.raises(ExactTypeError):
        CompositeType(
            model_id="stage",
            label="X",
            parent=(-1, 0),
            components=(base_type("A"), base_type("B")),
            attach=(1,),
        )
    with pytest.raises(ExactTypeError):
        CompositeType(
            model_id="stage",
            label="X",
            parent=(1, 0),
            components=(base_type("A"), base_type("B")),
            attach=(0, 0),
        )


def test_rule_and_vocabulary_are_read_only() -> None:
    source_pattern = {"labels": ["A", "B"], "config": {"x": 1}}
    rule = EncodingRule(
        rule_index=0,
        operation="component",
        output_label=("out", 0),
        output_fitting_size=3,
        pattern=source_pattern,
        parameter_names=("topology",),
    )
    source_pattern["labels"].append("C")
    assert rule.pattern["labels"] == ("A", "B")
    assert isinstance(rule.pattern, MappingProxyType)
    with pytest.raises(TypeError):
        rule.pattern["new"] = 3  # type: ignore[index]

    vocab = Vocabulary((rule,))
    assert vocab.labels == (("out", 0),)
    assert vocab.fitting_size(("out", 0)) == 3
    assert vocab.as_dict() == {("out", 0): 3}


def test_rule_pattern_keys_follow_the_string_key_contract() -> None:
    with pytest.raises(ConfigurationError, match="pattern keys must be strings"):
        EncodingRule(
            rule_index=0,
            operation="component",
            output_label=("out", 0),
            output_fitting_size=1,
            pattern={0: "not-a-string-key"},  # type: ignore[dict-item]
        )


def test_raw_normalization_is_nonmutating_and_exact(chain4: nx.DiGraph) -> None:
    before = snapshot_graph(chain4)
    encoded = normalize_raw_graph(chain4)
    assert_graph_unchanged(chain4, before)
    validate_encoded_tree(encoded)

    assert set(encoded.graph) == {
        "tree_coarsening_schema",
        "tree_coarsening_fitting_sizes",
        "tree_coarsening_provenance",
    }
    assert encoded.graph["tree_coarsening_schema"] == {"version": "1.0", "stages": ()}
    assert encoded.graph["tree_coarsening_fitting_sizes"] == {"A": 1, "B": 1, "C": 1}
    for node, data in encoded.nodes(data=True):
        raw = chain4.nodes[node]
        assert set(data) == {"label", "type", "size", "time", "super_uids"}
        assert data == {
            "label": raw["label"],
            "type": ("base", raw["label"]),
            "size": 1,
            "time": float(raw["time"]),
            "super_uids": (raw["uid"],),
        }
    assert all(data == {"attach_map": (0,)} for *_edge, data in encoded.edges(data=True))


def test_raw_contract_rejects_reserved_or_invalid_data() -> None:
    graph = make_tree(["A"], [None])
    graph.nodes[0]["size"] = 1
    with pytest.raises(GraphSchemaError):
        validate_raw_tree(graph)

    graph = make_tree(["A"], [None])
    graph.nodes[0]["label"] = ("not", "raw")
    with pytest.raises(LabelMetadataError):
        validate_raw_tree(graph)

    graph = make_tree(["A"], [None])
    graph.graph["tree_coarsening_custom"] = True
    with pytest.raises(GraphSchemaError):
        validate_raw_tree(graph)
