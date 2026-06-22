from __future__ import annotations

import ast
import hashlib
import random
from pathlib import Path

import networkx as nx
import pytest

from tree_coarsening import (
    EdgeBPECoarsener,
    NamedVertexCoarsener,
    ParametricStarCoarsener,
    parametric_star_label,
    validate_encoded_tree,
)
from tree_coarsening.coarseners.edge_bpe_numba import numba_available

from conftest import make_tree, raw_signature


ROOT = Path(__file__).resolve().parents[1]
REQUIRES_NUMBA = pytest.mark.skipif(
    not numba_available(),
    reason="optional Numba backend is not installed",
)


def _qualified_functions(tree: ast.AST) -> dict[str, str]:
    result: dict[str, str] = {}

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}"
                result[name] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")

    assert isinstance(tree, ast.Module)
    visit(tree.body)
    return result


def test_numba_bpe_kernel_is_byte_identical_to_v0121() -> None:
    baseline = ROOT / "audit" / "edge_bpe_numba_v0121_protected.py"
    current = ROOT / "tree_coarsening" / "coarseners" / "edge_bpe_numba.py"
    expected = "61fe574b599ca7f3eb755cab69981967fa54f66b7ac79f963b35fe73fe9d9f8d"
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == expected
    assert current.read_bytes() == baseline.read_bytes()


def test_protected_python_bpe_core_functions_are_ast_identical() -> None:
    baseline = _qualified_functions(
        ast.parse((ROOT / "audit" / "edge_bpe_v0121_protected.py").read_text())
    )
    current = _qualified_functions(
        ast.parse((ROOT / "tree_coarsening" / "coarseners" / "edge_bpe.py").read_text())
    )
    protected = {
        "count_pair_score",
        "normalized_pair_score",
        "size_weighted_pair_score",
        "_bump_count",
        "_set_new_label_statistics",
        "_update_label_counts_after_merge",
        "_TokenCodec.intern",
        "_TokenCodec.decode",
        "_TokenCodec.sort_key",
        "_UidRope.merge",
        "_UidRope.flatten",
        "EdgeBPERule.parent_token",
        "EdgeBPERule.child_token",
        "_CompactEdgeTree.from_raw_graph",
        "_CompactEdgeTree.rebuild_edge_index",
        "_CompactEdgeTree._edge_is_live",
        "_CompactEdgeTree._edge_key_unchecked",
        "_CompactEdgeTree._edge_key",
        "_CompactEdgeTree._add_edge",
        "_CompactEdgeTree._remove_edge",
        "_CompactEdgeTree._edge_sort_key",
        "_CompactEdgeTree.contract_pair",
        "EdgeBPECoarsener._select_best_pair",
    }
    assert len(protected) == 23
    assert protected <= baseline.keys()
    assert {name: current[name] for name in protected} == {
        name: baseline[name] for name in protected
    }


def _random_tree(seed: int, nodes: int = 28) -> nx.DiGraph:
    rng = random.Random(seed)
    graph = nx.DiGraph()
    for node in range(nodes):
        graph.add_node(
            node,
            label=rng.choice(("A", "B", "C", "D")),
            time=float(rng.randrange(7)),
            uid=("parity", seed, node),
        )
        if node:
            graph.add_edge(rng.randrange(node), node)
    return graph


@pytest.mark.parametrize("score", ["count", "normalized", "size_weighted"])
@REQUIRES_NUMBA
def test_python_and_numba_learn_identical_rules_and_event_counts(score: str) -> None:
    corpus = [_random_tree(100 + i) for i in range(3)]
    python_model = EdgeBPECoarsener(
        num_merges=7,
        min_pair_count=1,
        pair_score=score,
        backend="python",
        model_id=f"parity-{score}",
    ).fit(corpus)
    numba_model = EdgeBPECoarsener(
        num_merges=7,
        min_pair_count=1,
        pair_score=score,
        backend="numba",
        model_id=f"parity-{score}",
    ).fit(corpus)
    assert numba_model.backend_used_ == "numba"
    assert python_model.history_ == numba_model.history_
    assert python_model.encoder_.rules == numba_model.encoder_.rules


@pytest.mark.parametrize("score", ["count", "normalized", "size_weighted"])
@REQUIRES_NUMBA
def test_python_and_numba_match_on_variable_geometry_encoded_inputs(score: str) -> None:
    raw = [_star_bpe_parity_tree(700 + index, 2 + index) for index in range(4)]
    star = ParametricStarCoarsener(
        2,
        1,
        contract_d=2,
        model_id="parity-variable-star",
    ).fit(raw)
    encoded = [star.transform(graph) for graph in raw]

    family = parametric_star_label("P", "C")
    occurrences = [
        data
        for graph in encoded
        for _node, data in graph.nodes(data=True)
        if data["label"] == family
    ]
    assert {data["size"] for data in occurrences} == {2, 3, 4, 5}
    assert len({data["type"] for data in occurrences}) == 4

    kwargs = {
        "num_merges": 10,
        "min_pair_count": 1,
        "pair_score": score,
        "model_id": f"encoded-parity-{score}",
    }
    python_model = EdgeBPECoarsener(backend="python", **kwargs).fit(encoded)
    numba_model = EdgeBPECoarsener(backend="numba", **kwargs).fit(encoded)

    assert numba_model.backend_used_ == "numba"
    assert python_model.history_ == numba_model.history_
    assert python_model.encoder_.rules == numba_model.encoder_.rules


def _star(arity: int, *, prefix: str) -> nx.DiGraph:
    return make_tree(
        ["P"] + ["C"] * arity,
        [None] + [0] * arity,
        prefix=prefix,
    )


def test_bpe_treats_equal_star_labels_with_unequal_geometry_identically() -> None:
    raw3 = _star(3, prefix="bpe-star3")
    raw5 = _star(5, prefix="bpe-star5")
    star = ParametricStarCoarsener(3, 2, model_id="upstream-star").fit([raw3, raw5])
    encoded3 = star.transform(raw3)
    encoded5 = star.transform(raw5)

    bpe = EdgeBPECoarsener(
        num_merges=1,
        min_pair_count=2,
        pair_score="size_weighted",
        model_id="downstream-bpe",
    ).fit([encoded3, encoded5])
    assert len(bpe.history_) == 1
    event = bpe.history_[0]
    assert event["parent_label"] == "P"
    assert event["child_label"] == ("star", "P", "C")
    assert event["count"] == 2
    assert event["parent_size"] == 1
    assert event["child_size"] == 2

    for raw, upstream in ((raw3, encoded3), (raw5, encoded5)):
        downstream = bpe.transform(upstream)
        validate_encoded_tree(downstream)
        occurrence = next(iter(downstream.nodes(data=True)))[1]
        assert occurrence["label"] == bpe.encoder_.rules[0].output_label
        assert downstream.graph["tree_coarsening_fitting_sizes"][occurrence["label"]] == 3
        assert raw_signature(star.decode(bpe.decode(downstream))) == raw_signature(raw)


def test_bpe_history_is_independent_of_raw_networkx_keys() -> None:
    graph = _random_tree(81, nodes=45)
    mapping = {node: ("opaque", 1000 - node) for node in graph.nodes}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    kwargs = {
        "num_merges": 12,
        "min_pair_count": 1,
        "pair_score": "count",
        "backend": "python",
        "model_id": "key-invariance",
    }
    first = EdgeBPECoarsener(**kwargs).fit([graph])
    second = EdgeBPECoarsener(**kwargs).fit([relabeled])
    assert first.history_ == second.history_


def _star_bpe_parity_tree(seed: int, arity: int) -> nx.DiGraph:
    """A small fixture whose downstream BPE overlaps depend on stable ordering."""

    graph = nx.DiGraph()
    graph.add_node(0, label="P", time=0.0, uid=(seed, 0))
    next_node = 1
    for sibling in range(arity):
        child = next_node
        next_node += 1
        graph.add_node(
            child,
            label="C",
            time=float(sibling % 2),
            uid=(seed, child),
        )
        graph.add_edge(0, child)
        parent = child
        for depth in range(2 + (sibling + seed) % 3):
            node = next_node
            next_node += 1
            label = ("A", "B", "D", "E")[(sibling + depth + seed) % 4]
            graph.add_node(
                node,
                label=label,
                time=float(depth + 1),
                uid=(seed, node),
            )
            graph.add_edge(parent, node)
            parent = node
    return graph


def _historical_token(label: object) -> object:
    if isinstance(label, tuple) and label and label[0] == "edge_bpe":
        return ("edge_bpe", int(label[-1]))
    if isinstance(label, tuple):
        return tuple(_historical_token(item) for item in label)
    return label


def test_star_then_bpe_preserves_pre_refactor_overlap_history() -> None:
    raw = [_star_bpe_parity_tree(1000 + index, 2 + index % 5) for index in range(4)]
    star = ParametricStarCoarsener(
        2,
        1,
        contract_d=2,
        model_id="ordering-star",
    ).fit(raw, validate=False)
    encoded = [star.transform(graph, validate=False) for graph in raw]
    bpe = EdgeBPECoarsener(
        num_merges=12,
        min_pair_count=1,
        pair_score="count",
        backend="python",
        model_id="ordering-bpe",
    ).fit(encoded, validate=False)

    observed = [
        (
            _historical_token(event["parent_label"]),
            _historical_token(event["child_label"]),
            event["count"],
            event["actual_events"],
        )
        for event in bpe.history_
    ]
    assert observed == [
        ("B", "D", 8, 8),
        ("E", "A", 7, 7),
        (("star", "P", "C"), ("edge_bpe", 1), 4, 3),
        (("edge_bpe", 2), ("edge_bpe", 0), 3, 3),
        (("edge_bpe", 3), "D", 3, 3),
        (("edge_bpe", 4), ("edge_bpe", 1), 3, 2),
        ("A", ("edge_bpe", 0), 3, 3),
        (("edge_bpe", 6), "E", 2, 2),
        (("edge_bpe", 5), "B", 2, 2),
        ("P", ("edge_bpe", 8), 2, 2),
        (("star", "P", "C"), ("edge_bpe", 6), 1, 1),
        (("edge_bpe", 9), ("edge_bpe", 7), 1, 1),
    ]


def _integer_uid_tree(seed: int, nodes: int = 45) -> nx.DiGraph:
    """Return a tree whose integer UID order differs from lexical ``repr`` order."""

    rng = random.Random(seed)
    graph = nx.DiGraph()
    for node in range(nodes):
        graph.add_node(
            node,
            label=rng.choice(("A", "B", "C", "D")),
            time=float(rng.randrange(8)),
            uid=node,
        )
        if node:
            graph.add_edge(rng.randrange(node), node)
    return graph


def _visible_occurrence_semantics(graph: nx.DiGraph) -> tuple[object, object]:
    nodes = {
        tuple(data["super_uids"]): (
            data["label"],
            data["type"],
            data["size"],
            data["time"],
        )
        for _node, data in graph.nodes(data=True)
    }
    edges = frozenset(
        (
            tuple(graph.nodes[parent]["super_uids"]),
            tuple(graph.nodes[child]["super_uids"]),
            tuple(data["attach_map"]),
        )
        for parent, child, data in graph.edges(data=True)
    )
    return nodes, edges


def test_noop_upstream_stage_does_not_change_bpe_overlap_order() -> None:
    raw = _integer_uid_tree(4)
    noop = NamedVertexCoarsener(
        labels={"ABSENT"},
        model_id="noop-before-bpe",
    ).fit([raw], validate=False)
    staged = noop.transform(raw, validate=False)

    kwargs = {
        "num_merges": 12,
        "min_pair_count": 1,
        "pair_score": "count",
        "backend": "python",
        "model_id": "bpe-after-noop",
    }
    raw_fit = EdgeBPECoarsener(**kwargs).fit([raw], validate=False)
    staged_fit = EdgeBPECoarsener(**kwargs).fit([staged], validate=False)

    assert raw_fit.history_ == staged_fit.history_
    assert raw_fit.encoder_.rules == staged_fit.encoder_.rules
    assert _visible_occurrence_semantics(raw_fit.transform(raw, validate=False)) == (
        _visible_occurrence_semantics(raw_fit.transform(staged, validate=False))
    )
