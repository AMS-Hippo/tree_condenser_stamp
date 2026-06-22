from __future__ import annotations

from copy import deepcopy

import networkx as nx

from tree_coarsening import EdgeBPECoarsener, NamedVertexCoarsener, ParametricStarCoarsener

from conftest import encoded_occurrence_signature, make_tree, raw_signature


def _reinsert_in_reverse_order(graph: nx.DiGraph) -> nx.DiGraph:
    out = nx.DiGraph()
    out.graph.update(deepcopy(graph.graph))
    for node in reversed(tuple(graph.nodes)):
        out.add_node(node, **deepcopy(dict(graph.nodes[node])))
    for parent, child, data in reversed(tuple(graph.edges(data=True))):
        out.add_edge(parent, child, **deepcopy(dict(data)))
    return out


def _determinism_tree() -> nx.DiGraph:
    return make_tree(
        ["P", "C", "C", "C", "A", "B", "B", "D", "A", "E"],
        [None, 0, 0, 0, 1, 4, 2, 6, 3, 8],
        prefix="determinism",
    )


def test_fit_and_transform_ignore_networkx_insertion_order() -> None:
    original = _determinism_tree()
    reordered = _reinsert_in_reverse_order(original)

    model_pairs = (
        (
            ParametricStarCoarsener(2, 1, model_id="deterministic-star"),
            ParametricStarCoarsener(2, 1, model_id="deterministic-star"),
        ),
        (
            EdgeBPECoarsener(
                num_merges=8,
                min_pair_count=1,
                backend="python",
                model_id="deterministic-bpe",
            ),
            EdgeBPECoarsener(
                num_merges=8,
                min_pair_count=1,
                backend="python",
                model_id="deterministic-bpe",
            ),
        ),
        (
            NamedVertexCoarsener(
                labels={"A", "B", "D", "E"},
                model_id="deterministic-named",
            ),
            NamedVertexCoarsener(
                labels={"A", "B", "D", "E"},
                model_id="deterministic-named",
            ),
        ),
    )

    for first, second in model_pairs:
        first.fit([original])
        second.fit([reordered])
        assert first.encoder_.rules == second.encoder_.rules
        if isinstance(first, EdgeBPECoarsener):
            assert first.history_ == second.history_

        first_output = first.transform(original)
        second_output = first.transform(reordered)
        assert encoded_occurrence_signature(first_output) == encoded_occurrence_signature(
            second_output
        )
        assert tuple(first_output.nodes) == tuple(range(first_output.number_of_nodes()))
        assert tuple(second_output.nodes) == tuple(range(second_output.number_of_nodes()))
        assert raw_signature(first.decode(first_output)) == raw_signature(original)
        assert raw_signature(first.decode(second_output)) == raw_signature(reordered)
