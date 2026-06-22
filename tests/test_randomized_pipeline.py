from __future__ import annotations

import itertools

from tree_coarsening import (
    CompositeType,
    EdgeBPECoarsener,
    NamedVertexCoarsener,
    ParametricStarCoarsener,
    validate_encoded_tree,
)
from tree_coarsening.utils import add_starbursts, random_galton_watson_tree

from conftest import assert_graph_unchanged, raw_signature, snapshot_graph


def _random_raw(seed: int):
    graph = random_galton_watson_tree(
        max_nodes=26,
        mean_children=1.6,
        labels=("A", "B", "C", "P"),
        seed=seed,
        uid_prefix=f"random-{seed}-",
    )
    graph = add_starbursts(
        graph,
        n_bursts=min(3, len(graph)),
        burst_size_range=(2, 4),
        parent_label="P",
        child_label="S",
        tail_probability=0.5,
        seed=1000 + seed,
        uid_prefix=f"random-{seed}-star-",
    )
    graph.graph["seed"] = seed
    for node in graph:
        graph.nodes[node]["payload"] = {"seed": seed, "node": node}
    return graph


def _stage(name: str, *, model_id: str, score: str):
    if name == "star":
        return ParametricStarCoarsener(2, 1, contract_d=2, model_id=model_id)
    if name == "bpe":
        return EdgeBPECoarsener(
            num_merges=4,
            min_pair_count=1,
            pair_score=score,
            backend="python",
            model_id=model_id,
        )
    return NamedVertexCoarsener(
        labels={"A", "B"},
        component_policy="all",
        model_id=model_id,
    )


def test_randomized_three_stage_round_trips_and_partial_latest_decode() -> None:
    validation_levels = ("full", "structural", False)
    scores = ("count", "normalized", "size_weighted")
    permutations = tuple(itertools.permutations(("star", "bpe", "named")))

    for seed in range(12):
        raw = _random_raw(seed)
        raw_before = snapshot_graph(raw)
        expected = raw_signature(raw)
        for permutation_i, order in enumerate(permutations):
            current = raw
            stages = []
            for stage_i, name in enumerate(order):
                model = _stage(
                    name,
                    model_id=f"random-{seed}-{permutation_i}-{stage_i}-{name}",
                    score=scores[seed % len(scores)],
                ).fit([current], validate=validation_levels[(seed + stage_i) % 3])
                current = model.transform(
                    current,
                    validate=validation_levels[(seed + stage_i + 1) % 3],
                )
                validate_encoded_tree(current, level="full")
                stages.append(model)

            latest = stages[-1]
            visible_owned = [
                node
                for node, data in current.nodes(data=True)
                if isinstance(data["type"], CompositeType)
                and data["type"].model_id == latest.model_id
            ]
            if visible_owned:
                current = latest.decode(
                    current,
                    target=visible_owned[0],
                    by="node",
                    recursive=bool(seed % 2),
                    boundary_policy="expand",
                    validate=validation_levels[seed % 3],
                )
                validate_encoded_tree(current, level="full")

            for model in reversed(stages):
                current = model.decode(
                    current,
                    validate=validation_levels[(seed + 2) % 3],
                )
            assert raw_signature(current) == expected
        assert_graph_unchanged(raw, raw_before)
