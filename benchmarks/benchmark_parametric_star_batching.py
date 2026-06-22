"""Benchmark batched ParametricStar transform against its former sequential loop.

The sequential reference intentionally mirrors the implementation immediately
before batching was introduced. It is retained only as an audit baseline, not as
an alternative public backend.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tree_coarsening import ParametricStarCoarsener  # noqa: E402
from tree_coarsening.contraction import (  # noqa: E402
    Contraction,
    apply_mixed_contraction_batch,
)
from tree_coarsening.schema import append_stage, prepare_graph  # noqa: E402
from tree_coarsening.validation import (  # noqa: E402
    node_order_key,
    relabel_to_consecutive_parent_first,
    validate_encoded_tree,
)


def make_star_forest(groups: int) -> nx.DiGraph:
    """Return R -> P_i -> (C_i0, C_i1) for ``groups`` independent stars."""

    graph = nx.DiGraph()
    graph.add_node(0, label="R", time=0.0, uid=("benchmark", 0))
    next_node = 1
    for _ in range(groups):
        parent = next_node
        next_node += 1
        graph.add_node(
            parent,
            label="P",
            time=float(parent),
            uid=("benchmark", parent),
        )
        graph.add_edge(0, parent)
        for _child_index in range(2):
            child = next_node
            next_node += 1
            graph.add_node(
                child,
                label="C",
                time=float(child),
                uid=("benchmark", child),
            )
            graph.add_edge(parent, child)
    return graph


def sequential_reference(
    encoder: Any,
    graph: nx.DiGraph,
    *,
    validate: str | bool,
) -> nx.DiGraph:
    """Mirror the former one-contraction-at-a-time ParametricStar transform."""

    current = prepare_graph(graph, validate=validate)
    append_stage(current, model_id=encoder.model_id, vocab=encoder.vocab)
    for rule in encoder.rules:
        parent_label = rule.pattern["parent_label"]
        child_label = rule.pattern["child_label"]
        root = next(node for node, degree in current.in_degree() if degree == 0)
        stack = [root]
        while stack:
            parent = stack.pop()
            if parent not in current:
                continue
            if current.nodes[parent]["label"] == parent_label:
                members = tuple(
                    child
                    for child in sorted(
                        current.successors(parent),
                        key=lambda child: node_order_key(current, child),
                    )
                    if current.nodes[child]["label"] == child_label
                )
                if len(members) >= encoder.contract_d:
                    current = apply_mixed_contraction_batch(
                        current,
                        model_id=encoder.model_id,
                        planned=((rule, Contraction(rule.rule_index, members)),),
                        _validate_result=False,
                    )
            children_after = sorted(
                current.successors(parent),
                key=lambda child: node_order_key(current, child),
            )
            stack.extend(reversed(children_after))
    current = relabel_to_consecutive_parent_first(current)
    validate_encoded_tree(current, level=validate)
    return current


def occurrence_signature(graph: nx.DiGraph) -> tuple[Any, ...]:
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


def median_runtime(function: Any, repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[50, 100, 200, 400, 533],
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--validate",
        choices=("false", "structural", "full"),
        default="full",
        help="public validation level used by both implementations",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1 or any(groups < 1 for groups in args.groups):
        parser.error("groups and repetitions must be positive")

    validate: str | bool = False if args.validate == "false" else args.validate
    rows = []
    for groups in args.groups:
        graph = make_star_forest(groups)
        training = make_star_forest(1)
        model = ParametricStarCoarsener(2, 1, model_id="star-batch-benchmark").fit([training])
        batched_result = model.transform(graph, validate=validate)
        sequential_result = sequential_reference(model.encoder_, graph, validate=validate)
        if occurrence_signature(batched_result) != occurrence_signature(sequential_result):
            raise RuntimeError("batched and sequential transforms disagree")

        batched = median_runtime(
            lambda: model.transform(graph, validate=validate),
            args.repetitions,
        )
        sequential = median_runtime(
            lambda: sequential_reference(model.encoder_, graph, validate=validate),
            args.repetitions,
        )
        row = {
            "groups": groups,
            "nodes": graph.number_of_nodes(),
            "batched_seconds": batched,
            "sequential_seconds": sequential,
            "speedup": sequential / batched,
        }
        rows.append(row)
        print(
            f"nodes={row['nodes']:4d} groups={groups:4d} "
            f"batched={batched:.6f}s sequential={sequential:.6f}s "
            f"speedup={row['speedup']:.1f}x"
        )

    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "networkx": nx.__version__,
        },
        "validation": args.validate,
        "repetitions": args.repetitions,
        "rows": rows,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
