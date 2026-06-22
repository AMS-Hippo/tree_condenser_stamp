"""Benchmark independent Parametric Star rules before and after wave batching."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Hashable
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx  # noqa: E402

from tree_coarsening import ParametricStarCoarsener  # noqa: E402
from tree_coarsening.coarseners.parametric_star import ParametricStarEncoder  # noqa: E402
from tree_coarsening.contraction import (  # noqa: E402
    Contraction,
    apply_mixed_contraction_batch,
)
from tree_coarsening.schema import append_stage, prepare_graph  # noqa: E402
from tree_coarsening.validation import (  # noqa: E402
    deterministic_node_order,
    relabel_to_consecutive_parent_first,
    validate_encoded_tree,
)


def make_fixture(rule_count: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, label="ROOT", time=0.0, uid=("many-rules", 0))
    next_node = 1
    for rule_i in range(rule_count):
        parent = next_node
        next_node += 1
        graph.add_node(
            parent,
            label=f"P{rule_i}",
            time=float(rule_i + 1),
            uid=("many-rules", parent),
        )
        graph.add_edge(0, parent)
        for child_i in range(3):
            child = next_node
            next_node += 1
            graph.add_node(
                child,
                label=f"C{rule_i}",
                time=float(rule_i + 1) + 0.01 * (child_i + 1),
                uid=("many-rules", child),
            )
            graph.add_edge(parent, child)
    return graph


def one_batch_per_rule_reference(
    encoder: ParametricStarEncoder,
    graph: nx.DiGraph,
    *,
    validate: str | bool,
) -> nx.DiGraph:
    """Immediate pre-wave implementation: one traversal/rebuild per rule."""

    current = prepare_graph(graph, validate=validate)
    append_stage(current, model_id=encoder.model_id, vocab=encoder.vocab)
    for rule in encoder.rules:
        parent_label = rule.pattern["parent_label"]
        child_label = rule.pattern["child_label"]
        selected: set[Hashable] = set()
        planned: list[tuple[Any, Contraction]] = []
        order = deterministic_node_order(current)
        position = {node: i for i, node in enumerate(order)}
        for parent in order:
            if parent in selected or current.nodes[parent]["label"] != parent_label:
                continue
            members = tuple(
                child
                for child in sorted(current.successors(parent), key=position.__getitem__)
                if current.nodes[child]["label"] == child_label
            )
            if len(members) < encoder.contract_d:
                continue
            planned.append((rule, Contraction(rule.rule_index, members)))
            selected.update(members)
        if planned:
            current = apply_mixed_contraction_batch(
                current,
                model_id=encoder.model_id,
                planned=tuple(planned),
                _validate_result=False,
                _global_order=order,
            )
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
        for _, data in graph.nodes(data=True)
    }
    edges = frozenset(
        (
            tuple(graph.nodes[parent]["super_uids"]),
            tuple(graph.nodes[child]["super_uids"]),
            tuple(data["attach_map"]),
        )
        for parent, child, data in graph.edges(data=True)
    )
    return nodes, edges, dict(graph.graph)


def median_seconds(function, repetitions: int) -> float:
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def run_case(rule_count: int, *, repetitions: int, validate: str | bool) -> dict[str, Any]:
    graph = make_fixture(rule_count)
    model = ParametricStarCoarsener(
        2,
        1,
        model_id=f"many-rules-{rule_count}",
    ).fit([graph], validate=validate)
    encoder = model.encoder_
    assert isinstance(encoder, ParametricStarEncoder)
    assert len(encoder.rules) == rule_count

    wave_result = encoder.transform(graph, validate=validate)
    reference_result = one_batch_per_rule_reference(
        encoder,
        graph,
        validate=validate,
    )
    if occurrence_signature(wave_result) != occurrence_signature(reference_result):
        raise RuntimeError("wave batching changed Parametric Star output")

    wave_seconds = median_seconds(
        lambda: encoder.transform(graph, validate=validate),
        repetitions,
    )
    reference_seconds = median_seconds(
        lambda: one_batch_per_rule_reference(encoder, graph, validate=validate),
        repetitions,
    )
    return {
        "rules": rule_count,
        "nodes": graph.number_of_nodes(),
        "waves": len(encoder._independent_rule_waves()),
        "wave_seconds": wave_seconds,
        "one_batch_per_rule_seconds": reference_seconds,
        "speedup": reference_seconds / wave_seconds,
    }


def parse_validation(value: str) -> str | bool:
    if value == "false":
        return False
    if value in {"full", "structural"}:
        return value
    raise argparse.ArgumentTypeError("validation must be full, structural, or false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--validate", type=parse_validation, default="full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "networkx": nx.__version__,
        },
        "validation": args.validate,
        "repetitions": args.repetitions,
        "rows": [
            run_case(rule_count, repetitions=args.repetitions, validate=args.validate)
            for rule_count in args.rules
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
