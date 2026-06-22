"""Compare schema-1 NamedVertex performance with the v0.12.1 implementation."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def make_fixture(nx: Any, groups: int):
    graph = nx.DiGraph()
    graph.add_node(0, label="ROOT", time=0.0, uid="named-root")
    next_node = 1
    for group_i in range(groups):
        spacer = next_node
        next_node += 1
        graph.add_node(
            spacer,
            label="X",
            time=float(group_i + 1),
            uid=f"named-spacer-{group_i}",
        )
        graph.add_edge(0, spacer)
        parent = spacer
        for site_i, label in enumerate(("A", "B", "A", "B", "A")):
            node = next_node
            next_node += 1
            graph.add_node(
                node,
                label=label,
                time=float(group_i + 1) + 0.01 * (site_i + 1),
                uid=f"named-component-{group_i}-{site_i}",
            )
            graph.add_edge(parent, node)
            parent = node
    return graph


def raw_signature(graph: Any) -> tuple[Any, ...]:
    nodes = {
        data["uid"]: (data["label"], float(data["time"])) for _, data in graph.nodes(data=True)
    }
    edges = frozenset(
        (graph.nodes[parent]["uid"], graph.nodes[child]["uid"]) for parent, child in graph.edges
    )
    return nodes, edges


def median_seconds(function, repetitions: int) -> float:
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def worker(source_root: Path, generation: str, groups: int, repetitions: int) -> None:
    sys.path.insert(0, str(source_root))
    import networkx as nx
    from tree_coarsening import NamedVertexCoarsener

    graph = make_fixture(nx, groups)
    model = NamedVertexCoarsener(
        labels={"A", "B"},
        component_policy="all",
        model_id=f"named-cross-{generation}-{groups}",
    )
    if generation == "schema1":
        model.fit([graph], validate="full")
        encoded = model.transform(graph, validate="full")
        decoded = model.decode(encoded, validate="full")
        transform_seconds = median_seconds(
            lambda: model.transform(graph, validate="full"), repetitions
        )
        decode_seconds = median_seconds(lambda: model.decode(encoded, validate="full"), repetitions)
    else:
        model.fit([graph])
        encoded = model.transform(graph, validate=True)
        decoded = model.decode(encoded, validate=True)
        transform_seconds = median_seconds(
            lambda: model.transform(graph, validate=True), repetitions
        )
        decode_seconds = median_seconds(lambda: model.decode(encoded, validate=True), repetitions)

    if raw_signature(decoded) != raw_signature(graph):
        raise RuntimeError(f"{generation} failed exact raw round trip")
    print(
        json.dumps(
            {
                "generation": generation,
                "groups": groups,
                "nodes": graph.number_of_nodes(),
                "encoded_nodes": encoded.number_of_nodes(),
                "transform_seconds": transform_seconds,
                "decode_seconds": decode_seconds,
                "networkx": nx.__version__,
                "python": platform.python_version(),
            }
        )
    )


def invoke_worker(
    *,
    script: Path,
    source_root: Path,
    generation: str,
    groups: int,
    repetitions: int,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--worker",
            "--source-root",
            str(source_root),
            "--generation",
            generation,
            "--groups",
            str(groups),
            "--repetitions",
            str(repetitions),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--generation", choices=("schema1", "v0.12.1"))
    parser.add_argument("--groups", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--old-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.worker:
        if args.source_root is None or args.generation is None or len(args.groups) != 1:
            parser.error("worker mode requires one group count, source root, and generation")
        worker(args.source_root, args.generation, args.groups[0], args.repetitions)
        return

    if args.old_root is None:
        parser.error("--old-root is required")
    script = Path(__file__).resolve()
    current_root = script.parents[1]
    rows = []
    for groups in args.groups:
        current = invoke_worker(
            script=script,
            source_root=current_root,
            generation="schema1",
            groups=groups,
            repetitions=args.repetitions,
        )
        old = invoke_worker(
            script=script,
            source_root=args.old_root,
            generation="v0.12.1",
            groups=groups,
            repetitions=args.repetitions,
        )
        if current["nodes"] != old["nodes"] or current["encoded_nodes"] != old["encoded_nodes"]:
            raise RuntimeError("generations produced different graph sizes")
        rows.append(
            {
                "groups": groups,
                "nodes": current["nodes"],
                "encoded_nodes": current["encoded_nodes"],
                "schema1_transform_seconds": current["transform_seconds"],
                "v0_12_1_transform_seconds": old["transform_seconds"],
                "transform_ratio": current["transform_seconds"] / old["transform_seconds"],
                "schema1_decode_seconds": current["decode_seconds"],
                "v0_12_1_decode_seconds": old["decode_seconds"],
                "decode_ratio": current["decode_seconds"] / old["decode_seconds"],
            }
        )
    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "repetitions": args.repetitions,
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
