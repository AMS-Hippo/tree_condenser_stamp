"""Shared deterministic fixtures and signatures for benchmark notebooks."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from collections.abc import Hashable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import networkx as nx

from tree_coarsening import (
    EdgeBPECoarsener,
    NamedVertexCoarsener,
    ParametricStarCoarsener,
    TreeCoarsener,
)

TreeShape = Literal["near_binary", "near_star"]
MethodName = Literal["star", "bpe_python", "bpe_numba", "named"]

_SELECTED_LABELS = frozenset({"Z0", "Z1"})


def _add_raw_node(
    graph: nx.DiGraph,
    node: int,
    *,
    label: str,
    time: float,
    shape: TreeShape,
    seed: int,
) -> None:
    graph.add_node(
        node,
        label=label,
        time=float(time),
        uid=f"bench:{shape}:{seed}:{node}",
        payload=(shape, node % 17),
    )


def make_benchmark_tree(
    shape: TreeShape,
    n_nodes: int,
    *,
    seed: int = 20260620,
) -> nx.DiGraph:
    """Build an exact-size deterministic raw tree used by both benchmark notebooks.

    ``near_binary`` is a breadth-first complete binary tree. A small
    deterministic label-state transition keeps label proportions stable across
    depth while making ``P`` the only state with equal-labeled siblings.

    ``near_star`` has one high-degree root and short four-node branches of the
    form ``S -> Z0 -> Z1 -> Z0``. It is star-like without being a trivial
    one-layer star, so all three coarseners have nontrivial work to do.
    """

    if shape not in {"near_binary", "near_star"}:
        raise ValueError("shape must be 'near_binary' or 'near_star'.")
    if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) or n_nodes < 7:
        raise ValueError("n_nodes must be an integer of at least 7.")

    graph = nx.DiGraph()
    graph.graph.update(
        benchmark_shape=shape,
        benchmark_seed=seed,
        benchmark_nodes=n_nodes,
        user_metadata={"purpose": "release-performance", "revision": 1},
    )

    if shape == "near_binary":
        child_labels = {
            "P": ("S", "S"),
            "S": ("Z0", "Z1"),
            "Z0": ("Z1", "P"),
            "Z1": ("P", "Z0"),
        }
        labels: dict[int, str] = {0: "P"}
        for node in range(n_nodes):
            depth = (node + 1).bit_length() - 1
            label = labels[node]
            _add_raw_node(
                graph,
                node,
                label=label,
                time=depth + node / (1000.0 * n_nodes),
                shape=shape,
                seed=seed,
            )
            left = 2 * node + 1
            right = left + 1
            if left < n_nodes:
                graph.add_edge(node, left)
                labels[left] = child_labels[label][0]
            if right < n_nodes:
                graph.add_edge(node, right)
                labels[right] = child_labels[label][1]
        return graph

    _add_raw_node(graph, 0, label="P", time=0.0, shape=shape, seed=seed)
    next_node = 1
    branch = 0
    branch_labels = ("S", "Z0", "Z1", "Z0")
    while next_node < n_nodes:
        parent = 0
        for position, label in enumerate(branch_labels):
            if next_node >= n_nodes:
                break
            node = next_node
            next_node += 1
            _add_raw_node(
                graph,
                node,
                label=label,
                time=1.0 + position + branch / max(1.0, float(n_nodes)),
                shape=shape,
                seed=seed,
            )
            graph.add_edge(parent, node)
            parent = node
        branch += 1
    return graph


def make_coarsener(
    method: MethodName,
    *,
    model_id: str,
    bpe_num_merges: int = 12,
    bpe_min_pair_count: int = 2,
) -> TreeCoarsener:
    """Construct one benchmark coarsener with stable, visible settings."""

    if method == "star":
        return ParametricStarCoarsener(2, 1, contract_d=2, model_id=model_id)
    if method == "named":
        return NamedVertexCoarsener(
            labels=_SELECTED_LABELS,
            component_policy="all",
            fitting_size=1,
            model_id=model_id,
        )
    if method in {"bpe_python", "bpe_numba"}:
        return EdgeBPECoarsener(
            num_merges=bpe_num_merges,
            min_pair_count=bpe_min_pair_count,
            pair_score="count",
            backend="python" if method == "bpe_python" else "numba",
            model_id=model_id,
        )
    raise ValueError(f"unknown benchmark method {method!r}.")


def method_display_name(method: str) -> str:
    return {
        "star": "Parametric Star",
        "bpe_python": "Edge BPE (Python)",
        "bpe_numba": "Edge BPE (Numba)",
        "named": "Named Vertex",
    }.get(method, method)


def numba_is_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def warm_numba_backend(*, validate: str | bool = False) -> float | None:
    """Compile the optional BPE fitting path outside measured benchmark rows."""

    if not numba_is_available():
        return None
    graph = make_benchmark_tree("near_binary", 127, seed=0)
    model = make_coarsener(
        "bpe_numba",
        model_id="benchmark-numba-warmup",
        bpe_num_merges=2,
        bpe_min_pair_count=1,
    )
    started = perf_counter()
    model.fit([graph], validate=validate)
    return perf_counter() - started


def _freeze(value: Any) -> Any:
    """Convert nested attributes into a deterministic comparison value."""

    if isinstance(value, Mapping):
        items = [(_freeze(key), _freeze(item)) for key, item in value.items()]
        return ("mapping", tuple(sorted(items, key=lambda pair: repr(pair[0]))))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_freeze(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted((_freeze(item) for item in value), key=repr)))
    return value


def raw_signature(graph: nx.DiGraph) -> tuple[Any, ...]:
    """Canonical raw graph signature including node and graph attributes."""

    uid_by_node: dict[Hashable, Hashable] = {
        node: data["uid"] for node, data in graph.nodes(data=True)
    }
    nodes = tuple(
        sorted(
            ((_freeze(data["uid"]), _freeze(dict(data))) for _, data in graph.nodes(data=True)),
            key=repr,
        )
    )
    edges = tuple(
        sorted(
            (
                _freeze(uid_by_node[parent]),
                _freeze(uid_by_node[child]),
                _freeze(dict(data)),
            )
            for parent, child, data in graph.edges(data=True)
        )
    )
    return nodes, edges, _freeze(dict(graph.graph))


def encoded_signature(graph: nx.DiGraph) -> tuple[Any, ...]:
    """Canonical encoded signature independent of current NetworkX node keys."""

    occurrence_by_node = {
        node: _freeze(tuple(data["super_uids"])) for node, data in graph.nodes(data=True)
    }
    nodes = tuple(
        sorted(
            (
                occurrence_by_node[node],
                _freeze(dict(data)),
            )
            for node, data in graph.nodes(data=True)
        )
    )
    edges = tuple(
        sorted(
            (
                occurrence_by_node[parent],
                occurrence_by_node[child],
                _freeze(dict(data)),
            )
            for parent, child, data in graph.edges(data=True)
        )
    )
    return nodes, edges, _freeze(dict(graph.graph))


def project_root(start: Path | None = None) -> Path:
    """Find the repository root from a notebook or a direct Python process."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "tree_coarsening").is_dir():
            return candidate
    raise RuntimeError("could not locate the tree-coarsening repository root.")


def environment_metadata(root: Path | None = None) -> dict[str, Any]:
    """Return enough environment information to make benchmark rows auditable."""

    root = project_root(root)
    try:
        version = importlib.metadata.version("tree-coarsening")
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "networkx": nx.__version__,
        "tree_coarsening": version,
        "git_commit": commit,
    }
    try:
        metadata["numpy"] = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        metadata["numpy"] = None
    try:
        metadata["numba"] = importlib.metadata.version("numba")
    except importlib.metadata.PackageNotFoundError:
        metadata["numba"] = None
    try:
        import psutil

        metadata["physical_memory_bytes"] = psutil.virtual_memory().total
    except ImportError:
        metadata["physical_memory_bytes"] = None
    return metadata
