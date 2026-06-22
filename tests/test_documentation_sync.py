from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SCHEMA_SHA256 = "30a3005bbf0266ce8058cc6eb52fcd159b722b71b3f811231e91a2070c64b593"


def test_normative_api_copies_are_byte_identical_and_frozen() -> None:
    normative = (ROOT / "CURRENT_SCHEMA1_API.md").read_bytes()
    assert hashlib.sha256(normative).hexdigest() == FROZEN_SCHEMA_SHA256
    assert (ROOT / "tree_coarsening_api.md").read_bytes() == normative
    assert (ROOT / "docs" / "tree_coarsening_api.md").read_bytes() == normative
    assert (ROOT / "audit" / "FROZEN_SCHEMA.sha256").read_text().split()[0] == (
        FROZEN_SCHEMA_SHA256
    )


def test_current_documentation_does_not_reintroduce_legacy_api() -> None:
    paths = (
        ROOT / "README.md",
        ROOT / "CONTRIBUTING_COARSENERS.md",
        ROOT / "docs" / "parametric_star_coarsener.md",
        ROOT / "docs" / "edge_bpe_numba_experiment.md",
        ROOT / "notebooks" / "star_coarsener_example.ipynb",
        ROOT / "notebooks" / "edge_bpe_coarsener_example.ipynb",
        ROOT / "notebooks" / "named_vertex_coarsener_example.ipynb",
        ROOT / "benchmarks" / "benchmark_edge_bpe_numba.py",
        ROOT / "benchmarks" / "edge_bpe_timing_sweep.ipynb",
    )
    stale_fragments = (
        'node["super_label"]',
        'edge["attach_index"]',
        "from tree_coarsening import StarCoarsener",
        "vocab.symbols",
        "vocab.creation_order",
        "validate_inputs=",
    )
    for path in paths:
        content = path.read_text()
        for stale in stale_fragments:
            assert stale not in content, f"{path.relative_to(ROOT)} contains {stale!r}"
