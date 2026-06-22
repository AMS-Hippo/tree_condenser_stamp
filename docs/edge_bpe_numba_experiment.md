# Optional Numba backend for Edge BPE

`EdgeBPECoarsener` has a reference Python fitter and an optional compiled
Numba fitter:

```python
EdgeBPECoarsener(..., backend="python")  # default
EdgeBPECoarsener(..., backend="numba")
```

Install the optional dependencies with:

```bash
pip install -e ".[numba]"
```

## Shared semantics

Both fitters use only the current matching-label pair:

```python
(parent_node["label"], child_node["label"])
```

They must agree on:

- raw matching-edge frequency, including overlapping occurrences;
- `count`, `normalized`, and `size_weighted` scores;
- deterministic score ties and vertex-disjoint overlap selection;
- ordered rules, raw counts, and actual contraction-event counts;
- additive output fitting size.

Size-aware scores use label-level fitting sizes, not occurrence-specific exact
node sizes. Exact types and attachment maps are intentionally absent from the
fitting pair index.

## Compiled state

The protected Numba kernel uses flattened arrays for parent/child/sibling
links, integer label IDs, fitting sizes, times, liveness, tree IDs, and local
pair-bucket maintenance. Contractions update affected pair buckets instead of
recounting the whole forest.

Only fitting is compiled. Raw/schema normalization, stable tie resolution,
artifact construction, NetworkX transformation, exact type creation, and
decoding remain at the Python boundary.

## Warm-up and benchmark

The first Numba call may include JIT compilation. Benchmark first-call and
warmed behavior separately:

```bash
python benchmarks/benchmark_edge_bpe_numba.py path --nodes 100000
python benchmarks/benchmark_edge_bpe_numba.py star --nodes 100000
```

The cross-generation parity and timing audit is recorded in
`audit/BPE_PARITY_AND_PERFORMANCE.md`. The complete Numba source is also
byte-protected against the v0.12.1 baseline.
