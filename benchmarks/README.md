# Performance benchmark notebooks

These notebooks complement the correctness suite. They deliberately assert
exact round trips while measuring performance; a fast incorrect row is never
accepted.

## Recommended release run

Install the checkout with notebook and optional Numba dependencies:

```bash
python -m pip install -e ".[dev,numba]"
```

Then run, in this order:

1. **`coarsener_scaling_sweep.ipynb`** — fit, transform, decode, and total
   end-to-end time for Parametric Star, Edge BPE, and Named Vertex on
   near-binary and near-star trees.
2. **`two_stage_pipeline_sweep.ipynb`** — every ordered pair of the three
   coarseners, direct sequential-application time, manual application versus
   `combine(...)`, exact
   encoded equivalence, and exact reverse-order decoding.
3. **`edge_bpe_timing_sweep.ipynb`** — the existing detailed BPE scorer/backend
   sweep, including explicit Numba warm-up.

Each new notebook has three profiles selected by the environment variable
`TREE_COARSENING_BENCH_PROFILE`:

```text
smoke      small execution check
standard   recommended visual release benchmark (default)
extended   larger, slower sweep for a quiet machine
```

For example, launch Jupyter with:

```bash
TREE_COARSENING_BENCH_PROFILE=extended jupyter lab
```

The notebooks save CSV and JSON metadata beneath `benchmark_results/` when
`SAVE_RESULTS = True`. Compare runs only when Python, NetworkX, dependency
versions, validation level, hardware, and BPE warm-up policy agree.

The pipeline notebook defaults to the Python BPE backend so its plots are not
confounded by compiler start-up. Set `BPE_METHOD = "bpe_numba"` for the warmed
optimized path. To focus only on the primary Star/BPE interaction, replace
`PIPELINES` with the two Star↔BPE orderings in the controls cell.

## Interpretation

- Use the median as the primary line and inspect the minimum/maximum spread.
- Treat Numba compilation as a separate cold-start measurement; do not mix it
  into warmed steady-state rows.
- Inspect rule counts and compression ratios along with time. A method that
  becomes a no-op is not a valid performance improvement.
- Investigate an unexplained steady-state regression of roughly 10–15% or more
  on representative workloads, but do not overreact to one noisy synthetic
  point.
- Run the final benchmark on project data as well as these controlled shapes.

## What these notebooks do not cover

Before a public release, retain separate checks for:

- peak RSS and native-memory growth, especially on large NetworkX corpora;
- clean-process Numba first-use latency, distinct from warmed throughput;
- representative project data rather than only controlled synthetic shapes;
- current-versus-v0.12.1 timing on directly comparable workloads;
- partial decoding and boundary-expansion workloads;
- multi-tree corpora where tree count and total vertex count vary independently;
- all BPE scoring modes and larger merge budgets, covered by the specialized
  BPE sweep;
- Python 3.10–3.13 and minimum/latest dependency environments.
