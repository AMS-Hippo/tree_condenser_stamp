# Edge BPE parity and performance audit

**Baseline:** package `0.12.1` from `Before_Refactor.zip`  
**Schema-1 implementation:** current working tree  
**Contract:** `CURRENT_SCHEMA1_API.md` (frozen by `audit/FROZEN_SCHEMA.sha256`)

## Semantic differential

The same deterministic corpus was fitted independently with the v0.12.1 and
schema-1 packages. The comparison covered:

- `count`, `normalized`, and `size_weighted` scoring;
- the Python and Numba fitting backends;
- 28 merge steps on four 4,000-node trees;
- raw matching-edge counts, selected label pairs, endpoint occurrence counts,
  endpoint fitting sizes, scores, and actual vertex-disjoint contraction events.

The only normalization performed before comparison was:

```text
v0.12.1 token: ("edge_bpe", rank)
schema-1 token: ("edge_bpe", model_id, rank)
```

After that token normalization, all six learned histories were exactly equal.
The complete machine-readable histories are retained in:

- `audit/bpe_v0121_benchmark_old.json`
- `audit/bpe_schema1_benchmark_current.json`

## Static protection

- `tree_coarsening/coarseners/edge_bpe_numba.py` is byte-identical to the
  protected v0.12.1 kernel.
- Algorithmic Python functions that require no schema adaptation are AST-frozen
  in `tests/test_edge_bpe_parity_and_protection.py`.
- Schema-boundary functions are checked behaviorally rather than required to
  preserve obsolete representations such as `super_label` or fixed structural
  vocabularies.

## Performance methodology

Environment:

```text
Linux 4.4.0 x86_64, glibc 2.41
Python 3.13.5
NetworkX 3.6.1
NumPy 2.3.5
Numba 0.65.1
```

The benchmark used separate Python processes for the two package generations,
the same deterministic inputs, five timed repetitions, and medians. Numba was
warmed before timing. The fit corpus contained four 4,000-node trees; transform
used one 8,000-node tree; each model requested 28 merges.

### Algorithm-focused run

Both generations used their least eager public validation setting. Ratios below
are schema-1 time divided by v0.12.1 time; a ratio below 1 is faster.

| Backend | Score | Fit v0.12.1 | Fit schema-1 | Ratio | Transform v0.12.1 | Transform schema-1 | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Python | count | 0.1593 s | 0.1509 s | 0.947 | 0.1863 s | 0.1772 s | 0.951 |
| Python | normalized | 0.1551 s | 0.1548 s | 0.998 | 0.1750 s | 0.1430 s | 0.818 |
| Python | size_weighted | 0.1620 s | 0.1422 s | 0.878 | 0.1739 s | 0.1439 s | 0.828 |
| Numba | count | 0.1357 s | 0.1251 s | 0.921 | 0.1647 s | 0.1413 s | 0.858 |
| Numba | normalized | 0.1364 s | 0.1210 s | 0.887 | 0.1581 s | 0.1396 s | 0.883 |
| Numba | size_weighted | 0.1352 s | 0.1211 s | 0.896 | 0.1593 s | 0.1447 s | 0.908 |

No measured case regressed. These are wall-clock observations, not unit-test
thresholds; timing assertions would be too environment-sensitive.

### Default-validation run

A second three-repetition median used each generation's default validation and
`count` scoring:

| Backend | Fit v0.12.1 | Fit schema-1 | Ratio | Transform v0.12.1 | Transform schema-1 | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Python | 0.2622 s | 0.1539 s | 0.587 | 0.2517 s | 0.1626 s | 0.646 |
| Numba | 0.2316 s | 0.1441 s | 0.622 | 0.2502 s | 0.1508 s | 0.603 |

Raw timings are retained in:

- `audit/bpe_v0121_default_validation_benchmark.json`
- `audit/bpe_schema1_default_validation_benchmark.json`

## Conclusion

The schema-1 boundary changes do not alter the learned BPE program on this
cross-generation differential, the optimized Numba kernel is unchanged, and no
performance regression was observed in either the algorithm-focused or default
public-API benchmark.
