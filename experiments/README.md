# CARBANAK experiment notebooks

These notebooks are carried forward from the v0.12.1 experiments so the
project-specific CARBANAK workflow was not lost during the schema-1 refactor.
They are **not** part of the automated release gate because they require local
CARBANAK data that is not included in the repository.

The notebooks were lightly API-updated for schema 1.0:

- `StarCoarsener` was renamed to `ParametricStarCoarsener`.
- stage vocabulary inspection uses `len(encoder.vocab)`.
- fitting-corpus transforms are explicit per graph.

Default data location is still controlled by `CARBANAK_DATA_DIR`, falling back
to `../../data/CARBANAK` relative to the notebook/kernel working directory.

Recommended order:

1. `CARBANAK_RUN_FIRST_prep.ipynb`
2. `CARBANAK_star_edge_bpe_experiment.ipynb`
