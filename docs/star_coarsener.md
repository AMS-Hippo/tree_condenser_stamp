# Historical StarCoarsener name

The pre-schema-1 fixed-arity `StarCoarsener` is not part of the schema-1 public
API. Use [`ParametricStarCoarsener`](parametric_star_coarsener.md).

Schema 1.0 deliberately gives all qualifying arities of one learned `(P, C)`
family the same matching label `("star", P, C)`. Exact arity and geometry are
stored per occurrence in `CompositeType`, rather than encoded in the label or a
static decoder recipe.

Historical pre-schema-1 material is retained in the project recovery
checkpoints, not in the release worktree.
