"""Greedy continuation scheduling for schema-1 edge BPE.

This module deliberately reuses the ordinary edge-BPE compact trees, Python and
Numba contraction kernels, rule records, encoder, and structural decoder.  The
only algorithmic change is fit-time rule scheduling:

* choose the globally best pair ``(A, B)`` exactly as ordinary BPE does;
* contract its deterministic vertex-disjoint occurrence set;
* then force ``((A, B), B)``, ``(((A, B), B), B)``, and so on until the newly
  created parent label has no current ``B`` child;
* only then return to global pair selection.

Every forced continuation is emitted as an ordinary edge-BPE rule with its own
stage-namespaced output label.  This is necessary because schema 1.0 assigns one
fixed fitting size to each matching label.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

import networkx as nx

from ..decoder import TreeDecoder
from ..encoder import EncodingRule, TreeEncoder
from ..exceptions import ValidationError
from ..schema import fit_corpus_fitting_sizes
from ..stage_decoder import StructuralStageDecoder
from .edge_bpe import (
    EdgeBPECoarsener,
    EdgeBPEEncoder,
    EdgeBPERule,
    EdgeKey,
    _BPEVocabulary,
    _CompactEdgeTree,
    _NUMBA_PAIR_SCORE_MODES,
    _PairSelection,
    _TokenCodec,
    _initial_label_statistics,
    _set_new_label_statistics,
    _update_label_counts_after_merge,
    edge_bpe_token,
)


class GreedyBPECoarsener(EdgeBPECoarsener):
    """Edge BPE that greedily closes each selected pair over repeated children.

    ``num_merges`` counts globally selected seed pairs.  One seed may emit
    several ordered edge rules because its forced continuation is completed
    before the next global pair is selected.  Forced continuation ignores
    ``min_pair_count``; that threshold applies only to seed-pair selection.

    Transformation remains a finite fitted program, just like ordinary BPE.
    Consequently, a transform graph with a longer repeated-child run than any
    run learned during fitting may retain unmatched trailing children. Each
    continuation adds one left-deep ``CompositeType`` layer; the shared
    structural implementation caches derived geometry so normal transformation
    and full decoding do not consume the Python call stack in proportion to that
    depth.
    """

    def _pair_selection_for_key(
        self,
        key: EdgeKey,
        *,
        count: int,
        codec: _TokenCodec,
        label_counts: Sequence[int],
        label_sizes: Sequence[int],
    ) -> _PairSelection | None:
        """Describe one forced pair without applying the seed-count threshold."""

        if count <= 0:
            return None
        parent_id, child_id = key
        parent_count = int(label_counts[parent_id])
        child_count = int(label_counts[child_id])
        parent_size = int(label_sizes[parent_id])
        child_size = int(label_sizes[child_id])
        try:
            score = float(
                self._pair_score_function(
                    int(count),
                    parent_count,
                    child_count,
                    parent_size,
                    child_size,
                )
            )
        except Exception as exc:
            raise ValidationError(
                "pair_score failed for pair "
                f"({codec.decode(parent_id)!r}, {codec.decode(child_id)!r})."
            ) from exc
        if not math.isfinite(score):
            raise ValidationError(
                f"pair_score returned non-finite value {score!r} for "
                f"N(A,B)={count}, N(A)={parent_count}, N(B)={child_count}, "
                f"S(A)={parent_size}, S(B)={child_size}."
            )
        return _PairSelection(
            key=key,
            count=int(count),
            parent_count=parent_count,
            child_count=child_count,
            parent_size=parent_size,
            child_size=child_size,
            score=score,
        )

    def _fit(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        use_numba = self.backend == "numba"
        numba_forest: Any | None = None
        backend_used: Literal["python", "numba"] = "python"
        if use_numba:
            from .edge_bpe_numba import NumbaTrainingForest, require_numba

            require_numba()
            backend_used = "numba"

        fitting_sizes = fit_corpus_fitting_sizes(graphs)
        input_labels = tuple(sorted(fitting_sizes, key=repr))
        vocab = _BPEVocabulary(dict(fitting_sizes))
        codec = _TokenCodec()
        counts: Counter[EdgeKey] = Counter()
        states: list[_CompactEdgeTree] = []
        for graph in graphs:
            states.append(
                _CompactEdgeTree.from_graph(
                    graph,
                    codec=codec,
                    vocab=vocab,
                    pair_counts=None if use_numba else counts,
                    capture_output=False,
                    build_edge_index=not use_numba,
                )
            )

        if use_numba:
            # A single greedy seed can emit one rule per remaining edge, so the
            # ordinary ``num_merges``-based capacity bound is no longer valid.
            max_possible_rules = (
                0 if self.num_merges == 0 else sum(len(state.parent) - 1 for state in states)
            )
            initial_label_sizes = [
                vocab.fitting_size(codec.decode(label_id))
                for label_id in range(len(codec.id_to_token))
            ]
            numba_forest = NumbaTrainingForest.from_compact_states(
                states,
                label_capacity=len(codec.id_to_token) + max_possible_rules,
                initial_label_sizes=initial_label_sizes,
            )
            states.clear()
            label_counts: list[int] | None = None
            label_sizes: list[int] | None = None
        else:
            label_counts, label_sizes = _initial_label_statistics(states, codec, vocab)

        learned: list[EdgeBPERule] = []
        encoding_rules: list[EncodingRule] = []
        history: list[dict[str, Any]] = []
        rank = 0
        seed_merges = 0
        while self.num_merges is None or seed_merges < self.num_merges:
            if numba_forest is None:
                if label_counts is None or label_sizes is None:
                    raise RuntimeError("Python BPE fitting is missing label statistics.")
                selection = self._select_best_pair(counts, codec, label_counts, label_sizes)
            else:
                if self.pair_score_name_ is None:
                    raise RuntimeError("custom pair scorer reached Numba selection.")
                selection = numba_forest.select_best_pair(
                    self.min_pair_count,
                    codec,
                    score_mode=_NUMBA_PAIR_SCORE_MODES[self.pair_score_name_],
                )
            if selection is None:
                break

            greedy_child_id = selection.key[1]
            completed_seed = False
            while selection is not None:
                key = selection.key
                raw_count = selection.count
                parent_id, child_id = key
                parent_label = codec.decode(parent_id)
                child_label = codec.decode(child_id)
                token = edge_bpe_token(self.model_id, rank)

                vocab.add_fitting_size(token, selection.parent_size + selection.child_size)
                new_id = codec.intern(token)
                if numba_forest is None:
                    if label_counts is None or label_sizes is None:
                        raise RuntimeError("Python BPE fitting is missing label statistics.")
                    _set_new_label_statistics(
                        label_counts,
                        label_sizes,
                        label_id=new_id,
                        size=selection.parent_size + selection.child_size,
                    )
                    actual_events = sum(
                        state.contract_pair(key, new_label=new_id, pair_counts=counts)
                        for state in states
                    )
                    _update_label_counts_after_merge(
                        label_counts,
                        parent_id=parent_id,
                        child_id=child_id,
                        new_id=new_id,
                        events=actual_events,
                    )
                else:
                    numba_forest.register_label(
                        new_id,
                        size=selection.parent_size + selection.child_size,
                    )
                    actual_events = numba_forest.contract_pair(key, new_label=new_id)
                if actual_events == 0:
                    vocab.fitting_sizes.pop(token, None)
                    if numba_forest is None:
                        counts.pop(key, None)
                        break
                    raise RuntimeError(
                        "Numba pair index selected a positive-count label pair with no "
                        "contractible occurrence."
                    )

                learned.append(
                    EdgeBPERule(
                        rank=rank,
                        token=token,
                        parent_label=parent_label,
                        child_label=child_label,
                        count=raw_count,
                        score=selection.score,
                        parent_count=selection.parent_count,
                        child_count=selection.child_count,
                        parent_size=selection.parent_size,
                        child_size=selection.child_size,
                    )
                )
                encoding_rules.append(
                    EncodingRule(
                        rule_index=rank,
                        operation="edge",
                        output_label=token,
                        output_fitting_size=selection.parent_size + selection.child_size,
                        pattern={
                            "parent_label": parent_label,
                            "child_label": child_label,
                            "count_semantics": "raw_matching_edges",
                            "raw_count": raw_count,
                            "actual_events": actual_events,
                            "pair_score": self.pair_score_display_name_,
                            "parent_count": selection.parent_count,
                            "child_count": selection.child_count,
                        },
                        score=selection.score,
                    )
                )
                history.append(
                    {
                        "rank": rank,
                        "token": token,
                        "parent_label": parent_label,
                        "child_label": child_label,
                        "parent_token": parent_label,
                        "child_token": child_label,
                        "count": raw_count,
                        "count_semantics": "raw_matching_edges",
                        "parent_count": selection.parent_count,
                        "child_count": selection.child_count,
                        "parent_size": selection.parent_size,
                        "child_size": selection.child_size,
                        "score": selection.score,
                        "pair_score": self.pair_score_display_name_,
                        "actual_events": actual_events,
                    }
                )
                rank += 1
                completed_seed = True

                forced_key = (new_id, greedy_child_id)
                if numba_forest is None:
                    assert label_counts is not None and label_sizes is not None
                    forced_count = int(counts.get(forced_key, 0))
                    selection = self._pair_selection_for_key(
                        forced_key,
                        count=forced_count,
                        codec=codec,
                        label_counts=label_counts,
                        label_sizes=label_sizes,
                    )
                else:
                    bucket = numba_forest.pair_to_bucket.get(forced_key)
                    forced_count = 0 if bucket is None else int(numba_forest.bucket_count[bucket])
                    selection = self._pair_selection_for_key(
                        forced_key,
                        count=forced_count,
                        codec=codec,
                        label_counts=numba_forest.label_count,
                        label_sizes=numba_forest.label_size,
                    )

            if completed_seed:
                seed_merges += 1

        rules = tuple(encoding_rules)
        encoder = EdgeBPEEncoder(
            model_id=self.model_id,
            rules=rules,
            edge_rules=tuple(learned),
            input_labels=input_labels,
        )
        decoder = StructuralStageDecoder(model_id=self.model_id, rules=rules)
        self.backend_used_ = backend_used
        self.history_ = history
        return encoder, decoder


__all__ = ["GreedyBPECoarsener"]
