from __future__ import annotations

from collections.abc import Iterable, Sequence

import networkx as nx
import pytest

from tree_coarsening import (
    EncodingRule,
    RuleBasedEncoder,
    StructuralStageDecoder,
    TreeCoarsener,
    TreeDecoder,
    TreeEncoder,
)

from conftest import make_tree


class _NoOpEncoder(RuleBasedEncoder):
    def select_contractions(
        self,
        graph: nx.DiGraph,
        rule: EncodingRule,
    ) -> Sequence[Iterable[object]]:
        del graph, rule
        return ()


class _DiagnosticCoarsener(TreeCoarsener):
    def __init__(self, *, model_id: str) -> None:
        super().__init__(model_id=model_id)
        self.diagnostic_: list[str] = ["initial"]
        self.fail = False

    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        del graphs
        self.diagnostic_ = ["new"]
        if self.fail:
            raise RuntimeError("deliberate fit failure")
        rule = EncodingRule(0, "noop", ("noop", self.model_id), 1)
        rules = (rule,)
        return (
            _NoOpEncoder(model_id=self.model_id, rules=rules),
            StructuralStageDecoder(model_id=self.model_id, rules=rules),
        )


def test_failed_refit_restores_artifacts_and_public_diagnostics() -> None:
    graph = make_tree(["A"], [None], prefix="atomic")
    model = _DiagnosticCoarsener(model_id="atomic-stage").fit([graph])
    encoder_before = model.encoder_
    decoder_before = model.decoder_
    diagnostic_before = model.diagnostic_

    model.fail = True
    with pytest.raises(RuntimeError, match="deliberate"):
        model.fit([graph])

    assert model.encoder_ is encoder_before
    assert model.decoder_ is decoder_before
    assert model.diagnostic_ is diagnostic_before
    assert model.diagnostic_ == ["new"]


def test_invalid_refit_input_preserves_previous_fit() -> None:
    graph = make_tree(["A"], [None], prefix="valid")
    model = _DiagnosticCoarsener(model_id="input-stage").fit([graph])
    encoder_before = model.encoder_
    decoder_before = model.decoder_
    diagnostic_before = model.diagnostic_

    bad = nx.DiGraph()
    with pytest.raises(Exception):
        model.fit([bad])

    assert model.encoder_ is encoder_before
    assert model.decoder_ is decoder_before
    assert model.diagnostic_ is diagnostic_before


class _NestedDiagnosticCoarsener(_DiagnosticCoarsener):
    def __init__(self, *, model_id: str) -> None:
        super().__init__(model_id=model_id)
        self.nested_diagnostic_: tuple[list[str], dict[str, list[int]]] = (
            ["stable"],
            {"values": [1, 2]},
        )

    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        if self.fail:
            self.nested_diagnostic_[0].append("mutated")
            self.nested_diagnostic_[1]["values"].append(3)
            raise RuntimeError("nested diagnostic failure")
        return super()._fit(graphs)


def test_failed_refit_restores_nested_mutable_diagnostics() -> None:
    graph = make_tree(["A"], [None], prefix="nested-atomic")
    model = _NestedDiagnosticCoarsener(model_id="nested-atomic-stage").fit([graph])
    diagnostic_before = model.nested_diagnostic_
    list_before = diagnostic_before[0]
    dict_before = diagnostic_before[1]

    model.fail = True
    with pytest.raises(RuntimeError, match="nested diagnostic"):
        model.fit([graph])

    assert model.nested_diagnostic_ is diagnostic_before
    assert model.nested_diagnostic_[0] is list_before
    assert model.nested_diagnostic_[1] is dict_before
    assert model.nested_diagnostic_ == (["stable"], {"values": [1, 2]})
