"""User-facing abstract coarsener API for schema 1.0."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Hashable, Sequence
from copy import deepcopy
from typing import TypeVar
from uuid import uuid4

import networkx as nx

from .decoder import TreeDecoder
from .encoder import TreeEncoder
from .exceptions import ConfigurationError, NotFittedError
from .schema import (
    ValidationLevel,
    fit_corpus_fitting_sizes,
    normalize_validation_level,
    prepare_graph,
)


_TreeCoarsenerT = TypeVar("_TreeCoarsenerT", bound="TreeCoarsener")


def _restore_snapshot_in_place(original: object, saved: object) -> bool:
    """Restore common mutable diagnostic values while preserving identity."""

    if type(original) is not type(saved):
        return False
    if isinstance(original, list):
        original[:] = deepcopy(saved)
        return True
    if isinstance(original, dict):
        original.clear()
        original.update(deepcopy(saved))
        return True
    if isinstance(original, set):
        original.clear()
        original.update(deepcopy(saved))
        return True
    if isinstance(original, bytearray):
        original[:] = saved
        return True
    if isinstance(original, deque):
        original.clear()
        original.extend(deepcopy(saved))
        return True
    if isinstance(original, tuple):
        # Tuple items cannot be rebound, but mutable diagnostics nested inside a
        # tuple can still be restored recursively.
        if len(original) != len(saved):
            return False
        restored_any = False
        for original_item, saved_item in zip(original, saved, strict=True):
            restored_any = _restore_snapshot_in_place(original_item, saved_item) or restored_any
        return restored_any or original == saved
    if hasattr(original, "__dict__") and hasattr(saved, "__dict__"):
        vars(original).clear()
        vars(original).update(deepcopy(vars(saved)))
        return True
    return original == saved


class TreeCoarsener(ABC):
    """Base class that centralizes fit atomicity and public API behavior."""

    encoder_: TreeEncoder | None
    decoder_: TreeDecoder | None

    def __init__(self, *, model_id: str | None = None) -> None:
        if model_id is not None and (not isinstance(model_id, str) or not model_id):
            raise ConfigurationError("model_id must be a nonempty string or None.")
        self.model_id = model_id or f"{self.__class__.__name__}:{uuid4().hex}"
        self.encoder_ = None
        self.decoder_ = None

    def fit(
        self: _TreeCoarsenerT,
        graphs: Sequence[nx.DiGraph],
        *,
        validate: ValidationLevel = "full",
    ) -> _TreeCoarsenerT:
        state_before = self._snapshot_fit_state()
        level = normalize_validation_level(validate)
        try:
            if isinstance(graphs, nx.DiGraph) or isinstance(graphs, (str, bytes, bytearray)):
                raise TypeError("fit expects a nonempty sequence of DiGraphs, such as [graph].")
            try:
                graph_list = list(graphs)
            except TypeError as exc:
                raise TypeError("fit expects a nonempty sequence of DiGraphs.") from exc
            if not graph_list:
                raise ValueError("fit requires at least one graph.")
            if any(
                not isinstance(graph, nx.DiGraph) or graph.is_multigraph() for graph in graph_list
            ):
                raise TypeError("every fit input must be a non-multigraph networkx.DiGraph.")

            prepared = tuple(prepare_graph(graph, validate=level) for graph in graph_list)
            # This is a corpus-level schema invariant, not a method-specific
            # concern. Keeping it here avoids burdening every simple coarsener.
            fit_corpus_fitting_sizes(prepared)
            encoder, decoder = self._fit(prepared)
            self._validate_artifact_pair(encoder, decoder)
            self.encoder_ = encoder
            self.decoder_ = decoder
        except Exception:
            self._restore_fit_state(state_before)
            raise
        return self

    def _snapshot_fit_state(
        self,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Capture fitted state and public diagnostics for atomic refits.

        Fitted artifacts are retained by identity. Conventional public fitted
        diagnostics ending in ``_`` are deep-copied so both reassignment and
        in-place mutation can be rolled back.
        """

        state = dict(vars(self))
        diagnostic_values: dict[str, object] = {}
        for name, value in state.items():
            if name in {"encoder_", "decoder_"} or not name.endswith("_"):
                continue
            try:
                diagnostic_values[name] = deepcopy(value)
            except Exception:
                # Uncopyable custom diagnostics can still be restored after
                # reassignment through the shallow state snapshot. Their own
                # in-place mutation is outside what Python can generically undo.
                continue
        return state, diagnostic_values

    def _restore_fit_state(
        self,
        snapshot: tuple[dict[str, object], dict[str, object]],
    ) -> None:
        state, diagnostic_values = snapshot
        vars(self).clear()
        vars(self).update(state)
        for name, saved_value in diagnostic_values.items():
            original = state[name]
            if not _restore_snapshot_in_place(original, saved_value):
                setattr(self, name, saved_value)

    def transform(
        self,
        graph: nx.DiGraph,
        *,
        validate: ValidationLevel = "full",
    ) -> nx.DiGraph:
        if self.encoder_ is None:
            raise NotFittedError("call fit before transform.")
        return self.encoder_.transform(graph, validate=validate)

    def fit_transform(
        self,
        graphs: Sequence[nx.DiGraph],
        *,
        validate: ValidationLevel = "full",
    ) -> list[nx.DiGraph]:
        state_before = self._snapshot_fit_state()
        try:
            if isinstance(graphs, nx.DiGraph) or isinstance(graphs, (str, bytes, bytearray)):
                raise TypeError("fit_transform expects a nonempty sequence of DiGraphs.")
            graph_list = list(graphs)
            self.fit(graph_list, validate=validate)
            return [self.transform(graph, validate=validate) for graph in graph_list]
        except Exception:
            self._restore_fit_state(state_before)
            raise

    def decode(
        self,
        graph: nx.DiGraph,
        *,
        target: Hashable | None = None,
        by: str = "node",
        recursive: bool = True,
        boundary_policy: str = "expand",
        validate: ValidationLevel = "full",
    ) -> nx.DiGraph:
        if self.decoder_ is None:
            raise NotFittedError("call fit before decode.")
        return self.decoder_.decode(
            graph,
            target=target,
            by=by,
            recursive=recursive,
            boundary_policy=boundary_policy,
            validate=validate,
        )

    def inverse_transform(
        self,
        graph: nx.DiGraph,
        *,
        target: Hashable | None = None,
        by: str = "node",
        recursive: bool = True,
        boundary_policy: str = "expand",
        validate: ValidationLevel = "full",
    ) -> nx.DiGraph:
        return self.decode(
            graph,
            target=target,
            by=by,
            recursive=recursive,
            boundary_policy=boundary_policy,
            validate=validate,
        )

    def _validate_artifact_pair(self, encoder: TreeEncoder, decoder: TreeDecoder) -> None:
        if not isinstance(encoder, TreeEncoder) or not isinstance(decoder, TreeDecoder):
            raise ConfigurationError("_fit must return (TreeEncoder, TreeDecoder).")
        if encoder.model_id != decoder.model_id or encoder.model_id != self.model_id:
            raise ConfigurationError("coarsener, encoder, and decoder model IDs must agree.")
        if encoder.rules != decoder.rules:
            raise ConfigurationError("paired encoder and decoder rules must be identical.")
        if encoder.vocab != decoder.vocab:
            raise ConfigurationError("paired encoder and decoder vocabularies must be identical.")

    @abstractmethod
    def _fit(
        self,
        graphs: Sequence[nx.DiGraph],
    ) -> tuple[TreeEncoder, TreeDecoder]:
        """Construct new fitted artifacts without mutating existing state."""
