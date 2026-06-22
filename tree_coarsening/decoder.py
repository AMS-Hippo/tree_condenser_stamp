"""Decoder artifact interface for schema 1.0."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence

import networkx as nx

from .encoder import EncodingRule, Vocabulary, _WriteOnceArtifact
from .exceptions import ConfigurationError


class TreeDecoder(_WriteOnceArtifact, ABC):
    """Fitted schema-1 decoder artifact."""

    def __init__(self, *, model_id: str, rules: Sequence[EncodingRule]) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ConfigurationError("model_id must be a nonempty string.")
        self._model_id = model_id
        self._rules = tuple(rules)
        self._vocab = Vocabulary(self._rules)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def rules(self) -> tuple[EncodingRule, ...]:
        return self._rules

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @abstractmethod
    def decode(
        self,
        graph: nx.DiGraph,
        *,
        target: Hashable | None = None,
        by: str = "node",
        recursive: bool = True,
        boundary_policy: str = "expand",
        validate: str | bool = "full",
    ) -> nx.DiGraph:
        """Reverse this stage fully or partially without mutating *graph*."""
